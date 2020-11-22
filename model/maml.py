import os
import logging

import tensorflow as tf
from model.layers import UNet
from model.data_generator import DataGenerator
from model.metric import compute_mIoU
from model.image_constructor import construct_predicted_label_batch
from google.cloud import storage
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def inner_cross_entropy_loss(pred, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)))

def outer_cross_entropy_loss(pred, label, num_replicas):
    per_example_losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)))
    return per_example_losses / num_replicas

class MAML(tf.keras.Model):
    def __init__(self, k_support, k_query, num_inner_updates=1, inner_update_lr=0.4, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.inner_update_lr = inner_update_lr
        self.learn_inner_update_lr = learn_inner_update_lr
        self.model_layers = UNet(channels=3, fms=16)
        self.num_inner_updates = num_inner_updates
        self.k_support = k_support
        self.k_query = k_query
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.model_layers.model_weights.keys():
                self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j), dtype=tf.float32) for j in range(num_inner_updates)]

    @tf.function
    def call(self, inp, num_replicas, optim, isTraining):
        def inner_loop(inp): # inner_loop for single task
            input_support, input_query, label_support, label_query = inp
            weights = self.model_layers.model_weights

            with tf.GradientTape() as tape:
                output_support = self.model_layers(input_support, weights, isTraining)
                loss_support = inner_cross_entropy_loss(output_support, label_support)
                scaled_loss_support = optim.get_scaled_loss(loss_support)
            scaled_grads = tape.gradient(scaled_loss_support, list(weights.values()))
            grads = optim.get_unscaled_gradients(scaled_grads)
            gradients = dict(zip(weights.keys(), grads))

            if self.learn_inner_update_lr:
                fast_weights = dict(zip(weights.keys(), [weights[key] - tf.cast(self.inner_update_lr_dict[key][0]*tf.cast(gradients[key], dtype=tf.float32), dtype=tf.float16) for key in weights.keys()]))

            else:
                fast_weights = dict(zip(weights.keys(), [weights[key] - tf.cast(self.inner_update_lr*gradients[key], dtype=tf.float16) for key in weights.keys()]))

            for j in range(self.num_inner_updates-1):
                with tf.GradientTape() as tape:
                    tape.watch(fast_weights)
                    output_support = self.model_layers(input_support, fast_weights, isTraining)
                    loss_support = inner_cross_entropy_loss(output_support, label_support)
                    scaled_loss_support = optim.get_scaled_loss(loss_support)
                scaled_grads = tape.gradient(scaled_loss_support, list(fast_weights.values()))
                grads = optim.get_unscaled_gradients(scaled_grads)

                gradients = dict(zip(fast_weights.keys(), grads))
                if self.learn_inner_update_lr:
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - tf.cast(self.inner_update_lr_dict[key][j+1]*tf.cast(gradients[key], dtype=tf.float32), dtype=tf.float16) for key in fast_weights.keys()]))
                else:
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - tf.cast(self.inner_update_lr*gradients[key], dtype=tf.float16) for key in fast_weights.keys()]))

            output_query = self.model_layers(input_query, fast_weights, isTraining)
            loss_query = outer_cross_entropy_loss(output_query, label_query, num_replicas)
            # TODO: add evaluation method. e.g. accuracy...
            return output_query, loss_query

        input_support, input_query, label_support, label_query = inp
        batch_per_replica = input_support.shape[0]
        unused = inner_loop((input_support[0], input_query[0], label_support[0], label_query[0]))
        result = tf.map_fn(inner_loop,
                           elems=(input_support, input_query, label_support, label_query),
                           parallel_iterations=batch_per_replica,
                           dtype=(tf.float32, tf.float32))
        return result


def outer_train_step(inp, maml, num_replicas, optim):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = maml(inp, num_replicas, optim, True)
        per_replica_output_query, all_tasks_loss_query_per_replica = result # all tasks in a batch
        meta_loss_per_replica = tf.reduce_mean(all_tasks_loss_query_per_replica)
        scaled_meta_loss_per_replica = optim.get_scaled_loss(meta_loss_per_replica)
    scaled_gradients = outer_tape.gradient(scaled_meta_loss_per_replica, maml.trainable_variables)
    gradients = optim.get_unscaled_gradients(scaled_gradients)
    optim.apply_gradients(zip(gradients, maml.trainable_variables))
    return per_replica_output_query, meta_loss_per_replica


def outer_valid_step(inp, maml, num_replicas, optim):
    result = maml(inp, num_replicas, optim, False)
    per_replica_output_query, all_tasks_loss_query_per_replica = result # all tasks in a batch
    meta_loss_per_replica = tf.reduce_mean(all_tasks_loss_query_per_replica)
    return per_replica_output_query, meta_loss_per_replica


@tf.function
def distributed_train_step(inp, maml, num_replicas, optim, mirrored_strategy):
    output_query, per_replica_meta_losses = mirrored_strategy.run(outer_train_step, args=(inp, maml, num_replicas, optim))
    meta_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_meta_losses,
                                         axis=None)

    return output_query, meta_loss


@tf.function
def distributed_valid_step(inp, maml, num_replicas, optim, mirrored_strategy):
    output_query, per_replica_meta_losses = mirrored_strategy.run(outer_valid_step, args=(inp, maml, num_replicas, optim))
    meta_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_meta_losses,
                                         axis=None)
    return output_query, meta_loss


def training(meta_train_iterations, meta_batch_size, k_support, k_query, num_inner_updates, inner_update_lr, learn_inner_update_lr, meta_lr, job_dir):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    data_generator = DataGenerator(k_support, k_query, meta_batch_size, 'meta_train', job_dir)
    data_generator_valid = DataGenerator(2, 32, meta_batch_size, 'meta_val', job_dir)

    itr = 0
    meta_loss_log_dir_2 = os.path.join(job_dir, 'summary_6_intrain_2_intest/meta_loss')
    meta_metric_log_dir_2 = os.path.join(job_dir, 'summary_6_intrain_2_intest/meta_metric')
    meta_loss_summary_writer = tf.summary.create_file_writer(meta_loss_log_dir_2)
    meta_metric_writer = tf.summary.create_file_writer(meta_metric_log_dir_2)

    with mirrored_strategy.scope():
        maml = MAML(k_support, k_query, num_inner_updates=num_inner_updates,
                    inner_update_lr=inner_update_lr,
                    learn_inner_update_lr=learn_inner_update_lr)
        optim = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        optim = mixed_precision.LossScaleOptimizer(optim, loss_scale='dynamic')

    storage_client = storage.Client()
    acc_metric = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

    dataset = data_generator.create_dataset().take(meta_train_iterations)
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    num_replicas = mirrored_strategy.num_replicas_in_sync
    logging.info('mirrored_strategy.num_replicas_in_sync: %d' % (num_replicas))

    best_evel_mIoU = 0
    model_exp_str = 'mbs_'+str(meta_batch_size)+'.k_support_'+str(k_support)+'.k_query_'+str(k_query)+'.inner_steps_'+str(num_inner_updates)+'.inner_lr_'+str(inner_update_lr)+'.learn_inner_update_lr_'+str(learn_inner_update_lr)+'.meta_lr_'+str(meta_lr)
    model_file = os.path.join(job_dir, 'weights_inner_update_4', model_exp_str)

    for input_support_replica, input_query_replica, label_support_replica, label_query_replica, ids_replica, query_indices_replica in dist_dataset:
        itr = itr+1
        inp = (input_support_replica, input_query_replica, label_support_replica, label_query_replica)
        output_query_replicas, meta_loss = distributed_train_step(inp, maml, num_replicas, optim, mirrored_strategy)
        logging.info('Iteration %d: meta loss: %.5f ' % (itr, meta_loss))
        if itr % 1 == 0:
            if num_replicas > 1:
                output_query = output_query_replicas.values
                output_query = tf.concat(output_query, 0)

                label_query = label_query_replica.values
                label_query = tf.concat(label_query, 0)
            else:
                output_query = output_query_replicas
                label_query = label_query_replica
            label_query = tf.cast(label_query, dtype=tf.float32)
            pred = tf.one_hot(tf.argmax(output_query, axis=-1), depth=data_generator.LABEL_SIZE)
            with tf.device('/CPU:0'):
                mIoU = compute_mIoU(label_query[:,:,:,:,1:], pred[:,:,:,:,1:], data_generator.LABEL_SIZE-1)
            logging.info('Iteration %d: mean IoU: %.5f ' % (itr, mIoU))

            with tf.device('/CPU:0'):
                acc_metric.update_state(label_query[:,:,:,:,1:], tf.math.softmax(output_query)[:,:,:,:,1:])
                acc = acc_metric.result()
            logging.info('Iteration %d: accuracy: %.5f ' % (itr, acc))
            acc_metric.reset_states()

            with meta_loss_summary_writer.as_default():
                tf.summary.scalar('train-meta-loss', meta_loss, step=itr)
            with meta_metric_writer.as_default():
                tf.summary.scalar('train mean IoU', mIoU, step=itr)
            with meta_metric_writer.as_default():
                tf.summary.scalar('train accuracy', acc, step=itr)

        # evaluation session
        if itr % 150 == 0:
            valid_set = data_generator_valid.sample_batch() # only one batch, size of meta_batch_size
            dist_valid_dataset_single_elem = mirrored_strategy.experimental_distribute_dataset(valid_set)
            for input_support_val_replica, input_query_val_replica, label_support_val_replica, label_query_val_replica, ids_val_replica, query_indices_val_replica in dist_valid_dataset_single_elem: # only one elem in the dataset
                inp_valid = (input_support_val_replica, input_query_val_replica, label_support_val_replica, label_query_val_replica)
                output_query_valid_replicas, meta_loss_valid = distributed_valid_step(inp_valid, maml, num_replicas, optim, mirrored_strategy)
                logging.info('[VALIDATION] Iteration %d: meta loss: %.5f ' % (itr, meta_loss_valid))
                if num_replicas > 1:
                    output_query_valid = output_query_valid_replicas.values
                    output_query_valid = tf.concat(output_query_valid, 0)

                    label_query_valid = label_query_val_replica.values
                    label_query_valid = tf.concat(label_query_valid, 0)

                    ids_valid = tf.concat(ids_val_replica.values, 0)
                    query_indices_valid = tf.concat(query_indices_val_replica.values, 0)
                else:
                    output_query_valid = output_query_valid_replicas
                    label_query_valid = label_query_val_replica
                    ids_valid = ids_val_replica
                    query_indices_valid = query_indices_val_replica
                label_query_valid = tf.cast(label_query_valid, dtype=tf.float32)
                pred_valid = tf.one_hot(tf.argmax(output_query_valid, axis=-1), depth=data_generator.LABEL_SIZE)

                with tf.device('/CPU:0'):
                    mIoU_valid = compute_mIoU(label_query_valid[:,:,:,:,1:], pred_valid[:,:,:,:,1:], data_generator.LABEL_SIZE-1)
                logging.info('[VALIDATION] Iteration %d: mean IoU: %.5f ' % (itr, mIoU_valid))

                if mIoU_valid > best_evel_mIoU:
                    best_evel_mIoU = mIoU_valid
                    logging.info("saving to ",  model_file)
                    maml.save_weights(model_file)

                with tf.device('/CPU:0'):
                    acc_metric.update_state(label_query_valid[:,:,:,:,1:], tf.math.softmax(output_query_valid)[:,:,:,:,1:])
                    acc_valid = acc_metric.result()
                logging.info('[VALIDATION] Iteration %d: accuracy: %.5f ' % (itr, acc_valid))
                acc_metric.reset_states()

                with meta_metric_writer.as_default():
                    tf.summary.scalar('eval mean IoU', mIoU_valid, step=itr)
                with meta_metric_writer.as_default():
                    tf.summary.scalar('eval accuracy', acc_valid, step=itr)

                with tf.device('/CPU:0'):
                    construct_predicted_label_batch(itr, ids_valid, query_indices_valid, label_query_valid, pred_valid, job_dir, storage_client)

                with meta_loss_summary_writer.as_default():
                    tf.summary.scalar('eval-meta-loss', meta_loss_valid, step=itr)
