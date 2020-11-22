import tensorflow as tf
import logging
from tensorflow.python.platform import flags
from model.maml import training

FLAGS = flags.FLAGS

flags.DEFINE_integer('meta_train_iterations', 1, 'number of metatraining iterations')
flags.DEFINE_integer('meta_batch_size', 1, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('k_support', 8, 'k shot for support set')
flags.DEFINE_integer('k_query', 32, 'k shot for query set')
flags.DEFINE_integer('num_inner_updates', 5, 'number of inner updates for inner loop')
flags.DEFINE_bool('learn_inner_update_lr', False, 'to learn inner update learning rate or not')
flags.DEFINE_float('inner_update_lr', 0.01, 'learning rate for inner loop')
flags.DEFINE_float('meta_lr', 0.001, 'meta learning rate')
flags.DEFINE_string('job-dir', '.', 'Location to store output artifacts for training job')

def main():
    logging.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    logging.info("GPU NAME:", tf.test.gpu_device_name())
    if not tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        logging.info("GPU Not available")
    else:
        logging.info("GPU available")
    training(FLAGS.meta_train_iterations, FLAGS.meta_batch_size, FLAGS.k_support, FLAGS.k_query, FLAGS.num_inner_updates, FLAGS.inner_update_lr, FLAGS.learn_inner_update_lr, FLAGS.meta_lr, FLAGS['job-dir'].value)


if __name__ == "__main__":
    main()