import tensorflow as tf
from PIL import Image
import numpy as np

import os
import io

colors = tf.constant([[[[0, 0, 0], [255, 255, 0], [0, 255, 255], [0, 255, 0],
                      [255, 255, 255], [0, 0, 255], [255, 0, 255]]]], dtype='float32')


def convert_to_rgb(prediction): # (306, 306, 7)
    shape = prediction.shape
    pred = tf.reshape(prediction, [shape[0], shape[1], shape[2], 1])
    pred = pred * colors # (306, 306, 7, 3)
    pred = tf.reduce_sum(pred, axis=2)
    return pred


def generate_image(img_arr):
    img = img_arr.astype(np.uint8)
    assert len(img.shape) ==3, 'Input shall be HxWx3 '
    assert img.shape[2] == 3, 'Input shall have RGB in the last dim'
    generated = Image.fromarray(img)
    return generated


def reconstruct_raw_image(img_arr):
    img = img_arr * 255
    img = img.astype(np.uint8)
    generated = Image.fromarray(img)
    return generated


def construct_predicted_label_batch(itr, ids, query_indices_batch, label_query_batch, pred_batch, job_dir, storage_client):
    batch = ids.shape[0]
    bucket = storage_client.bucket('deepglobe_cs330_1')
    for b in range(batch):
        id = ids[b].numpy()
        query_indices = query_indices_batch[b]
        label_query = label_query_batch[b]
        pred = pred_batch[b]
        construct_predicted_label(itr, id, query_indices, label_query, pred, bucket)


def construct_predicted_label(itr, id, query_indices, label_query, pred, bucket):
    label_query_np = label_query.numpy()
    pred_np = pred.numpy()
    label_reconstruct_tiles = np.zeros([64, 306, 306, 7])
    pred_reconstruct_tiles = np.zeros([64, 306, 306, 7])

    for q in range(query_indices.shape[0]):
        i = query_indices[q]
        label_reconstruct_tiles[i] = label_query_np[q]
        pred_reconstruct_tiles[i] = pred_np[q]

    label_reconstruct = tf.reshape(label_reconstruct_tiles, [1, 8, 8, 306*306, 7])
    label_reconstruct = tf.stack(tf.split(label_reconstruct, 306*306, 3), axis=0)
    label_reconstruct = tf.reshape(label_reconstruct, [306*306, 8, 8, 7])
    label_reconstruct = tf.batch_to_space(label_reconstruct, [306, 306], [[0, 0], [0, 0]])
    label_reconstruct = tf.reshape(label_reconstruct, [2448,2448,7])

    label_rgb = convert_to_rgb(tf.cast(label_reconstruct, dtype=tf.float32))
    label_result = generate_image(label_rgb.numpy())
    dir = 'results_inner_update_4/' + str(itr)

    path = os.path.join(dir, str(int(id)) + '_label.png')
    label_img_byte_array = io.BytesIO()
    label_result.save(label_img_byte_array, format='PNG')
    blob = bucket.blob(path)
    blob.upload_from_string(label_img_byte_array.getvalue(), content_type="image/png")

    pred_reconstruct = tf.reshape(pred_reconstruct_tiles, [1, 8, 8, 306*306, 7])
    pred_reconstruct = tf.stack(tf.split(pred_reconstruct, 306*306, 3), axis=0)
    pred_reconstruct = tf.reshape(pred_reconstruct, [306*306, 8, 8, 7])
    pred_reconstruct = tf.batch_to_space(pred_reconstruct, [306, 306], [[0, 0], [0, 0]])
    pred_reconstruct = tf.reshape(pred_reconstruct, [2448,2448,7])

    pred_rgb = convert_to_rgb(tf.cast(pred_reconstruct, dtype=tf.float32))
    pred_result = generate_image(pred_rgb.numpy())
    path = os.path.join(dir, str(int(id)) + '_pred.png')
    pred_img_byte_array = io.BytesIO()
    pred_result.save(pred_img_byte_array, format='PNG')
    blob = bucket.blob(path)
    blob.upload_from_string(pred_img_byte_array.getvalue(), content_type="image/png")

