import os

from google.cloud import storage
from tensorflow.python.platform import flags
import tensorflow as tf
from model.image_constructor import convert_to_rgb, generate_image, reconstruct_raw_image

FLAGS = flags.FLAGS


class DataGenerator(object):
    RAW_IMAGE_SIZE = (2448, 2448)
    ONE_SHOT_SIZE = (306, 306)
    TOTAL_TILES = 64
    IMAGE_SUFFIX = '_sat.jpg'
    LABEL_SUFFIX = '_mask.png'

    LABELS = [[0, 0, 0], [255, 255, 0], [0, 255, 255], [0, 255, 0],
              [255, 255, 255], [0, 0, 255], [255, 0, 255]]
    LABEL_SIZE = 7

    """
    Data generator generates batches of DeepGlobe image data and the label for each image
    """
    def __init__(self, k_support, k_query, batch_size, batch_type, job_dir):
        """
        Args:
            k_support: the size of the support size. From 1 to 32
            k_query: the size of the query size. From 1 to 32
            batch_size: size of meta batch size
            batch_type: meta_train/meta_val/meta_test
        """
        self.k_support = k_support
        self.k_query = k_query
        self.batch_size = batch_size
        self.storage_client = storage.Client()
        if batch_type == "meta_train":
            self.folder = 'dataset/land-train'
        elif batch_type == "meta_val":
            self.folder = 'dataset/land-valid'
        else:
            self.folder = 'dataset/land-test'

        self.path = os.path.join(job_dir, self.folder)
        self.exp_ids = [self._extract_id(blob.name) for blob in self.storage_client.list_blobs('deepglobe_cs330_1', prefix=self.folder)]
        self.exp_ids = list(filter(None, list(set(self.exp_ids))))

    def _extract_id(self, file_path):
        filename = file_path.split(os.sep)[-1]
        if filename.endswith(self.IMAGE_SUFFIX):
            return filename[:-len(self.IMAGE_SUFFIX)]
        if filename.endswith(self.LABEL_SUFFIX):
            return filename[:-len(self.LABEL_SUFFIX)]
        return ""

    def create_dataset(self):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
        Returns:
          A a tuple of (1) Image batch and (2) Label batch where
          image batch has shape [B, K, 306, 306, 3] and label batch has shape
          [B, K, 306, 306, 1] where B is batch size, K is number of samples per
          class (i.e. k_support+k_query)
        """

        # expnames are the filenames without extension.
        image_label_dataset = (tf.data.Dataset.from_tensor_slices(self.exp_ids)
                               .map(self._parse_fn, num_parallel_calls=4).repeat().shuffle(10).batch(self.batch_size))
        return image_label_dataset

    def sample_batch(self):
        batch = (tf.data.Dataset.from_tensor_slices(self.exp_ids)
                 .map(self._parse_fn, num_parallel_calls=4).shuffle(10).repeat().batch(self.batch_size)).take(1)
        return batch

    def _parse_fn(self, id):
        """
        parse each record, sample tiles based on k_support and k_query.

        return: support_image, support_label, query_image, query_label
        """
        dtype = tf.float16
        image_filename = id + self.IMAGE_SUFFIX
        image_path = self.path + os.sep + image_filename

        label_filename = id + self.LABEL_SUFFIX
        label_path = self.path + os.sep + label_filename

        image = tf.io.read_file(image_path)
        # with tf.io.gfile.GFile(image_path, "r") as reader:
        #     image = reader.read()
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, dtype) / 255.0

        image = tf.expand_dims(image, axis=0)
        image.set_shape([1, self.RAW_IMAGE_SIZE[0], self.RAW_IMAGE_SIZE[1], 3])

        patches = tf.image.extract_patches(images=image, sizes=[1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], 1], strides=[1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], 1], rates=[1, 1, 1, 1], padding='VALID')
        image = tf.reshape(patches, [-1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], 3])

        label = tf.io.read_file(label_path)
        # with tf.io.gfile.GFile(label_path, "r") as reader:
        #     label = reader.read()
        label = tf.image.decode_jpeg(label)

        mask = tf.zeros([self.RAW_IMAGE_SIZE[0], self.RAW_IMAGE_SIZE[1]], tf.uint8)
        for i, l in enumerate(self.LABELS):
            mask += tf.cast(tf.reduce_all(tf.equal(label, [[l]]), -1), tf.uint8) * i # (RAW_IMAGE_SIZE[0], RAW_IMAGE_SIZE[1])
        one_hot_label = tf.one_hot(mask, self.LABEL_SIZE, dtype=dtype)
        one_hot_label = tf.expand_dims(one_hot_label, axis=0)
        one_hot_label.set_shape([1, self.RAW_IMAGE_SIZE[0], self.RAW_IMAGE_SIZE[1], self.LABEL_SIZE])

        label_patches = tf.image.extract_patches(images=one_hot_label, sizes=[1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], 1], strides=[1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], 1], rates=[1, 1, 1, 1], padding='VALID')
        one_hot_label = tf.reshape(label_patches, [-1, self.ONE_SHOT_SIZE[0], self.ONE_SHOT_SIZE[1], self.LABEL_SIZE])

        indices = tf.random.shuffle(tf.range(self.TOTAL_TILES))
        support_indices = indices[:self.k_support]
        query_indices = indices[self.k_support:self.k_support+self.k_query]

        support_image = tf.gather(image, support_indices)
        support_label = tf.gather(one_hot_label, support_indices)

        query_image = tf.gather(image, query_indices)
        query_label = tf.gather(one_hot_label, query_indices)
        return support_image, query_image, support_label, query_label, id, query_indices


def main():
    dg = DataGenerator(0, 3, 1, 'meta_train')
    for support_image, query_image, support_label, query_label, label_path in dg.create_dataset().take(1):
        first_label = query_label[0,-1,:,:,:]
        first_rgb_label = convert_to_rgb(first_label)
        print(first_rgb_label.numpy().dtype)
        first_img_label = generate_image(first_rgb_label.numpy())
        first_img_label.save("label.png")

        first_input = query_image[0,-1,:,:,:]
        first_img_input = reconstruct_raw_image(first_input.numpy())
        first_img_input.save("input.png")

        print(label_path)


if __name__ == "__main__":
    main()
