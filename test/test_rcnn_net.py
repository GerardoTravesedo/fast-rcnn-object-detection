import numpy as np
import tensorflow as tf

import learning_rate_manager as rm
import detection.rcnn_net as rcnn_net
import dataset.dataset_reader as reader


class TestRoiPoolingLayer(tf.test.TestCase):

    def test_net(self):
        with self.test_session() as sess:
            image_input_batch = tf.placeholder(tf.float32, shape=(1, 600, 600, 3))
            roi_input_batch = tf.placeholder(tf.float32, shape=(None, 4))
            class_label_batch = tf.placeholder(tf.float32, shape=(None, 21))
            detection_label_batch = tf.placeholder(tf.float32, shape=(None, 4))
            learning_rate = tf.placeholder(tf.float32, name="LearningRate")

            target = rcnn_net.get_net(
                21, 4, 15, 400, image_input_batch, roi_input_batch, class_label_batch,
                detection_label_batch, learning_rate)

            sess.run(tf.global_variables_initializer())

            # Testing integration with data reader
            training_reader = reader.DatasetReader(
                ["test/data/test-batch-reader-dataset/batch/rcnn_dataset_1"], 1, 64, 16)
            training_batch = training_reader.get_batch()

            class_label_batch_test = np.zeros((64, 21))
            detection_label_batch_test = np.zeros((64, 4))

            learning_rate_manager = rm.LearningRateManager(0.001, 0.6, 80)

            result_loss, result_training, result_test = sess.run(target, feed_dict={
                image_input_batch: training_batch["images"],
                roi_input_batch: training_batch["rois"],
                class_label_batch: class_label_batch_test,
                detection_label_batch: detection_label_batch_test,
                learning_rate: learning_rate_manager.learning_rate
            })

            assert result_loss.shape == (64, 1)
            assert result_test.shape[1] == 5

    def test_get_base_net(self):
        with self.test_session() as sess:
            image_input_batch = tf.placeholder(tf.float32, shape=(1, 600, 600, 3))
            roi_input_batch = tf.placeholder(tf.float32, shape=(None, 4))

            he_init = tf.contrib.layers.variance_scaling_initializer()

            target = rcnn_net.get_base_net(15, 400, image_input_batch, roi_input_batch, he_init)

            # Initialization has to happen after defining the graph
            sess.run(tf.global_variables_initializer())

            # Testing integration with data reader
            training_reader = reader.DatasetReader(
                ["test/data/test-batch-reader-dataset/batch/rcnn_dataset_1"], 1, 64, 16)
            training_batch = training_reader.get_batch()

            result_last_op, result_pool_layer = sess.run(target, feed_dict={
                image_input_batch: training_batch["images"],
                roi_input_batch: training_batch["rois"]
            })

            assert result_last_op.shape == (64, 400)
            # 64 rois, each with a 7x7x64 pooling output
            assert result_pool_layer.shape == (64, 7, 7, 64)

    def test_get_classification_branch(self):
        with self.test_session() as sess:
            he_init = tf.contrib.layers.variance_scaling_initializer()
            previous_output = tf.placeholder(tf.float32, shape=(None, 7 * 7 * 64))

            target = rcnn_net.get_classification_branch(21, previous_output, he_init)

            sess.run(tf.global_variables_initializer())

            # 4 rois with all zeros
            previous_output_test = np.zeros((64, 7 * 7 * 64))

            result = sess.run(target, feed_dict={
                previous_output: previous_output_test
            })

            assert result.shape == (64, 21)

    def test_get_detection_branch(self):
        with self.test_session() as sess:
            he_init = tf.contrib.layers.variance_scaling_initializer()
            previous_output = tf.placeholder(tf.float32, shape=(None, 7 * 7 * 64))

            target = rcnn_net.get_detection_branch(21, 4, previous_output, he_init)

            sess.run(tf.global_variables_initializer())

            # 4 rois with all zeros
            previous_output_test = np.zeros((64, 7 * 7 * 64))

            result = sess.run(target, feed_dict={
                previous_output: previous_output_test
            })

            # 64 rois, each with 20 groups of 4 regression targets
            assert result.shape == (64, 20, 4)

    def test_get_multitask_loss(self):
        with self.test_session() as sess:
            class_softmax = tf.placeholder(tf.float32, shape=(None, 21))
            detection_regressor = tf.placeholder(tf.float32, shape=(None, 20, 4))
            class_label_batch = tf.placeholder(tf.float32, shape=(None, 21))
            detection_label_batch = tf.placeholder(tf.float32, shape=(None, 4))

            target = rcnn_net.get_multitask_loss(
                class_softmax, detection_regressor, class_label_batch, detection_label_batch)

            sess.run(tf.global_variables_initializer())

            class_softmax_test = np.zeros((64, 21))
            detection_regressor_test = np.zeros((64, 20, 4))
            class_label_batch_test = np.zeros((64, 21))
            detection_label_batch_test = np.zeros((64, 4))

            result = sess.run(target, feed_dict={
                class_softmax: class_softmax_test,
                detection_regressor: detection_regressor_test,
                class_label_batch: class_label_batch_test,
                detection_label_batch: detection_label_batch_test
            })

            assert result.shape == (64, 1)

    def test_get_predicted_objects(self):
        with self.test_session() as sess:
            roi_input_batch = tf.placeholder(tf.float32, shape=(None, 4))
            class_softmax = tf.placeholder(tf.float32, shape=(None, 21))
            detection_regressor = tf.placeholder(tf.float32, shape=(None, 20, 4))

            target = rcnn_net.get_predicted_objects(
                21, roi_input_batch, class_softmax, detection_regressor)

            sess.run(tf.global_variables_initializer())

            roi_input_batch_test = np.zeros((64, 4))
            class_softmax_test = np.zeros((64, 21))
            class_softmax_test[:, 1] = np.ones(64)
            detection_regressor_test = np.zeros((64, 20, 4))

            result = sess.run(target, feed_dict={
                roi_input_batch: roi_input_batch_test,
                class_softmax: class_softmax_test,
                detection_regressor: detection_regressor_test,
            })

            # All rois are 'person' and they all overlap, so we only get one roi as valid
            assert result.shape == (1, 5)


if __name__ == '__main__':
    tf.test.main()
