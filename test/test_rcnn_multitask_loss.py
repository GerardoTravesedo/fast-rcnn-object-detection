import numpy as np
import tensorflow as tf

import detection.rcnn_multitask_loss as mloss


class TestRcnnMultitaskLoss(tf.test.TestCase):

    def test_get_loss_1(self):
        with self.test_session() as sess:
            class_predictions = tf.placeholder(tf.float32, shape=(None, 3))
            detection_predictions = tf.placeholder(tf.float32, shape=(None, 2, 4))

            class_labels = tf.placeholder(tf.int32, shape=(None, 3))
            detection_labels = tf.placeholder(tf.float32, shape=(None, 4))

            target = mloss.RCNNMultitaskLoss(
                class_predictions, detection_predictions, class_labels, detection_labels)\
                .multitask_loss()

            sess.run(tf.global_variables_initializer())

            # Example with two RoIs (background included with index 0)
            class_probs_test = np.array([[0.6, 0.2, 0.2], [0.2, 0.1, 0.7]])
            # Background class not included in output
            # Each row has two sets of targets, one per foreground class
            # There are two rows because we are testing with two rois
            detection_test = np.array([[[0.2, 0.3, 0.4, 0.5], [1.1, 0.9, 0.5, 1.0]],
                                       [[1.2, 1.3, 0.4, 0.5], [1.2, 0.9, 0.5, 1.0]]])

            class_labels_test = np.array([[1, 0, 0], [0, 0, 1]])
            # Needs to be all zeros for background
            box_labels_test = np.array([[0., 0., 0., 0.], [1.1, 1.3, 0.5, 2.1]])

            result = sess.run(target, feed_dict={
                class_predictions: class_probs_test,
                detection_labels: box_labels_test,
                class_labels: class_labels_test,
                detection_predictions: detection_test
            })

            self.assertAllClose([[0.510825], [1.041675]], result, rtol=1e-6)

if __name__ == '__main__':
    tf.test.main()
