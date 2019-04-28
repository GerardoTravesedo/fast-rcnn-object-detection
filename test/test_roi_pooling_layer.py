import numpy as np
import tensorflow as tf

import detection.roi_pooling_layer as pooling

# Shape (4, 5, 2) -> subimage
FEATURE_MAP_TEST = np.array([[[122, 2], [2, 3], [3, 400], [4, 5], [22, 23]],
                             [[6, 7], [137, 8], [8, 9], [9, 10], [25, 26]],
                             [[12, 130], [13, 14], [14, 15], [15, 16], [28, 209]],
                             [[17, 18], [18, 19], [19, 20], [320, 21], [300, 31]]])


class TestRoiPoolingLayer(tf.test.TestCase):

    def test_get_pooling_layer_two_rois(self):
        rois_test = [[0, 0, 5, 4], [1, 1, 1, 1]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        self.assertAllEqual([[[137, 8], [25, 400]], [[18, 130], [320, 209]]], result[0])
        self.assertAllEqual([[[137, 8], [8, 9]], [[13, 14], [14, 15]]], result[1])

    def test_get_pooling_layer_three_rois_ratio_2(self):
        rois_test = [[0, 0, 5, 4], [5, 4, 5, 4], [9, 7, 1, 1]]
        result = self._get_pooling_layer(2, 2, 2, rois_test)
        self.assertAllEqual([[[122, 2], [3, 400]], [[6, 7], [137, 9]]], result[0])
        self.assertAllEqual([[[14, 15], [28, 209]], [[19, 20], [320, 31]]], result[1])
        self.assertAllEqual([[[15, 16], [28, 209]], [[320, 21], [300, 31]]], result[2])

    def test_get_pooling_layer_roi_right_side(self):
        rois_test = [[4, 2, 1, 1]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        # Shape (1, 2, 2, 2) -> Just one roi, W=2, H=2, Channels=2
        self.assertAllEqual([[[[15, 16], [28, 209]], [[320, 21], [300, 31]]]], result)

    def test_get_pooling_layer_roi_bottom_side(self):
        rois_test = [[2, 3, 1, 1]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        # Shape (1, 2, 2, 2) -> Just one roi, W=2, H=2, Channels=2
        self.assertAllEqual([[[[14, 15], [15, 16]], [[19, 20], [320, 21]]]], result)

    def test_get_pooling_layer_roi_left_side(self):
        rois_test = [[0, 1, 3, 3]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        # Shape (1, 2, 2, 2) -> Just one roi, W=2, H=2, Channels=2
        self.assertAllEqual([[[[6, 7], [137, 9]], [[17, 130], [19, 20]]]], result)

    def test_get_pooling_layer_roi_top_side(self):
        rois_test = [[2, 0, 3, 3]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        # Shape (1, 2, 2, 2) -> Just one roi, W=2, H=2, Channels=2
        self.assertAllEqual([[[[3, 400], [22, 23]], [[14, 15], [28, 209]]]], result)

    def test_get_pooling_layer_roi_bottom_right_side(self):
        rois_test = [[0, 3, 1, 1]]
        result = self._get_pooling_layer(2, 2, 1, rois_test)
        # Shape (1, 2, 2, 2) -> Just one roi, W=2, H=2, Channels=2
        self.assertAllEqual([[[[12, 130], [13, 14]], [[17, 18], [18, 19]]]], result)

    def _get_pooling_layer(self, pooling_w, pooling_h, pooling_ratio, rois_test):
        with self.test_session() as sess:
            featuremap = tf.placeholder(tf.float32, shape=(4, 5, 2))
            rois = tf.placeholder(tf.float32, shape=(None, 4))

            pooling_layer = pooling.RoiPoolingLayer(
                featuremap, rois, pooling_h, pooling_w, pooling_ratio).get_roi_pooling_layer()

            sess.run(tf.global_variables_initializer())

            result = sess.run(pooling_layer, feed_dict={
                featuremap: FEATURE_MAP_TEST,
                rois: rois_test
            })

            return result


if __name__ == '__main__':
    tf.test.main()
