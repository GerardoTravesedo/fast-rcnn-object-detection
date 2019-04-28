import numpy as np
import tensorflow as tf

import detection.detect_and_classify_instances as dc


class TestDetectAndClassifyInstances(tf.test.TestCase):

    def test_find_bboxes_from_offsets_four_rois_one_class(self):
        with self.test_session() as sess:
            # Each roi contains 4 fields (x, y, w, h)
            rois = tf.placeholder(tf.float32, shape=(None, 4))
            # Each roi contains 4 fields (tx, ty, tw, th)
            rois_reg_targets = tf.placeholder(tf.float32, shape=(None, 4))

            # The four rois we use to test
            r_test = np.array([[0, 0, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]])
            # The regression targets for the two boxes for one class
            r_reg_test = np.array([[0.2, 0.2, 0, 0], [0, 0, 0, 0],
                                   [0, 0, 0.1823, 0.4700], [-0.2, -0.2, -0.51, 0.18]])

            # The result for each bbox has format (x1, y1, x2, y2)
            expected_result = np.array([[1., 1., 5., 5.], [5., 5., 9., 9.],
                                        [5., 5., 10., 12.], [4., 4., 6., 9.]])

            result = sess.run(dc.find_bboxes_from_offsets(rois, rois_reg_targets), feed_dict={
                rois: r_test,
                rois_reg_targets: r_reg_test
            })

            self.assertAllClose(expected_result, result, rtol=1e-6)

    def test_detect_objects_iou_overlap(self):
        previous_objects_test = np.array([[0., 0., 5., 5., 1.]])

        rois_test = np.array([[5., 5., 5., 5.], [5., 4., 5., 5.], [12., 10., 6., 6.]])

        class_rois_info_test = np.array([[0., 0., 0., 0., 0.4, 2.],
                                         [0., 0., 0., 0., 0.6, 2.],
                                         [0.3, 0.2, 0., 0.470, 0.6, 2.]])

        # The previous object + two new objects for class 2
        expected_result = np.array([[0., 0., 5., 5., 1.],
                                    [5., 4., 9., 8., 2.],
                                    [14., 11., 19., 20., 2.]])

        self._test_object_detection(
            previous_objects_test, rois_test, class_rois_info_test, expected_result)

    def test_detect_objects_no_objects_for_class(self):
        previous_objects_test = np.array([[0., 0., 5., 5., 1.]])

        rois_test = np.array([[5., 5., 5., 5.], [12., 10., 6., 6.]])

        class_rois_info_test = np.array([[0., 0., 0., 0., 0.2, 2.],
                                         [0.3, 0.2, 0., 0.470, 0.3, 2.]])

        # No new detected objects since their scores are low
        expected_result = np.array([[0., 0., 5., 5., 1.]])

        self._test_object_detection(
            previous_objects_test, rois_test, class_rois_info_test, expected_result)

    def test_detect_and_classify_two_rois(self):
        with self.test_session() as sess:
            # Let's say that we have 3 classes for this test
            rois_class_scores = tf.placeholder(tf.float32, shape=(None, 3))
            # 4 fields for every roi (x, y, w, h)
            rois = tf.placeholder(tf.float32, shape=(None, 4))
            # Each roi has reg targets for every class (3 in this test)
            rois_reg_targets = tf.placeholder(tf.float32, shape=(None, 2, 4))

            rois_test = np.array([[5., 5., 5., 5.], [12., 10., 6., 6.]])
            rois_class_scores_test = np.array([[0.1, 0.3, 0.6], [0.6, 0.3, 0.1]])
            # Two sets of reg targets (one per class not including background) for each roi
            rois_reg_targets_test = np.array([[[0., 0., 0., 0.], [0.2, 0.1, 0.1, 0.]],
                                              [[0., 0., 0., 0.], [0., 0., 0., 0.]]])

            # Just the first roi has a score over 0.4 for a class (class 2 in this test)
            # Second roi has score > 0.4 for background, so it is not included in the detedcted
            # objects
            expected_result = np.array([[6., 6., 11., 10., 2.]])

            result = sess.run(dc.detect_and_classify(
                rois, rois_class_scores, rois_reg_targets, 2, 0.4, 3), feed_dict={
                rois: rois_test,
                rois_class_scores: rois_class_scores_test,
                rois_reg_targets: rois_reg_targets_test
            })

            self.assertAllClose(expected_result, result, rtol=1e-6)

    def _test_object_detection(
        self, previous_objects_test, rois_test, class_rois_info_test, expected_result):
        with self.test_session() as sess:
            # Tensor with previous detected objects for other classes. There are 5 fields:
            # (x1, y1, x2, y2) and the class at the end
            previous_detected_objects = tf.placeholder(tf.float32, shape=(None, 5))
            # 4 fields for every roi (x, y, w, h)
            rois = tf.placeholder(tf.float32, shape=(None, 4))
            # 6 fields for the roi info = (x, y, w, h) + class score + class
            class_rois_info = tf.placeholder(tf.float32, shape=(None, 6))

            result = sess.run(dc.detect_objects_for_class(
                previous_detected_objects, rois, class_rois_info, 2, 0.4), feed_dict={
                previous_detected_objects: previous_objects_test,
                rois: rois_test,
                class_rois_info: class_rois_info_test
            })

            self.assertAllClose(expected_result, result, rtol=1e-6)


if __name__ == '__main__':
    tf.test.main()
