import numpy as np

import dataset.roi_tools as roi_tools
import tools.image_tools as image_tools


class TestRoiTools(object):

    def test_find_rois(self):
        input_path = "test/data/one_object_image.jpg"
        input_pixels = image_tools.image_to_pixels(input_path)
        rois = roi_tools.find_rois_selective_search(input_pixels)
        # If something changes in the way the rois are calculates we will know thanks to
        # this assertion
        assert len(rois) == 793
        # If something changes in the roi output format we will know thanks to this assertion
        np.testing.assert_equal(np.array([447, 325, 4, 3]), rois[0])

    def test_calculate_iou_25_percent(self):
        bbox_1 = np.array([0, 0, 10, 10])
        bbox_2 = np.array([0, 5, 10, 10])
        iou = roi_tools.calculate_iou(bbox_1, bbox_2)
        assert float("{0:.2f}".format(iou)) == 0.33

    def test_calculate_iou_50_percent(self):
        bbox_1 = np.array([0, 0, 20, 20])
        bbox_2 = np.array([10, 0, 10, 20])
        iou = roi_tools.calculate_iou(bbox_1, bbox_2)
        assert float("{0:.2f}".format(iou)) == 0.50

    def test_calculate_iou_0_percent_vertically(self):
        bbox_1 = np.array([0, 0, 10, 10])
        bbox_2 = np.array([10, 0, 10, 10])
        iou = roi_tools.calculate_iou(bbox_1, bbox_2)
        assert float("{0:.2f}".format(iou)) == 0.0

    def test_calculate_iou_0_percent_horizontally(self):
        bbox_1 = np.array([0, 0, 10, 10])
        bbox_2 = np.array([10, 0, 10, 10])
        iou = roi_tools.calculate_iou(bbox_1, bbox_2)
        assert float("{0:.2f}".format(iou)) == 0.0

    def test_find_foreground_roi_labels_one_object(self):
        gt_object = {"class": "person", "bbox": np.array([0, 0, 20, 20])}
        roi_bbox = np.array([10, 0, 10, 20])

        labels = roi_tools.find_roi_labels(roi_bbox, [gt_object])
        labels["reg_target"][2] = float("{0:.2f}".format(labels["reg_target"][2]))

        expected_reg_targets = np.array([-1.0, 0.0, 0.69, 0])
        expected_class = np.zeros(21)
        expected_class[1] = 1

        np.testing.assert_equal(expected_class, labels["class"])
        np.testing.assert_equal(roi_bbox, labels["bbox"])
        np.testing.assert_equal([-1.0, 0.0, 0.69, 0], expected_reg_targets)

    def test_find_background_roi_labels_one_object(self):
        gt_object = {"class": "person", "bbox": np.array([0, 0, 10, 10])}
        roi_bbox = np.array([0, 5, 10, 10])

        labels = roi_tools.find_roi_labels(roi_bbox, [gt_object])

        expected_class = np.zeros(21)
        expected_class[0] = 1
        np.testing.assert_equal(expected_class, labels["class"])
        np.testing.assert_equal(roi_bbox, labels["bbox"])
        np.testing.assert_equal(np.array([0, 0, 0, 0]), labels["reg_target"])

    def test_find_foreground_roi_labels_two_objects(self):
        gt_object_1 = {"class": "person", "bbox": np.array([0, 0, 20, 20])}
        gt_object_2 = {"class": "dog", "bbox": np.array([27, 48, 20, 20])}
        roi_bbox = np.array([10, 0, 10, 20])

        labels = roi_tools.find_roi_labels(roi_bbox, np.array([gt_object_1, gt_object_2]))
        labels["reg_target"][2] = float("{0:.2f}".format(labels["reg_target"][2]))

        expected_reg_targets = np.array([-1.0, 0.0, 0.69, 0])
        expected_class = np.zeros(21)
        expected_class[1] = 1

        np.testing.assert_equal(expected_class, labels["class"])
        np.testing.assert_equal(roi_bbox, labels["bbox"])
        np.testing.assert_equal(expected_reg_targets, labels["reg_target"])

    def test_find_rois_from_gt_boxes(self):
        gt_object = {"class": "person", "bbox": np.array([0, 0, 10, 10])}
        result = roi_tools.find_foreground_rois_from_ground_truth_boxes([gt_object], (30, 20))

        assert 32 == len(result)
        np.testing.assert_equal(
            np.array([0, 1, 9, 10]), result['01910']['bbox'])
        assert roi_tools.calculate_iou(gt_object["bbox"], result['01910']['bbox']) > 0.7

        expected_class = np.zeros(21)
        expected_class[1] = 1
        for roi in result.values():
            np.testing.assert_equal(expected_class, roi["class"])
            assert roi_tools.calculate_iou(gt_object["bbox"], roi['bbox']) > 0.7
