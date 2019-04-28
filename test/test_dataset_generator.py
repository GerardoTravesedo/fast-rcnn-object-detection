import numpy as np

import dataset.dataset_generator as dataset_generator


class TestDatasetGenerator(object):

    def test_get_bbox(self):
        input_raw_bbox = {"xmin": 10, "ymin": 10, "xmax": 16, "ymax": 18}
        result_bbox = dataset_generator.get_bbox(input_raw_bbox)
        np.testing.assert_equal(np.array([10, 10, 6, 8]), result_bbox)

    def test_get_bbox_resized(self):
        image_original_size = (20, 20)
        image_resized_size = (40, 40)
        original_bbox = np.array([2, 3, 2, 4])
        result = dataset_generator.get_bbox_resized(
            image_original_size, image_resized_size, original_bbox)
        np.testing.assert_equal(np.array([4, 6, 4, 8]), result)

    def test_get_image_data(self):
        input_image_path = "test/data/one_object_image.jpg"
        input_annotation_path = "test/data/xml_annotation_one_object.xml"
        result = dataset_generator.get_image_data_training(input_image_path, input_annotation_path)
        assert "image" in result
        assert (600, 600, 3) == result["image"].shape
        assert "image_name" in result
        assert "test/data/one_object_image.jpg" == result["image_name"]
        assert "gt_bboxes" in result
        assert 1 == len(result["gt_bboxes"])
        assert "rois" in result
        assert 15 == len(result["rois"])
        assert "rois_background" in result
        assert 200 == len(result["rois_background"])

        one_roi = result["rois"][0]
        assert 21 == len(one_roi["class"])

