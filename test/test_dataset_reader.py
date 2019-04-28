import numpy as np

import dataset.dataset_reader as dataset_reader


class TestRoiTools(object):

    def test_read_mini_batches(self):
        input_folder = "test/data/test-batch-reader-dataset/batch/"
        input_files = [input_folder + "rcnn_dataset_0", input_folder + "rcnn_dataset_1"]
        target = dataset_reader.DatasetReader(input_files, 1, 64, 16)

        # BATCH 1 -> One image

        batch1 = target.get_batch()
        assert batch1["images"].shape == (1, 600, 600, 3)
        assert batch1["rois"].shape == (64, 4)
        assert batch1["class_labels"].shape == (64, 21)
        assert batch1["reg_target_labels"].shape == (64, 4)
        # Checking that there are 59 background rois for every image
        assert np.sum(batch1["class_labels"], axis=0)[0] == 59
        # Checking ground truth information
        assert batch1["gt_objects"].shape == (4,)
        np.testing.assert_equal(
            batch1["gt_objects"][0], {"class": "aeroplane", "bbox": np.array([124, 166, 325, 224])})
        np.testing.assert_equal(
            batch1["gt_objects"][1], {"class": "aeroplane", "bbox": np.array([159, 187,  76,  74])})
        np.testing.assert_equal(
            batch1["gt_objects"][2], {"class": "person", "bbox": np.array([234, 384, 21, 104])})
        np.testing.assert_equal(
            batch1["gt_objects"][3], {"class": "person", "bbox": np.array([31, 403, 21, 104])})

        # BATCH 2 -> One image

        batch2 = target.get_batch()
        assert batch2["images"].shape == (1, 600, 600, 3)
        assert batch2["rois"].shape == (64, 4)
        assert batch2["class_labels"].shape == (64, 21)
        assert batch2["reg_target_labels"].shape == (64, 4)
        # Checking that there are 59 background rois for every image
        assert np.sum(batch2["class_labels"], axis=0)[0] == 59
        # Checking ground truth information
        assert batch2["gt_objects"].shape == (3,)
        np.testing.assert_equal(
            batch2["gt_objects"][0], {"class": "aeroplane", "bbox": np.array([10, 175, 588, 255])})
        np.testing.assert_equal(
            batch2["gt_objects"][1], {"class": "aeroplane", "bbox": np.array([505, 327,  73,  42])})
        np.testing.assert_equal(
            batch2["gt_objects"][2], {"class": "aeroplane", "bbox": np.array([390, 308, 103,  57])})

        # BATCH 3 -> One image, change in file

        batch3 = target.get_batch()
        assert batch3["images"].shape == (1, 600, 600, 3)
        assert batch3["rois"].shape == (64, 4)
        assert batch3["class_labels"].shape == (64, 21)
        # Checking that there are 59 background rois for the image
        assert np.sum(batch3["class_labels"], axis=0)[0] == 59
        assert batch3["reg_target_labels"].shape == (64, 4)
        # Checking ground truth information
        assert batch3["gt_objects"].shape == (1,)
        np.testing.assert_equal(
            {"class": "person", "bbox": np.array([214, 121, 216, 300])}, batch3["gt_objects"][0])

        # BATCH 4 -> Emtpy, there are no more images

        batch4 = target.get_batch()
        assert batch4 == {}

