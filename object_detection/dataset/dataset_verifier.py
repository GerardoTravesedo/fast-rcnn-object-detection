import math
import pickle
from os import listdir

import numpy as np

from tools import image_tools

CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane",
           "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable",
           "pottedplant", "sofa", "tvmonitor"]

ORIGINAL_IMAGE_SIZE = (600, 600, 3)


def verify_files(folder_path):
    """
    This functions checks if all the records in all the files in the give folder are valid. Check
    the description of check_record for accurate information about how every record is validated

    :param folder_path: folder that contains the dataset files
    """
    for dataset_file in listdir(folder_path):
        verify_file(folder_path + dataset_file)


def verify_file(file_path):
    """
    This function checks if a bunch of dataset generated records in the given file are valid. Check
    the description of check_record for accurate information about how every record is validated

    :param file_path: path to file that contains the records
    """
    print("Verifying dataset file {}".format(file_path))

    with open(file_path, mode='rb') as f:
        data = pickle.load(f, encoding='latin1')
        print("Number of records: {}".format(len(data)))

    for record in data:
        try:
            check_record(record)
        except AssertionError:
            print("File with error: {}".format(record["image_name"]))
            print("Content of record: {}".format(record))
            raise


def check_record(record):
    """
    This function checks if a generated record is valid. The validations that it performs are the
    following:
      - The size of the image is the correct one
      - It contains the name of the image
      - Ground truth information comes in a dictionary, and the class and bbox are valid
      - All rois have valid bboxes and classes. The size of the regression targets is correct
      - All background rois have valid bboxes and classes. The size of the regression
      targets is correct and is full of zeros

    :param record: Dictionary containing information for a dataset record
    """
    assert isinstance(record, dict)
    assert record["image"].shape == ORIGINAL_IMAGE_SIZE
    assert record["image_name"].endswith(".jpg")

    # Checking ground truth information
    for gt_box in record["gt_bboxes"]:
        assert isinstance(gt_box, dict)
        check_bbox(gt_box["bbox"])
        assert gt_box["class"] in CLASSES

    # Checking foreground rois
    for roi in record["rois"]:
        assert isinstance(roi, dict)
        check_bbox(roi["bbox"])
        assert np.sum(roi["class"]) == 1
        assert roi["reg_target"].shape == (4,)
        check_reg_target(record["gt_bboxes"], roi)

    # Checking background rois
    for background_roi in record["rois_background"]:
        assert isinstance(background_roi, dict)
        check_bbox(background_roi["bbox"])
        expected_background_class = np.zeros(21)
        expected_background_class[0] = 1
        np.testing.assert_equal(background_roi["class"], expected_background_class)
        np.testing.assert_equal(background_roi["reg_target"], np.zeros(4))


def check_reg_target(gt_bboxes, roi):
    roi_bbox = roi["bbox"]
    roi_reg_targets = roi["reg_target"]

    gx = round(roi_bbox[2] * roi_reg_targets[0] + roi_bbox[0])
    gy = round(roi_bbox[3] * roi_reg_targets[1] + roi_bbox[1])
    gw = round(roi_bbox[2] * math.exp(roi_reg_targets[2]))
    gh = round(roi_bbox[3] * math.exp(roi_reg_targets[3]))

    assert [gt_box for gt_box in gt_bboxes
            if np.array_equal(gt_box["bbox"], np.array([gx, gy, gw, gh]))]


def check_bbox(bbox):
    """
    This function checks if a given bbox is valid. To determine validity, it checks that the
    coordinates are not negative and the bbox falls withing the original image size

    :param bbox: bbox field indicating [x, y, width, height]
    """
    assert bbox.shape == (4,)
    negative_elements = [box_element for box_element in bbox if box_element < 0]
    assert not negative_elements
    assert bbox[0] + bbox[2] <= ORIGINAL_IMAGE_SIZE[0]
    assert bbox[1] + bbox[3] <= ORIGINAL_IMAGE_SIZE[1]


def generate_images_with_bboxes(file_path):
    """
    This function regenerates the original images drawing foreground rois and gt boxes on it, and
    then writes them all to an output folder for human inspection

    :param file_path: path to an rcnn dataset generated file
    """
    output_folder = "../verification/"

    with open(file_path) as f:
        data = pickle.load(f, encoding='latin1')

    for record in data:
        image = record["image"]
        image_name = record["image_name"].split("/")[-1]
        rois_bboxes = [roi["bbox"] for roi in record["rois"]]
        gt_bboxes = [gt["bbox"] for gt in record["gt_bboxes"]]

        print("Generating image {}".format(image_name))

        image_tools.generate_image_with_bboxes(
            image, image_name, rois_bboxes, gt_bboxes, output_folder)


if __name__ == '__main__':
    training_dataset_folder = "../../../datasets/images/pascal-voc/transformed/training/"
    test_dataset_folder = "../../../datasets/images/pascal-voc/transformed/test/"
    verify_files(test_dataset_folder)
    #generate_images_with_bboxes(test_dataset_folder + "rcnn_dataset_0")
