import shutil
from os import listdir

import numpy as np

import dataset.xml_parser as xml_parser
import dataset.roi_tools as roi_tools
from tools import image_tools

INPUT_FOLDER = "../../../datasets/images/pascal-voc/original/VOCdevkit/VOC2012/"
IMAGES_PATH = INPUT_FOLDER + "JPEGImages/"
ANNOTATIONS_PATH = INPUT_FOLDER + "Annotations/"

# Generating training and tests sets
DEST_TRAINING_FOLDER_IMAGE = "../../../datasets/images/pascal-voc/separated/training/image/"
DEST_TRAINING_FOLDER_ANNOTATION = "../../../datasets/images/pascal-voc/separated/training/annotation/"
DEST_TEST_FOLDER_IMAGE = "../../../datasets/images/pascal-voc/separated/test/image/"
DEST_TEST_FOLDER_ANNOTATION = "../../../datasets/images/pascal-voc/separated/test/annotation/"

# Generating reduced training and tests sets
DEST_TRAINING_FOLDER_IMAGE_REDUCED = "../../../datasets/images/pascal-voc/separated/training-reduced/image/"
DEST_TRAINING_FOLDER_ANNOTATION_REDUCED = "../../../datasets/images/pascal-voc/separated/training-reduced/annotation/"
DEST_TEST_FOLDER_IMAGE_REDUCED = "../../../datasets/images/pascal-voc/separated/test-reduced/image/"
DEST_TEST_FOLDER_ANNOTATION_REDUCED = "../../../datasets/images/pascal-voc/separated/test-reduced/annotation/"


def generate_reduced_training_test_sets(filter_classes=None):
    """
    Separates original PASCAL VOC dataset into training (80%) and test(20%) for a small group
    of classes

    :param filter_classes: list with valid classes
    """
    def move_files(annotation_file, annotation_dest_path, image_dest_path):
        # Moving annotation file that only contains valid objects to the output folder
        shutil.copy(ANNOTATIONS_PATH + annotation_file, annotation_dest_path + annotation_file)
        # Moving the corresponding image file to the output folder
        image_file = annotation_file.split(".")[0] + ".jpg"
        shutil.copy(IMAGES_PATH + image_file, image_dest_path + image_file)

    current_image = 0

    for filename in listdir(ANNOTATIONS_PATH):
        annotation_path = ANNOTATIONS_PATH + filename
        # If image only contains objects with the specified classes, then we move it to the
        # destination folder
        if not filter_classes or xml_parser.contains_valid_classes(annotation_path, filter_classes):
            # One every five files goes to test destination folder
            # The rest go to the training destination folder
            if current_image % 5 == 0:
                move_files(
                    filename,
                    DEST_TEST_FOLDER_ANNOTATION_REDUCED,
                    DEST_TEST_FOLDER_IMAGE_REDUCED
                )
            else:
                move_files(
                    filename,
                    DEST_TRAINING_FOLDER_ANNOTATION_REDUCED,
                    DEST_TRAINING_FOLDER_IMAGE_REDUCED
                )

            current_image += 1


def generate_training_test_sets():
    """
    Separates original PASCAL VOC dataset into training (80%) and test(20%)
    """
    def move_files(init_path, dest_training_path, dest_test_path):
        """
        Moves files from original location to destination location

        :param
            init_path: path to the folder where the original file is located
            dest_training_path: path to the folder where the training files will be stored
            dest_test_path: path to the folder where the test files will be stored
        """
        current_image = 0

        for filename in sorted(listdir(init_path)):
            current_image = current_image + 1
            if current_image % 5 == 0:
                shutil.copy(init_path + filename, dest_test_path + filename)
            else:
                shutil.copy(init_path + filename, dest_training_path + filename)

    move_files(IMAGES_PATH, DEST_TRAINING_FOLDER_IMAGE, DEST_TEST_FOLDER_IMAGE)
    move_files(ANNOTATIONS_PATH, DEST_TRAINING_FOLDER_ANNOTATION, DEST_TEST_FOLDER_ANNOTATION)


def get_image_data_training(image_path, annotation_path):
    """
    Generates the data necessary for rcnn training. This info includes:
        - Resized image pixels
        - Ground truth data including bounding boxes and classes of the different objects
        - RoIs for both foreground and background classes

    :param image_path: path to the image we are generating data for
    :param annotation_path: path to the annotations of the image were are generating data for

    :return: information about the image including pixels, ground truth data and rois
    """
    image_info = {}

    # Adding resized image to the dictionary
    image_in_pixels = image_tools.image_to_pixels(image_path)
    resized_image_in_pixels = image_tools.resize_image(image_in_pixels, 600, 600)
    image_info["image"] = resized_image_in_pixels
    image_info["image_name"] = image_path

    # Adding all the resized ground-truth bboxes to the dictionary
    gt_boxes = []
    image_annotations = xml_parser.parse_xml(annotation_path)

    for annotation in image_annotations:
        resized_gt_bbox = get_bbox_resized(
            image_in_pixels.shape, resized_image_in_pixels.shape, get_bbox(annotation["bbox"]))
        gt_boxes.append({"class": annotation["class"], "bbox": resized_gt_bbox})

    image_info["gt_bboxes"] = np.array(gt_boxes)

    # Adding rois to the dictionary
    image_info["rois"], image_info["rois_background"] = \
        roi_tools.find_rois_complete(resized_image_in_pixels, gt_boxes, 4, 500)

    if len(image_info["rois"]) == 0:
        print("There are no ROIs for image: " + image_path + ". It must be a background image")

    return image_info


def get_bbox(raw_bbox):
    """
    Given a raw bbox from the PASCAL VOC dataset, it transforms it into the format
    expected by RCNN: [x, y, w, h] where (x, y) represents the top left corner of the
    image and (w, h) represent the width and height

    :param
        image: dictionary with raw bbox
    """
    x_min = raw_bbox["xmin"]
    x_max = raw_bbox["xmax"]
    y_min = raw_bbox["ymin"]
    y_max = raw_bbox["ymax"]

    # [x, y, w, h]
    return np.array([x_min, y_min, x_max - x_min, y_max - y_min])


def get_bbox_resized(original_image_shape, resized_image_shape, original_bbox):
    """
    Translates a bbox in the original image (with the original size) into a bbox in the resized
    image. If the resized image is bigger than the original one, then the bbox wil also be bigger

    :param
        original_image_shape: shape of the original image as extracted from PASCAL VOC dataset
        resized_image_shape: shape of the resized image prepared for RCNN
        original_bbox: bbox of a given object in the original image
    """
    # Calculating how much a pixel from the original image is growing/shrinking horizontally
    # in the resized image (for example, if original image is 16x16 and resized one is 32x32
    # then every pixel in the original image is now two in the resized one in the x axis)
    y_factor = resized_image_shape[0] / float(original_image_shape[0])
    # Calculating how much a pixel from the original image is growing/shrinking vertically
    # in the resized image (for example, if original image is 16x16 and resized one is 32x32
    # then every pixel in the original image is now two in the resized one in the y axis)
    x_factor = resized_image_shape[1] / float(original_image_shape[1])

    # [x, y, w, h]
    return np.array([
        int(original_bbox[0] * x_factor), int(original_bbox[1] * y_factor),
        int(original_bbox[2] * x_factor), int(original_bbox[3] * y_factor)
    ])
