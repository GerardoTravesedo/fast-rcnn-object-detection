import math

import numpy as np

import dataset_generator, xml_parser, roi_tools
from tools import image_tools

INPUT_FOLDER = "dataset-training-test/training/"

IMAGE_FOLDER = INPUT_FOLDER + "image/"
ANNOTATION_FOLDER = INPUT_FOLDER + "annotation/"

INPUT_ANNOTATION = ANNOTATION_FOLDER + "2007_001225.xml"
INPUT_IMAGE = IMAGE_FOLDER + "2007_001225.jpg"


def show_image_with_highest_iou_roi(image, gt_bboxes):
    rois = roi_tools.find_rois_selective_search(image)

    bboxes_to_display = []
    for gt_bbox in gt_bboxes:
        bboxes_to_display.append(gt_bbox)

        max_iou = 0
        max_roi = None

        for roi in rois:
            iou = roi_tools.calculate_iou(gt_bbox, roi)
            if iou > max_iou:
                max_iou = iou
                max_roi = roi

        print("Max IoU: " + str(max_iou))
        bboxes_to_display.append(max_roi)

    print(bboxes_to_display)
    image_tools.show_image_with_bboxes(resized_image_in_pixels, bboxes_to_display)


def show_image_all_rois(image):
    rois = roi_tools.find_rois_selective_search(image)
    image_tools.show_image_with_bboxes(image, rois)


def show_image_all_custom_rois(gt_boxes, image):
    rois = roi_tools.find_rois_from_ground_truth_boxes(gt_boxes, image.shape)
    print(rois)
    image_tools.show_image_with_bboxes(image, rois)


def print_iou_info(image, gt_bboxes):
    rois_fore, rois_back = roi_tools.find_rois_complete(image, gt_bboxes, 4, 1000)

    count_background = 0
    count_foreground = 0

    for gt_bbox in gt_bboxes:
        for roi in rois_fore:
            iou = roi_tools.calculate_iou(gt_bbox["bbox"], roi)
            if iou >= 0.5:
                print(iou)
                count_foreground = count_foreground + 1
            else:
                count_background = count_background + 1

    print("Foreground rois: " + str(count_foreground))
    print("Background rois: " + str(count_background))


def verify_regression_targets():
    data = dataset_generator.get_image_data_training(INPUT_IMAGE, INPUT_ANNOTATION)

    print("Expected:")
    print(data["gt_bboxes"])
    foreground_rois = data["rois"]

    roi_bbox = foreground_rois[0]["bbox"]
    roi_target = foreground_rois[0]["reg_target"]

    result_x = roi_bbox["w"] * roi_target["tx"] + roi_bbox["x"]
    result_y = roi_bbox["h"] * roi_target["ty"] + roi_bbox["y"]
    result_w = roi_bbox["w"] * math.exp(roi_target["tw"])
    result_h = roi_bbox["h"] * math.exp(roi_target["th"])

    print("Result:")
    print(result_x)
    print(result_y)
    print(result_w)
    print(result_h)


# image_in_pixels = image_tools.image_to_pixels(INPUT_IMAGE)
# resized_image_in_pixels = image_tools.resize_image(image_in_pixels, 600, 600)
#
# image_annotations = xml_parser.parse_xml(INPUT_ANNOTATION)
#
# gt_bboxes = []
#
# for annotation in image_annotations:
#     bbox = dataset_generator.get_bbox(annotation["bbox"])
#
#     resized_bbox = dataset_generator.get_bbox_resized(
#         image_in_pixels.shape, resized_image_in_pixels.shape, bbox)
#
#     gt_bboxes.append({"class": annotation["class"], "bbox": resized_bbox})

#verify_regression_targets()
#print_iou_info(resized_image_in_pixels, gt_bboxes)
#show_image_with_highest_iou_roi(resized_image_in_pixels, gt_bboxes)
#show_image_all_rois(resized_image_in_pixels)
#print dataset_generator.get_image_data(INPUT_IMAGE, INPUT_ANNOTATION)
#dataset_generator.generate_training_test_sets()

#dataset_generator.generate_roi_report(IMAGE_FOLDER, ANNOTATION_FOLDER, "reports/")

#show_image_all_custom_rois(gt_bboxes, resized_image_in_pixels)
#print_iou_info(resized_image_in_pixels, gt_bboxes)

#dataset_generator.generate_reduced_training_test_sets(["cat", "bird", "aeroplane"])


xml_data = xml_parser.parse_xml("../dataset-training-test/test-reduced/annotation/2007_002619.xml")
print(dataset_generator.get_bbox(xml_data[0]["bbox"]))
bbox = dataset_generator.get_bbox_resized((500, 380), (600, 600), dataset_generator.get_bbox(xml_data[0]["bbox"]))

print(bbox)

# rois = roi_tools.find_foreground_rois_from_ground_truth_boxes([
#     {'class': xml_data[0]["class"], 'bbox': np.array(bbox)}], (600, 600))

background_rois = roi_tools.find_random_background_rois([
    {'class': xml_data[0]["class"], 'bbox': np.array(bbox)}], (600, 600), 200, 10, 10, 200, 200)

background_bboxes = [roi["bbox"] for roi in background_rois.values()]

image_tools.generate_image_with_bboxes(
    image_tools.resize_image(
        image_tools.image_to_pixels("../dataset-training-test/test-reduced/image/2007_002619.jpg"),
        600, 600),
    "test.jpg", background_bboxes, [bbox], "output/detection/")

print(background_rois)
