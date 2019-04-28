import selectivesearch
import math
import numpy as np
import itertools
import random

NUMBER_CLASSES = 21


def find_rois_complete(image_pixels, gt_boxes, min_rois_foreground, max_rois_background):
    """
    Generates a minimum number of foreground rois and a maximum number of background ones

    1) First it generates a few foreground rois from the ground thruth boxes
    2) Then it randomly generates background rois checking that the IoU with every gt box is < 0.5
    3) If it hasn't reached the minimum number of foreground rois yet, it runs selective search

    :param
        image_pixels: pixels from the image
        gt_boxes: Ground truth boxes from the image
        min_rois_foreground: minimum number of foreground rois to find
        max_rois_background: maximum number of background rois to find
    """
    # Initial scale for selective search. It will be reduces to find more foreground rois
    init_scale = 500

    # Adding some foreground rois generated from the ground truth boxes
    rois_foreground = \
        find_foreground_rois_from_ground_truth_boxes(gt_boxes, image_pixels.shape)
    # Adding random background rois
    rois_background = \
        find_random_background_rois(gt_boxes, image_pixels.shape, 200, 10, 10, 200, 200)

    # Find rois using selective search until we have the required number of foreground rois
    while len(rois_foreground) < min_rois_foreground and init_scale > 100:
        print("Only {} foreground rois were generated, but {} are required. "
              "Running selective search".format(len(rois_foreground), min_rois_foreground))
        # Finding rois for current scale
        rois = find_rois_selective_search(image_pixels, scale=init_scale)
        # For each roi, we find if it is foreground or background
        for roi in rois:
            roi_info = find_roi_labels(roi, gt_boxes)
            key = str(roi[0]) + str(roi[1]) + str(roi[2]) + str(roi[3])
            if len(rois_background) < max_rois_background and roi_info["class"][0] == 1:
                rois_background[key] = roi_info
            elif roi_info["class"][0] == 0:
                rois_foreground[key] = roi_info
        # Reducing scale for next iteration of selective search
        init_scale = init_scale - 100

    # If selective search couldn't find any positive rois even trying multiple parameters, we
    # generate our own positive rois by moving the ground truth box slightly
    if len(rois_foreground) == 0:
        print("It couldn't find any foreground rois")
    else:
        return np.array(rois_foreground.values()), np.array(rois_background.values())


def find_rois_selective_search(image_pixels, scale=200, sigma=0.9, min_size=10):
    """
    Uses the selective search library to find rois

    :param
        image_path: path to an image
        image_pixels: pixels from the image
    """
    # Higher scale means higher preference for larger components (k / |C|, where |C| is the
    # number of pixels in the component and k is the scale; for a large k, it would be difficult
    # for a small component to be have a separation boundary with the neighboring components since
    # the division is large). Smaller components are allowed when there is a sufficiently large
    # difference between neighboring components (the higher k / |C|, the higher the difference
    # between neighboring components has to be)
    img_lbl, regions = \
        selectivesearch.selective_search(image_pixels, scale=scale, sigma=sigma, min_size=min_size)

    unique_rois = {}

    # Deduplicating rois
    for region in regions:
        # rect format: [x, y, w, h]
        rect = region["rect"]
        key = str(rect[0]) + str(rect[1]) + str(rect[2]) + str(rect[3])
        if key not in unique_rois:
            # From [x, y, w, h] to {x, y, w, h}
            unique_rois[key] = rect

    return np.array(unique_rois.values())


def find_foreground_rois_from_ground_truth_boxes(gt_boxes, image_shape):
    """
    Finds foreground rois from the ground truth boxes.

    1) It finds possible new values for each field x_min, y_min, width, height by adding
    and subtracting small fractions of the original box's width and height

    2) It finds all the combinations of new fields

    3) It keeps only those with IoU > 0.7 with the original gt box

    Example of result:

    {'1234' : {'class': [0, 1, 0], 'bbox': [1, 2, 3, 4], 'reg_targets': [-0.1, 0.003, 1.1, 0]},
     '6789': {'class': [0, 0, 1], 'bbox': [6, 7, 8, 9], 'reg_targets': [0.1, -0.917, 0.97, 0.01]}}

    :param: gt_boxes: Ground truth boxes from the image
    :param: image_shape: shape of the image that contains the boxes

    :return: map containing the foreground rois.
    The format is: {'key': {'class': CLASS, 'bbox': BBOX}, 'reg_targets': TARGETS}
    where the key is constructed from the bbox fields of the roi and the value is another
    dictionary with the roi information
    """
    image_height_pixels = image_shape[0]
    image_width_pixels = image_shape[1]
    foreground_rois = {}

    def find_possible_coordinate_values(coordinate_value, axis_length, max_possible_value):
        possible_values = set()

        max_axis_displacement = axis_length / 6
        min_axis_displacement = max_axis_displacement / 2

        if not coordinate_value + max_axis_displacement > max_possible_value:
            possible_values.add(coordinate_value + max_axis_displacement)

        if not coordinate_value + min_axis_displacement > max_possible_value:
            possible_values.add(coordinate_value + min_axis_displacement)

        if coordinate_value - max_axis_displacement > 0:
            possible_values.add(coordinate_value - max_axis_displacement)

        if coordinate_value - min_axis_displacement > 0:
            possible_values.add(coordinate_value - min_axis_displacement)

        return possible_values

    for gt_box in gt_boxes:
        gt_class = gt_box["class"]
        gt_box = gt_box["bbox"]

        possible_min_x = \
            find_possible_coordinate_values(gt_box[0], gt_box[2], image_width_pixels - 1)

        possible_max_x = \
            find_possible_coordinate_values(
                gt_box[0] + gt_box[2] - 1, gt_box[2], image_width_pixels - 1)

        possible_min_y = \
            find_possible_coordinate_values(gt_box[1], gt_box[3], image_height_pixels - 1)

        possible_max_y = \
            find_possible_coordinate_values(
                gt_box[1] + gt_box[3] - 1, gt_box[3], image_height_pixels - 1)

        all_combinations = list(
            itertools.product(*[possible_min_x, possible_max_x, possible_min_y, possible_max_y]))

        for combination in all_combinations:
            bbox = [combination[0], combination[2], combination[1] - combination[0] + 1,
                    combination[3] - combination[2] + 1]

            iou = calculate_iou(gt_box, bbox)

            if iou > 0.7:
                # We create a hash key from the coordinates to prevent having duplicates
                key = str(bbox[0]) + str(bbox[1]) + str(bbox[2]) + str(bbox[3])
                foreground_rois[key] = \
                    {"bbox": np.array(bbox),
                     "class": class_string_to_index(gt_class),
                     "reg_target": np.array(find_regression_targets(gt_box, bbox))}

    return foreground_rois


def find_random_background_rois(
    gt_boxes, image_shape, number_background_rois, min_width, min_height, max_width, max_height):
    """
    This function generates a map with the specified number of background rois randomly generated.

    The key in this map is used to prevent duplicate rois, and is generated by putting together
    all the bbox fields for a given roi.

    The value is the actual roi information, containing the class and the bbox.

    Example of result:

    {'1234' : {'class': [1, 0, 0], 'bbox': [1, 2, 3, 4], 'reg_targets': [0, 0, 0, 0]},
     '6789': {'class': [1, 0, 0], 'bbox': [6, 7, 8, 9], 'reg_targets': [0, 0, 0, 0]}}

    This function ONLY generated background rois, so the class vector will always contain a 1 at the
    first position and a vector of zeros for the reg_targets.

    It also makes sure none of the generated background rois have IoU >= 0.5 with any gt boxes

    :param gt_boxes: list of ground truth boxes with format [x, y, w, h]
    :param image_shape: shape of the image we are calculating background rois for
    :param number_background_rois: number of rois to generate
    :param min_width: minimum width for the rois we generate
    :param min_height: minimum height for the rois we generate
    :param max_width: maximum width for the rois we generate
    :param max_height: maximum height for the rois we generate

    :return: map containing the background rois.
    The format is: {'key': {'class': CLASS, 'bbox': BBOX}, 'reg_targets': TARGETS}
    where the key is constructed from the bbox fields of the roi and the value is another
    dictionary with the roi information
    """
    background_rois = {}

    while len(background_rois) < number_background_rois:
        random_height = random.randint(min_height, max_height)
        random_width = random.randint(min_width, max_width)

        # We have to control the max possible value so it is not larger than the image size
        random_x = random.randint(0, image_shape[1] - random_width - 1)
        random_y = random.randint(0, image_shape[0] - random_height - 1)

        random_roi = [random_x, random_y, random_width, random_height]

        ious = [calculate_iou(gt_box["bbox"], random_roi) for gt_box in gt_boxes]
        max_iou = max(ious)

        if max_iou < 0.5:
            # We create a hash key from the coordinates to prevent having duplicates
            key = str(random_roi[0]) + str(random_roi[1]) + str(random_roi[2]) + str(random_roi[3])
            background_rois[key] = {"bbox": np.array(random_roi),
                                    "class": class_string_to_index("background"),
                                    "reg_target": np.zeros(4)}

    return background_rois


def calculate_iou(gt_bbox, roi_bbox):
    """
    Calculates intersection over union between the ground truth bbox and a particular roi bbox

    :param
        gt_bbox: ground truth bbox
        roi_bbox: region of interest bbox
    """
    # Calculating corners of intersection box
    # Top left corner
    intersect_top_left_x = max(gt_bbox[0], roi_bbox[0])
    intersect_top_left_y = max(gt_bbox[1], roi_bbox[1])
    # Bottom right corner
    intersect_bottom_right_x = \
        min(gt_bbox[0] + gt_bbox[2] - 1, roi_bbox[0] + roi_bbox[2] - 1)
    intersect_bottom_right_y = \
        min(gt_bbox[1] + gt_bbox[3] - 1, roi_bbox[1] + roi_bbox[3] - 1)

    # We add +1 because the two boxes could be overlapping on one line of pixels (one edge), and
    # that shouldn't count as 0
    area_intersection = max(0, intersect_bottom_right_x - intersect_top_left_x + 1) * \
                        max(0, intersect_bottom_right_y - intersect_top_left_y + 1)

    area_gt_bbox = gt_bbox[2] * gt_bbox[3]
    area_roi_bbox = roi_bbox[2] * roi_bbox[3]

    union_area = area_gt_bbox + area_roi_bbox - area_intersection

    return area_intersection / float(union_area)


def find_roi_labels(roi_bbox, gt_objects):
    """
    Generates labels for a given roi. The labels are composed of a class and a bbox regression
    target.

    The class is found by calculating the IoU with all the ground truth boxes and keeping the
    class of the one with highest value

    The regression targets are found using the following formulas:

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = log(Gw / Pw)
    th = log(Gh / Ph)

    :param
        roi_bbox: region of interest bbox
        gt_objects: all the objects in the image (contains class and bbox)
    """
    max_iou = 0.5
    roi_class = None
    roi_bbox_target = np.zeros(1)

    # Finding the gt object with the highest IoU with the roi
    for gt_object in gt_objects:
        iou = calculate_iou(gt_object["bbox"], roi_bbox)

        if iou >= max_iou:
            max_iou = iou
            roi_class = gt_object["class"]
            roi_bbox_target = gt_object["bbox"]

    # If roi_bbox_target only has zeros, any returns false
    if roi_class and roi_bbox_target.any():
        # [tx, ty, tw, th]
        regression_targets = find_regression_targets(roi_bbox_target, roi_bbox)
        return {"bbox": np.array(roi_bbox),
                "class": class_string_to_index(roi_class),
                "reg_target": np.array(regression_targets)}
    else:
        # If roi doesn't have IoU > 0.5 with any gt object, then it is background and it doesn't
        # have regression targets
        return {"bbox": np.array(roi_bbox),
                "class": class_string_to_index("background"),
                "reg_target": np.zeros(4)}


def find_regression_targets(gt_box, roi_bbox):
    """
    The regression targets are found using the following formulas:

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = log(Gw / Pw)
    th = log(Gh / Ph)

    :param gt_box: ground truth box used to find the reg targets
    :param roi_bbox: roi box we need to find reg targets for

    :return: regression targets [tx, ty, tw, th]
    """
    # Calculating regression targets according to formulas on paper
    tx = (gt_box[0] - roi_bbox[0]) / float(roi_bbox[2])
    ty = (gt_box[1] - roi_bbox[1]) / float(roi_bbox[3])
    tw = math.log(gt_box[2] / float(roi_bbox[2]))
    th = math.log(gt_box[3] / float(roi_bbox[3]))
    # [tx, ty, tw, th]
    return [tx, ty, tw, th]


def class_string_to_index(class_string):
    """
    Converts a class in string format into an array with all values to 0 but a 1 for the index
    of the right class

    :param
        class_string: string representing the class name
    """
    switcher = {
        "background": 0, "person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5, "horse": 6,
        "sheep": 7, "aeroplane": 8, "bicycle": 9, "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
        "train": 14, "bottle": 15, "chair": 16, "diningtable": 17, "pottedplant": 18,
        "sofa": 19, "tvmonitor": 20
    }

    class_index = switcher.get(class_string, -1)

    if class_index == -1:
        raise Exception("Invalid class " + class_string)

    classes = np.zeros(NUMBER_CLASSES)
    classes[class_index] = 1

    return classes
