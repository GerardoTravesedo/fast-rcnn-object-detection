import tensorflow as tf


def detect_and_classify(
        rois, rois_class_scores, rois_reg_targets, max_output_size,
        score_threshold, number_classes):
    """

    :param rois: rois found for the current images. Shape = (# rois, # bbox fields = 4)
    :param rois_class_scores: Class scores for the different rois.
        Shape = (# rois, # classes)
    :param rois_reg_targets: Regression targets for each roi for each class.
        Shape = (# rois, # classes - background, # reg fields = 4)
    :param max_output_size: maximum number of boxes to be selected by non max suppression
    :param score_threshold: threshold for deciding when to remove boxes based on score
    :param number_classes: number of classes used for classification (including background)

    :return: All instances of objects detected along with their classes
        Shape = (# objects detected, # detection fields = 5) where detection fields contain:
            - 4 fields for the bbox (x, y, w, h)
            - 1 field for the class
    """
    # Removing background class since we don't want to find objects for it
    no_background_class = rois_class_scores[:, 1:]
    roi_class_scores_shape = tf.shape(no_background_class)
    # Reshaping class scores. Example: [[0.3, 0.6], [0.6, 0.3]] -> [[[0.3], [0.6]], [[[0.6], [0.3]]]
    # In the example above we have two rois, two classes (after removing background)
    # We do this to be able to concat it to the reg target tensor
    reshaped_class_scores = tf.reshape(
        no_background_class, shape=[roi_class_scores_shape[0], roi_class_scores_shape[1], 1])
    # Generating tensor of class indices that will be included in the roi information
    # This generates a tensor with content [[1], [2], .., [20]]
    class_indices = tf.reshape(tf.tile(tf.to_float(
        tf.range(1, number_classes)), multiples=[roi_class_scores_shape[0]]),
        shape=[roi_class_scores_shape[0], roi_class_scores_shape[1], 1])
    # Concat reg targets (we have one group per roi per class) with the corresponding
    # class scores and with the class index
    # There are 6 elements together now:
    # 4 regression target fields for class and roi + score for class and roi + class index
    all_roi_info_per_class = \
        tf.concat([rois_reg_targets, reshaped_class_scores, class_indices], axis=2)
    # Finding transpose so instead of having rows representing rois, we have rows
    # representing classes
    # After this operation, there will be NUMBER_CLASSES rows, each with NUMBER_ROIS elements
    # inside, each with 5 fields inside = 4 reg targets + score for that class and roi
    per_class_rows = tf.transpose(all_roi_info_per_class, perm=[1, 0, 2])

    # Initial values for while loop variables
    i0 = tf.constant(0)
    detected_objects = tf.zeros([1, 5])

    # Loop vars contains the initial value of the variable that we will use to iterate.
    # This while loop takes two variables: class counter i, current list of detected objects
    # At every iteration, it first checks if we have covered all the classes already (i < # classes)
    # If there are still classes to cover, it calls method detect_objects_for_class to find
    # objects for that class
    # Each iteration must return the next value for the while loop
    # variables (i + 1, list of detected objects with new ones appended)
    detected_objects = tf.while_loop(
                         # Iteration condition (we are going to traverse all classes)
                         lambda i, objects: i < number_classes - 1,
                         # For each row/class, detect objects of that class
                         lambda i, objects:
                         [i + 1,
                          detect_objects_for_class(
                              objects, rois, per_class_rows[i, :, :],
                              max_output_size, score_threshold)
                          ],
                         # The initial values of the while iteration variables
                         loop_vars=[i0, detected_objects],
                         # Because we are going to keep appending detected objects to the final
                         # tensor, its shape is going to change, so we specify None
                         shape_invariants=[i0.get_shape(), tf.TensorShape([None, 5])])

    # We get rid of the while counter i (detected_objects[0]) and the initialization tensor full of
    # zeros
    return detected_objects[1][1:, :]


def detect_objects_for_class(
        previous_detected_objects, rois, class_rois_info, max_output_size, score_threshold):
    # Shape = (# rois, 4 fields = x1,y1, x2, y2)
    class_rois_detection_boxes = find_bboxes_from_offsets(rois, class_rois_info)
    # 1-D tensor with all the rois scores for this class
    class_scores = class_rois_info[:, 4]
    # Tensor with the index of the current class. Example: For class person, index = 1
    class_indices = class_rois_info[:, 5:6]

    object_bbox_and_class = tf.concat([class_rois_detection_boxes, class_indices], axis=1)

    # Non-max suppression: keeping only objects with highest score among those that overlap more
    # than a threshold
    # Returns a list of indices of the bboxes selected
    # If scores for all rois are too low, it wont include objects of that class in the detection
    non_max_suppression_indices = tf.image.non_max_suppression(
        class_rois_detection_boxes, class_scores,
        max_output_size=max_output_size, score_threshold=score_threshold, iou_threshold=0.3)

    # Finding bboxes from indices
    extract_object_bboxes_by_indices = \
        tf.gather(object_bbox_and_class, indices=non_max_suppression_indices)

    # If no objects detected for this class, keep the list of already detected objects the same
    # If there are new detected objects for this class, append them to the list of already detected
    # ones (previous classes)
    return tf.cond(tf.equal(tf.size(extract_object_bboxes_by_indices), 0),
                   lambda: previous_detected_objects,
                   lambda: tf.concat(
                       [previous_detected_objects, extract_object_bboxes_by_indices], axis=0))


def find_bboxes_from_offsets(rois, rois_reg_targets):
    rois_x = rois[:, 0:1]
    rois_y = rois[:, 1:2]
    rois_w = rois[:, 2:3]
    rois_h = rois[:, 3:4]

    tx = rois_reg_targets[:, 0:1]
    ty = rois_reg_targets[:, 1:2]
    tw = rois_reg_targets[:, 2:3]
    th = rois_reg_targets[:, 3:4]

    gx = tf.add(tf.multiply(rois_w, tx), rois_x)
    gy = tf.add(tf.multiply(rois_h, ty), rois_y)
    gw = tf.multiply(rois_w, tf.exp(tw))
    gh = tf.multiply(rois_h, tf.exp(th))

    x2 = tf.subtract(tf.add(gx, gw), 1)
    y2 = tf.subtract(tf.add(gy, gh), 1)

    # Return [x1, y1, x2, y2] representing the diagonal bbox coordinates
    return tf.concat([tf.round(gx), tf.round(gy), tf.round(x2), tf.round(y2)], axis=1)
