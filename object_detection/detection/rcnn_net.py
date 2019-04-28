import tensorflow as tf

import detect_and_classify_instances as pc
import rcnn_multitask_loss as mloss
import reduced_resnet_builder
import roi_pooling_layer


def get_net(
        number_classes, number_regression_fields, number_resnet_layers, number_hidden_nodes,
        image_input_batch, roi_input_batch, class_label_batch, detection_label_batch,
        learning_rate):
    """
    This function generates the tensorflow network that will be used for object detection

    :param number_classes: number of classes that the network will be predicting
    :param number_regression_fields: number of regression fields calculated by this network
    :param number_resnet_layers: number of resnet layers
    :param number_hidden_nodes: number of hidden units to use for fully connected layers
    :param image_input_batch: batch of input images
    :param roi_input_batch: batch of input rois
    :param class_label_batch: batch of class labels
    :param detection_label_batch: batch of detection labels
    :param learning_rate: learning rate to use

    :return: 3 operations to be called within a session:
        multitask loss op, training op and detection op
    """
    he_init = tf.contrib.layers.variance_scaling_initializer()

    # Base network including resnet and roi pooling layer
    base_net, _ = get_base_net(
        number_resnet_layers, number_hidden_nodes, image_input_batch, roi_input_batch, he_init)

    # Two separate branches for rois classification and regression (detection)
    classification_branch = get_classification_branch(
        number_classes, base_net, he_init)

    detection_branch = get_detection_branch(
        number_classes, number_regression_fields, base_net, he_init)

    # Combined loss for classification and detection
    multitask_loss = get_multitask_loss(
        classification_branch, detection_branch, class_label_batch, detection_label_batch)

    training = get_training(learning_rate, multitask_loss)

    test = get_predicted_objects(
        number_classes, roi_input_batch, classification_branch, detection_branch)

    return multitask_loss, training, test


def get_base_net(
        number_resnet_layers, number_hidden_nodes, image_input_batch, roi_input_batch, he_init):
    """
    This function generated the base of the whole rcnn network including the core resnet body and
    the roi pooling layer. This base net doesn't include the classification and detection branches

    :param number_resnet_layers: number of resnet layers
    :param number_hidden_nodes: number of hidden units to use for fully connected layers
    :param image_input_batch: batch of input images
    :param roi_input_batch: batch of input rois
    :param he_init: he kernel init

    :return: last operation to be used before connecting the classification and detection branches
    and roi polling operation
    """
    # Resnet network
    resnet = reduced_resnet_builder.ReducedResnetBuilder(he_init) \
        .build_resnet(image_input_batch, number_resnet_layers)

    # Forcing it to use just ONE image in this version of the software
    pooling_layer = get_roi_pooling_layer(roi_input_batch, resnet[0])

    # Reshaping output so we can feed it to fully connected layers (everything in a long vector)
    pool2_flat = tf.reshape(pooling_layer, [-1, 7 * 7 * 64])

    fc_layer_1 = tf.layers.dense(
        pool2_flat, number_hidden_nodes, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
        name="BaseNet-FC1")

    batch_norm_1 = tf.layers.batch_normalization(fc_layer_1, name="BaseNet-Batch1")

    fc_layer_2 = tf.layers.dense(
        batch_norm_1, number_hidden_nodes, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
        name="BaseNet-FC2")

    batch_norm_2 = tf.layers.batch_normalization(fc_layer_2, name="BaseNet-Batch2")

    return batch_norm_2, pooling_layer


def get_roi_pooling_layer(roi_input_batch, resnet_output):
    """
    This function creates a roi pooling layer

    :param roi_input_batch:batch of input rois
    :param resnet_output: last operation from resnet

    :return: roi pooling layer
    """
    # HxW = 7x7; the ratio of pixels in original image to roi pooling output is 4
    return roi_pooling_layer \
        .RoiPoolingLayer(resnet_output, roi_input_batch, 7, 7, 4).get_roi_pooling_layer()


def get_classification_branch(number_classes, previous_output, he_init):
    """
    This function creates the classification branch used in RCNN to give each roi a score for the
    different classes

    :param number_classes: the number of classes is the number of scores that this branch will
    generate per roi
    :param previous_output: previous operation that this branch will be connected to
    :param he_init: he kernel init

    :return: softmax over all classes for the roi batch
    """
    class_fc = tf.layers.dense(
        previous_output, number_classes, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
        name="Logits")
    return tf.nn.softmax(class_fc)


def get_detection_branch(number_classes, number_regression_fields, previous_output, he_init):
    """
    This function creates the object detection branch used in RCNN to calculate regression targets
    for each roi and class. Regression targets are values used to calculate the final object
    position from the roi coordinates for a given class. If there are 20 classes not including
    background, this branch will predict 20 regression target groups for each roi.

    :param number_classes: the number of classes - 1 is the number of regression target groups
    that this branch will calculate for each roi. Background class is not included.
    :param number_regression_fields: Number of values that this branch will output
    :param previous_output: previous operator that this branch will be connected to
    :param he_init: he kernel init

    :return: number_classes - 1 regression target groups per roi in the batch
    """
    detection_fc = tf.layers.dense(
        previous_output, number_classes, activation=tf.nn.leaky_relu, kernel_initializer=he_init)

    batch_norm_1 = tf.layers.batch_normalization(detection_fc, name="Detection-Batch1")

    # The output has to be 4 regression numbers for each class that is not background
    detection_regressor = tf.layers.dense(
        batch_norm_1, number_regression_fields * (number_classes - 1),
        activation=tf.nn.leaky_relu, kernel_initializer=he_init, name="DetectionFields")
    # So far we have all the regression targets together in a vector for all classes. We need to
    # convert that into a matrix where rows represents classes and columns represent the predicted
    # regression targets
    detection_regressor_shape = tf.shape(detection_regressor)
    return tf.reshape(
        detection_regressor,
        [detection_regressor_shape[0], number_classes - 1, number_regression_fields])


def get_multitask_loss(
        class_softmax, detection_regressor, class_label_batch, detection_label_batch):
    """
    This function creates the tensorflow operation that calculates the combined loss for
    classification and detection

    :param class_softmax: softmax scores for the rois in the batch
    :param detection_regressor: regression targets for the rois in the batch
    :param class_label_batch: class labels
    :param detection_label_batch: regression target labels

    :return: combined loss for classification and detection (formula from fast rcnn paper)
    """
    # Combined loss for classification and detection
    return mloss.RCNNMultitaskLoss(
        class_predictions=class_softmax,
        detection_predictions=detection_regressor,
        class_labels=class_label_batch,
        detection_labels=detection_label_batch) \
        .multitask_loss()


def get_training(learning_rate, multitask_loss):
    """
    This function returns the tensorflow operator to train the network using gradient descent with
    Adam optimizer

    :param learning_rate: learning rate value to use in the current training iteration
    :param multitask_loss: loss to use for learning

    :return: operator to train network
    """
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(multitask_loss)


def get_predicted_objects(number_classes, roi_input_batch, class_softmax, detection_regressor):
    """
    This function detects and classifies objects in images using non-max suppression across all
    rois in the batch

    :param number_classes: number of classes to use for classification and detection
    :param roi_input_batch: batch of input rois
    :param class_softmax: softmax scores for the rois in the batch
    :param detection_regressor: regression targets for the rois in the batch

    :return: detected objects along with their class
    """
    # Extracting predicted objects with bboxes and classes (non-max suppression)
    return pc.detect_and_classify(
        roi_input_batch, class_softmax, detection_regressor, 4, 0.5, number_classes)
