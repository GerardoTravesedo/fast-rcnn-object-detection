import tensorflow as tf


class RCNNMultitaskLoss:
    def __init__(self, class_predictions, detection_predictions, class_labels, detection_labels):
        """
        :param class_predictions: batch of classification scores for the different rois
        :param detection_predictions: batch of predicted regression targets for the different rois
        :param class_labels: batch of classification labels for the different rois
        :param detection_labels: batch of regression target labels for the different rois
        """
        self._class_predictions = class_predictions
        self._detection_predictions = detection_predictions
        self._class_labels = class_labels
        self._detection_labels = detection_labels
        self._classification_loss = 0.
        self._detection_loss = 0.

    def multitask_loss(self):
        """
        :return: tensorflow operator to calculate combined classification and detection loss
        """
        self._classification_loss = self._get_classification_loss()
        self._detection_loss = self._get_detection_loss()
        return self._get_total_loss()

    def _get_classification_loss(self):
        # Notice that we use labels as the weights, so it only keeps the log value
        # for the true class
        log_loss = tf.losses.log_loss(
            self._class_labels, self._class_predictions,
            weights=self._class_labels, reduction="none")
        # From [[ 0. 0. 0.5108254  0.]] to [[0.5108254]]
        return tf.reduce_sum(log_loss, 1, keepdims=True)

    def _get_detection_loss(self):
        # It keeps the first dimension as is (rois), but removes the first column of the second
        # dimension (background class)
        class_labels_no_background = self._class_labels[:, 1:]
        # Generating a tensor with same shape as the predicted regression targets
        # This vector will be used to calculate the loss only for the true classes
        # Example: [[0, 0], [0, 1]] -- reshape --> [[[0], [0]], [[0], [1]]] -- tile -->
        # --> [[[0 0 0 0], [0 0 0 0]], [[0 0 0 0], [1 1 1 1]]]
        class_labels_shape = tf.shape(class_labels_no_background)
        reshaped_class_labels = tf.reshape(class_labels_no_background,
                                           shape=(class_labels_shape[0], class_labels_shape[1], 1))
        extended_class_labels = tf.tile(reshaped_class_labels, [1, 1, 4])

        only_true_class_predicted_reg_targets = tf.reduce_sum(
            tf.multiply(tf.to_float(extended_class_labels), self._detection_predictions), 1)

        # Subtracting the predicted reg target for the true class and the true reg target
        reg_target_subtract = \
            tf.subtract(only_true_class_predicted_reg_targets, self._detection_labels)

        # Map over all the regression values (differences between predicted and true value)
        # For each regression value, it calculates smooth l1 over it
        def smooth_l1(x):
            return tf.map_fn(lambda reg_value:
                             tf.cond(tf.less(tf.abs(reg_value), 1.),
                                     lambda: tf.multiply(0.5, tf.pow(reg_value, 2)),
                                     lambda: tf.subtract(tf.abs(reg_value), 0.5)), x)

        # Map over all the rois
        smooth_l1 = tf.map_fn(smooth_l1, reg_target_subtract)

        # The final detection loss is the addition of all the smooth l1 values for the roi
        return tf.reduce_sum(smooth_l1, 1, keepdims=True)

    def _get_total_loss(self):
        return tf.add(self._classification_loss, self._detection_loss)
