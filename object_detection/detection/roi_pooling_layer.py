import tensorflow as tf


class RoiPoolingLayer:
    def __init__(
        self, feature_map, rois_batch, roi_pooling_height,
            roi_pooling_width, ratio_image_to_feature_map):
        """
        :param feature_map: feature map to use as input for pooling
        :param rois_batch: batch of rois
        :param roi_pooling_height: number of rows to split the feature map into to do max pooling
        :param roi_pooling_width: number of columns to split the feature map into to do max pooling
        :param ratio_image_to_feature_map: ratio of original image size to feature map size
        """
        self._feature_map = feature_map
        self._feature_map_shape = tf.shape(feature_map)
        self._roi_batch = rois_batch
        self._roi_pooling_height = roi_pooling_height
        self._roi_pooling_width = roi_pooling_width
        self._ratio_image_to_feature_map = float(ratio_image_to_feature_map)

    def get_roi_pooling_layer(self):
        """
        :return: tensorflow operator that does roi pooling
        """
        # Each roi in the batch has dimensions based on the original image size. Since we are
        # extracting the corresponding portion of the feature map, which is smaller than the
        # original size, we need to resize the rois
        ratio_rois = tf.scalar_mul(1 / self._ratio_image_to_feature_map, self._roi_batch)
        # If x or y are decimal, we find the floor to not lose anything
        # If w or h are decimal, we find the ceiling to not lose anything
        resized_rois =  \
            tf.concat([tf.floor(ratio_rois[:, 0:2]), tf.ceil(ratio_rois[:, 2:4])], axis=1)

        return tf.map_fn(lambda x: self._process_roi(x), resized_rois, infer_shape=False)

    def _process_roi(self, roi):
        x = tf.to_int32(roi[0])
        y = tf.to_int32(roi[1])
        w = tf.to_int32(roi[2])
        h = tf.to_int32(roi[3])

        # If roi is too small, we increase it to match the size HxW
        # For example if size of roi is 5x5 but roi pooling requires 7x7, we increase the roi size
        # It extends the roi to the right and bottom to get to the correct size. If the roi is next
        # to the right or bottom sides, it increases it in the other direction

        # If roi's width is too small, we make it the same as the pooling width
        x_length_diff = tf.abs(tf.subtract(self._roi_pooling_width, w))
        new_w = tf.cond(tf.less(w, self._roi_pooling_width),
                        lambda: tf.add(w, x_length_diff),
                        lambda: w)
        # If new width exceeds the length of the feature map, we move x to the left
        new_x = tf.cond(self._is_at_end_of_axis(x, new_w, 1),
                        lambda: tf.subtract(x, x_length_diff),
                        lambda: x)

        # If roi's height is too small, we make it the same as the pooling height
        y_length_diff = tf.abs(tf.subtract(self._roi_pooling_height, h))
        new_h = tf.cond(tf.less(h, self._roi_pooling_width),
                        lambda: tf.add(h, y_length_diff),
                        lambda: h)
        # If new height exceeds the length of the feature map, we move y up
        new_y = tf.cond(self._is_at_end_of_axis(y, new_h, 0),
                        lambda: tf.subtract(y, y_length_diff),
                        lambda: y)

        return self._roi_pooling(new_x, new_y, new_w, new_h)

    def _roi_pooling(self, x, y, w, h):
        # Extracting portion of feature map that belongs to this roi
        roi_crop = tf.image.crop_to_bounding_box(self._feature_map, y, x, h, w)
        # splitting and max reducing horizontal axis
        x_reduced = reduce_axis(roi_crop, w, self._roi_pooling_width, 1)
        # splitting and max reducing vertical axis
        return reduce_axis(x_reduced, h, self._roi_pooling_height, 0)

    def _is_at_end_of_axis(self, init_pixel, length, axis):
        last_pixel_roi = tf.add(init_pixel, length)
        last_pixel_axis = self._feature_map_shape[axis]
        return tf.greater(last_pixel_roi, last_pixel_axis)


def reduce_axis(crop, crop_size, pooling_size, axis):
    # Calculating the size of every chunk (horizontal or vertical depending on axis)
    # If w = 30 and W = 7, then we need 6 groups of 4 elements and a final one of 6
    # The size would be 4 because floor(30/7) = 4
    group_size = tf.to_int32(
        tf.floor(tf.scalar_mul(1 / float(pooling_size), tf.to_float(crop_size))))

    groups = []

    # Creating each of the chunks but the last one. We will deal with the last
    # one later because it could have a different size
    for group in range(0, pooling_size - 1):
        # Finding next chunk from the roi
        # We keep two dimensions the same because we are only splitting one axis
        new_group = []
        if axis == 0:
            new_group = crop[group * group_size:(group + 1) * group_size, :, :]
        elif axis == 1:
            new_group = crop[:, group * group_size:(group + 1) * group_size, :]
        # We max reduce each group for the specific axis
        # Example horizontal reduction:
        # [[[122, 2], [2, 3]], [[6, 7], [137, 8]]] -> [[[122, 3]], [[137, 8]]]
        new_group_max = tf.reduce_max(new_group, axis, keepdims=True)
        groups.append(new_group_max)

    # Doing the same than in the for loop but for the last group, which could
    # have a different size
    new_last_group = []
    if axis == 0:
        new_last_group = crop[(pooling_size - 1) * group_size:, :, :]
    elif axis == 1:
        new_last_group = crop[:, (pooling_size - 1) * group_size:, :]

    new_last_group_max = tf.reduce_max(new_last_group, axis, keepdims=True)
    groups.append(new_last_group_max)

    # At this point we have reduced one dimension. For example, if the shape of the input
    # was (4, 2, 2) and H = 2, we have a tensor with shape (2, 2, 2). Notice that we reduced
    # the y axis from 4 to 2

    # We split the original crop into chunks, reduced each one of them individually for the specific
    # axis, and now we concat them back to get a final reduced axis
    return tf.concat(groups, axis)
