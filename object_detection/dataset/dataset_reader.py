import pickle
import random
import numpy as np


class DatasetReader:
    def __init__(self, input_files, number_images_in_batch, number_rois_per_image,
            max_foreground_rois_per_image):
        """
        Creates object that will be used to get mini-batches for rcnn training and prediction

        :param
            input_files: list of files that contains all the entries of the dataset
        """
        self.files = input_files
        self.current_file = 0

        self.number_images_in_batch = number_images_in_batch
        self.number_rois_per_image = number_rois_per_image
        self.max_foreground_rois_per_image = max_foreground_rois_per_image

        with open(input_files[self.current_file], 'rb') as fo:
            # Each item (dictionary) in the list returned by load method contains the
            # following fields:
            #
            # image - Pixels of the resized input image
            # gt_bboxes - Information about the ground truth boxes for the image (not needed here)
            # rois - Information about foreground rois. Contains bbox, class and reg target
            # rois_background - Information about background rois. Contains bbox and class
            self.data = pickle.load(fo)

        # next_records represents the row that marks the beginning of the next batch
        self.next_record = 0
        self.total_records = len(self.data)

    def get_batch(self):
        """
        Generates a mini-batch that consists of NUMBER_IMAGES images and ROIS_PER_IMAGE rois
        for each one of them
        """
        # If we already processed the entire last file, the mini-batch will be empty
        if self.next_record >= self.total_records and self.current_file >= len(self.files) - 1:
            return {}

        # If we are at the end of the current file and there are more input files to be processed,
        # we moved to the next file
        if self.next_record >= self.total_records:
            # Current file now points to the next input file
            self.current_file += 1
            # We load the next input file into memory
            with open(self.files[self.current_file], 'rb') as fo:
                self.data = pickle.load(fo)
            # We set the current position at the beginning of the new file
            self.next_record = 0
            # Updating the total records in file, since the new file can have a different size
            self.total_records = len(self.data)

        # If remaining images in current file < NUMBER_IMAGES, we only process those at the end
        # of the file, not moving to the next file to compensate the missing one (mini-batch will
        # be shorter)
        if self.total_records - self.next_record < self.number_images_in_batch:
            records_to_fetch = self.total_records - self.next_record
        else:
            records_to_fetch = self.number_images_in_batch

        # Getting the corresponding number of input images from the file
        # The format of each image is a dictionary with all the necessary fields to do RCNN.
        # We'll need to reformat the data to create separate batches for the different
        # components (images, rois, class labels, reg labels)
        data_batch = self.data[self.next_record:self.next_record + records_to_fetch]

        images_names_batch = np.array([image_data["image_name"] for image_data in data_batch])
        images_batch = np.array([image_data["image"] for image_data in data_batch])
        rois_batch, class_labels_batch, reg_target_labels_batch = \
            self._find_rois_batch(data_batch)

        gt_objects_batch = np.array([image_data["gt_bboxes"] for image_data in data_batch])

        print "Batch: [size: {}, initial_index: {}, final_index: {}, images: {}]"\
            .format(records_to_fetch, self.next_record, self.next_record + records_to_fetch,
                    str(images_names_batch))

        self.next_record = self.next_record + self.number_images_in_batch

        # Returning element at index 0 since we are only dealing with one image in this version
        # This class is capable of managing more images per class
        return {"images": images_batch,
                "images_names": images_names_batch,
                "rois": rois_batch[0],
                "class_labels": class_labels_batch[0],
                "reg_target_labels": reg_target_labels_batch[0],
                "gt_objects": gt_objects_batch[0]}

    def _find_rois_batch(self, data_batch):
        """
        Given a batch of images, this method generates batches for rois belonging to those images.

        There are three mini-batches generated here:
           - rois bbox mini-batch: bboxes representing rois for the images
           - rois class labels: class labels for each of the rois
           - rois reg target labels: regression target labels for each of the rois
        """
        # Each list below will contain NUMBER_IMAGES items inside
        # Each item inside is a list with ROIS_PER_IMAGE items
        rois_batch = []
        class_labels_batch = []
        reg_target_labels_batch = []

        for image in data_batch:
            dataset_batch = []
            # Getting all the foreground and background rois coming from the dataset
            # images in the batch
            foreground_rois = image["rois"]
            background_rois = image["rois_background"]

            # If there less or the same number of foreground rois than the max expected, then
            # include them all in the rois batch.
            # If there are more foreground images than the max, then randomly pick
            # MAX_FOREGROUND_ROIS_PER_IMAGE of them
            if len(foreground_rois) <= self.max_foreground_rois_per_image:
                dataset_batch.extend(foreground_rois)
                max_background_images = self.number_rois_per_image - len(foreground_rois)
            else:
                dataset_batch.extend([random.choice(foreground_rois)
                                      for _ in range(self.max_foreground_rois_per_image)])
                max_background_images = \
                    self.number_rois_per_image - self.max_foreground_rois_per_image

            # Adding the remaining number of rois for the batch using background rois
            dataset_batch.extend([random.choice(background_rois)
                                  for _ in range(max_background_images)])

            # Randomly permutating the list of rois to avoid that all the foregrounds ones are at
            # the beginning and background ones at the end
            dataset_batch = np.random.permutation(dataset_batch)

            # Adding list that contains ROIS_PER_IMAGE elements to each of the batches
            # Append method adds the list to a list -> [1, 2, 3].append([4, 5]) result in
            # [1, 2, 3, [4, 5]]
            rois_batch.append([roi["bbox"] for roi in dataset_batch])
            class_labels_batch.append([roi["class"] for roi in dataset_batch])
            reg_target_labels_batch.append([roi["reg_target"] for roi in dataset_batch])

        return np.array(rois_batch), np.array(class_labels_batch), np.array(reg_target_labels_batch)
