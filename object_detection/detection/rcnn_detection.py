import time
from os import listdir

import tensorflow as tf

import config.config_reader as conf_reader
import learning_rate_manager as rm
import detection.rcnn_net as rcnn_net
import dataset.dataset_reader as ds_reader
import tools.output_analyzer as output_analyzer

from tensorflow.python.client import device_lib


class RCNNDetection:

    def __init__(self, session, config):
        """
        :param session: tensorflow session to run operations
        :param config: rcnn configuration file path
        """
        # Shape of this placeholder is generally (2, 600, 600, 3), but when the last batch in a
        # file is read, the batch size can be less than 2 (just one image left).
        # That's why we specify None
        # image_input_batch = tf.placeholder(
        #     tf.float32, shape=(None, CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS),
        #     name="ImageInputBatch")

        self._config = config

        # We are only using one image per batch in this version
        # Shape = (1, 600, 600, 3)
        self._image_input_batch = tf.placeholder(
            tf.float32, shape=
            (1, config.get_number_image_pixels(),
             config.get_number_image_pixels(),
             config.get_number_image_channels()),
            name="ImageInputBatch")

        # Each RoI has 4 values (x, y, h, w)
        self._roi_input_batch = tf.placeholder(
            tf.float32, shape=(None, config.get_roi_bbox_fields()),
            name="RoiInputBatch")

        # Class labels batch with shape = (# rois, # classes)
        self._class_label_batch = tf.placeholder(
            tf.float32, shape=(None, config.get_number_classes()),
            name="ClassLabelsBatch")

        # Detections labels batch with shape = (# rois, 4)
        self._detection_label_batch = tf.placeholder(
            tf.float32, shape=(None, config.get_number_regression_fields()),
            name="DetectionLabelsBatch")

        self._learning_rate = tf.placeholder(tf.float32, name="LearningRate")

        self._sess = session

    def get_net(self):
        """
        :return: tensorflow operators to trigger training and testing
        """
        return rcnn_net.get_net(
            self._config.get_number_classes(),
            self._config.get_number_regression_fields(),
            self._config.get_number_resnet_layers(),
            self._config.get_number_hidden_nodes(),
            self._image_input_batch,
            self._roi_input_batch,
            self._class_label_batch,
            self._detection_label_batch,
            self._learning_rate)

    def train_net(self, training, multitask_loss, training_batch_files):
        """
        This function trains the rcnn network

        :param training: tensorflow operator to train the network (will be run using the session)
        :param multitask_loss: tensorflow operator to get the result of the multitask loss. This info
        will be logged to be able to analyze it later
        :param training_batch_files: list of files to use to train the net
        """
        # Used to save and restore the model variables
        saver = tf.train.Saver()

        # If this model was already partially training before, load it from disk
        if self._config.get_model_load():
            # Restore variables from disk.
            saver.restore(self._sess, self._config.get_model_path())
            print("Model restored.")

        print("Starting training")
        training_start_time = time.time()

        iteration = 0
        learning_rate_manager = rm.LearningRateManager(
            self._config.get_learning_rate_initial_value(),
            self._config.get_learning_rate_manager_threshold(),
            self._config.get_learning_rate_manager_steps())

        for epoch in range(0, self._config.get_number_epochs()):
            print("Epoch: {0}".format(str(epoch)))
            # Training with all the PASCAL VOC records for each epoch
            # We train with 1 image per batch and 64 rois per image. From those 64, we'll use a max
            # of 16 foreground images. The rest will be background.
            training_reader = ds_reader.DatasetReader(
                training_batch_files,
                self._config.get_number_images_batch(),
                self._config.get_number_rois_per_image_batch(),
                self._config.get_number_max_foreground_rois_per_image_batch())
            training_batch = training_reader.get_batch()

            # Empty batch means we are done processing all images and rois for this epoch
            while training_batch != {}:
                _, loss = self._sess.run([training, multitask_loss], feed_dict={
                    self._image_input_batch: training_batch["images"],
                    self._roi_input_batch: training_batch["rois"],
                    self._class_label_batch: training_batch["class_labels"],
                    self._detection_label_batch: training_batch["reg_target_labels"],
                    self._learning_rate: learning_rate_manager.learning_rate
                })

                print("Error: {}".format(loss))

                # Logging information about the multitask loss to be able to analyze it later
                output_analyzer.write_error_to_file(
                    self._config.get_training_error_file(), iteration, loss)
                # Adding error to learning rate manager so it can calculate when to reduce it
                learning_rate_manager.add_error(loss)

                iteration = iteration + 1

                training_batch = training_reader.get_batch()

            # Save model variables to disk
            if self._config.get_model_save():
                save_path = saver.save(self._sess, self._config.get_model_path())
                print("Model saved in path: {0} for epoch {1}".format(save_path, epoch))
                print("Initial learning rate to use when training in the future: {0}"
                      .format(str(learning_rate_manager.learning_rate)))

        print("Done training. It took {0} minutes".format((time.time() - training_start_time) / 60))

    def test(self, prediction, test_batch_files):
        """
        This function detects and classifies objects in the given images

        :param prediction: tensorflow operator to detect objects (will be run using the session)
        :param test_batch_files: list of files to use to test the net
        """
        # If this model was already partially trained before, load it from disk
        # If model was trained in this same execution, the load already happened, so we
        # can skip it here
        if not self._config.get_model_train() and self._config.get_model_load():
            # Restore variables from disk.
            tf.train.Saver().restore(self._sess, self._config.get_model_path())
            print("Model restored.")

        print("Starting prediction")
        prediction_start_time = time.time()

        # It generates batches from the list of test files
        test_reader = ds_reader.DatasetReader(
            test_batch_files,
            self._config.get_number_images_batch(),
            self._config.get_number_rois_per_image_batch(),
            self._config.get_number_max_foreground_rois_per_image_batch())
        test_batch = test_reader.get_batch()

        while test_batch != {}:
            predicted_classes = self._sess.run(prediction, feed_dict={
                self._image_input_batch: test_batch["images"],
                self._roi_input_batch: test_batch["rois"]
            })

            # Preparing some information for logging
            gt_boxes = [gt_object["bbox"] for gt_object in test_batch["gt_objects"]]
            gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
            gt_classes = [gt_object["class"] for gt_object in test_batch["gt_objects"]]

            # Logging information about the prediction to be able to analyze it later
            output_analyzer.write_image_detection_predictions_to_file(
                self._config.get_test_output_file(),
                "../output/detectionimages/",
                test_batch["images"][0],
                test_batch["images_names"][0].split("/")[-1],
                gt_boxes,
                gt_classes,
                predicted_classes[:, :4],
                predicted_classes[:, 4])
            test_batch = test_reader.get_batch()

        print("Done predicting. It took {0} minutes"
              .format((time.time() - prediction_start_time) / 60))


def run(properties_path, training_batch_files, test_batch_files):
    """
    Trains and tests the RCNN network

    :param properties_path: path to rcnn properties file
    :param training_batch_files: list of files to use to train
    :param test_batch_files: list of files to use to test
    """
    with tf.Session() as sess:
        config = conf_reader.ConfigReader(properties_path)

        should_train = config.get_model_train()
        should_test = config.get_model_test()
        should_load_model = config.get_model_load()
        should_save_model = config.get_model_save()

        if should_test and not should_load_model and not should_train:
            raise Exception(
                "Model cannot be tested without training the model or loading an existing one")

        if should_save_model and not should_train:
            raise Exception(
                "Model cannot be saved since it won't be modified")

        rcnn_detection = RCNNDetection(sess, config)

        multitask_loss_op, training_op, prediction_op = rcnn_detection.get_net()

        # Initialization has to happen after defining the graph
        sess.run(tf.global_variables_initializer())

        # In order to be able to see the graph, we need to add this line after the graph is defined
        tf.summary.FileWriter(config.get_logs_path(), graph=tf.get_default_graph())

        if should_train:
            rcnn_detection.train_net(training_op, multitask_loss_op, training_batch_files)

        if should_test:
            rcnn_detection.test(prediction_op, test_batch_files)

        print("Run the command line:\n"
              "--> tensorboard --logdir=/tmp/tensorflow_logs "
              "\nThen open http://2usmtravesed.local:6006/ into your web browser")


if __name__ == '__main__':
    print("Available devices: {}".format(device_lib.list_local_devices()))

    properties = "../config/config.ini"

    training_folder = "../../../datasets/images/pascal-voc/transformed/training-reduced/"
    #training_folder = "../../dataset-rcnn/hyper-reduced/"
    training_files = [training_folder + file_name for file_name in listdir(training_folder)]

    test_folder = "../../../datasets/images/pascal-voc/transformed/test-reduced/"
    #test_folder = "../../dataset-rcnn/hyper-reduced/"
    test_files = [test_folder + file_name for file_name in listdir(test_folder)]

    run(properties, training_files, test_files)
