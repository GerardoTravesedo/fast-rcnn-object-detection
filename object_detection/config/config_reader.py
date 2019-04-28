import ConfigParser

NET_SECTION = "Net"

MODEL_SECTION = "Model"

OUTPUT_SECTION = "Output"


class ConfigReader:

    def __init__(self, prop_file):
        config = ConfigParser.ConfigParser()
        config.read(prop_file)
        self._config = config

    def get_model_train(self):
        return self._config.getboolean(MODEL_SECTION, "model.train")

    def get_model_test(self):
        return self._config.getboolean(MODEL_SECTION, "model.test")

    def get_model_load(self):
        return self._config.getboolean(MODEL_SECTION, "model.load")

    def get_model_save(self):
        return self._config.getboolean(MODEL_SECTION, "model.save")

    def get_model_path(self):
        return self._config.get(MODEL_SECTION, "model.path")

    def get_number_epochs(self):
        return self._config.getint(NET_SECTION, "number.epochs")

    def get_number_images_batch(self):
        return self._config.getint(NET_SECTION, "number.images.batch")

    def get_number_rois_per_image_batch(self):
        return self._config.getint(NET_SECTION, "number.rois.per.image.batch")

    def get_number_max_foreground_rois_per_image_batch(self):
        return self._config.getint(
            NET_SECTION, "number.max.foreground.rois.per.image.batch")

    def get_number_image_channels(self):
        return self._config.getint(NET_SECTION, "number.image.channels")

    def get_number_image_pixels(self):
        return self._config.getint(NET_SECTION, "number.image.pixels")

    def get_number_resnet_layers(self):
        return self._config.getint(NET_SECTION, "number.resnet.layers")

    def get_number_hidden_nodes(self):
        return self._config.getint(NET_SECTION, "number.hidden.nodes")

    def get_number_classes(self):
        return self._config.getint(NET_SECTION, "number.classes")

    def get_number_regression_fields(self):
        return self._config.getint(NET_SECTION, "number.regression.fields")

    def get_roi_bbox_fields(self):
        return self._config.getint(NET_SECTION, "number.roi.bbox.fields")

    def get_learning_rate_initial_value(self):
        return self._config.getfloat(NET_SECTION, "learning.rate.initial.value")

    def get_learning_rate_manager_threshold(self):
        return self._config.getfloat(NET_SECTION, "learning.rate.manager.threshold")

    def get_learning_rate_manager_steps(self):
        return self._config.getint(NET_SECTION, "learning.rate.manager.steps")

    def get_logs_path(self):
        return self._config.get(OUTPUT_SECTION, "logs.path")

    def get_test_output_file(self):
        return self._config.get(OUTPUT_SECTION, "test.output.file")

    def get_training_error_file(self):
        return self._config.get(OUTPUT_SECTION, "training.error.file")

