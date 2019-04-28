import config.config_reader as config_reader
import pytest


class TestConfigReader(object):

    @pytest.fixture
    def config_reader_object(self):
        return config_reader.ConfigReader("test/data/test_config.ini")

    def test_get_model_train(self, config_reader_object):
        result = config_reader_object.get_model_train()
        assert result

    def test_get_model_test(self, config_reader_object):
        result = config_reader_object.get_model_test()
        assert not result

    def test_get_model_load(self, config_reader_object):
        result = config_reader_object.get_model_load()
        assert not result

    def test_get_model_save(self, config_reader_object):
        result = config_reader_object.get_model_save()
        assert result

    def test_get_model_path(self, config_reader_object):
        result = config_reader_object.get_model_path()
        assert result == "/tmp/fast-rcnn-model.ckpt"

    def test_get_number_epochs(self, config_reader_object):
        result = config_reader_object.get_number_epochs()
        assert result == 15

    def test_get_number_images_batch(self, config_reader_object):
        result = config_reader_object.get_number_images_batch()
        assert result == 1

    def test_get_number_rois_per_image_batch(self, config_reader_object):
        result = config_reader_object.get_number_rois_per_image_batch()
        assert result == 64

    def test_get_number_max_foreground_rois_per_image_batch(self, config_reader_object):
        result = config_reader_object.get_number_max_foreground_rois_per_image_batch()
        assert result == 16

    def test_get_number_image_channels(self, config_reader_object):
        result = config_reader_object.get_number_image_channels()
        assert result == 3

    def test_get_number_image_pixels(self, config_reader_object):
        result = config_reader_object.get_number_image_pixels()
        assert result == 600

    def test_get_number_resnet_layers(self, config_reader_object):
        result = config_reader_object.get_number_resnet_layers()
        assert result == 15

    def test_get_number_hidden_nodes(self, config_reader_object):
        result = config_reader_object.get_number_hidden_nodes()
        assert result == 800

    def test_get_number_classes(self, config_reader_object):
        result = config_reader_object.get_number_classes()
        assert result == 21

    def test_get_number_regression_fields(self, config_reader_object):
        result = config_reader_object.get_number_regression_fields()
        assert result == 4

    def test_get_roi_bbox_fields(self, config_reader_object):
        result = config_reader_object.get_roi_bbox_fields()
        assert result == 4

    def test_get_learning_rate_initial_value(self, config_reader_object):
        result = config_reader_object.get_learning_rate_initial_value()
        assert result == 0.001

    def test_get_learning_rate_manager_threshold(self, config_reader_object):
        result = config_reader_object.get_learning_rate_manager_threshold()
        assert result == 0.6

    def test_get_learning_rate_manager_steps(self, config_reader_object):
        result = config_reader_object.get_learning_rate_manager_steps()
        assert result == 80

    def test_get_logs_path(self, config_reader_object):
        result = config_reader_object.get_logs_path()
        assert result == "/tmp/tensorflow_logs/rcnn-detector/"

    def test_get_test_output_file(self, config_reader_object):
        result = config_reader_object.get_test_output_file()
        assert result == "./output/result.txt"

    def test_get_training_error_file(self, config_reader_object):
        result = config_reader_object.get_training_error_file()
        assert result == "./output/error.txt"