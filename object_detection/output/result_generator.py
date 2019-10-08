import result_analysis.classification_analysis as ca
import dataset.roi_tools as rt
import tools.output_analyzer as oa

# TODO: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59


class ResultGenerator(object):

    def __init__(self, output_folder):
        self.class_labels = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat",
                             "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant",
                             "sofa", "tvmonitor"]
        self.metric_container = ca.ClassificationAnalysis(self.class_labels)
        self.output_folder = output_folder

    def add_record(self, test_batch, predicted_info):
        gt_boxes = [gt_object["bbox"] for gt_object in test_batch["gt_objects"]]
        gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
        gt_classes = [gt_object["class"] for gt_object in test_batch["gt_objects"]]

        predicted_boxes = predicted_info[:, :4]
        predicted_classes = [rt.class_index_to_string(i) for i in predicted_info[:, 4]]

        oa.write_image_detection_predictions_to_file(
            self.output_folder + "/predictions.txt",
            self.output_folder + "/detection-images/",
            test_batch["images"][0],
            test_batch["images_names"][0].split("/")[-1],
            gt_boxes,
            gt_classes,
            predicted_boxes,
            predicted_classes)

        for index in range(len(gt_classes)):
            self.metric_container.add_record(gt_classes[index], predicted_classes[index])

    def generate_analysis_report(self):
        """
        Generates the following files:
            - metrics.txt containing precision, recall and F1 for every class
            - confusion-matrix.png containing a representation of the confusion matrix
        """
        with open(self.output_folder + "metrics.txt", "w+") as f:
            for label in self.class_labels:
                f.write("{class: " + label +
                        ", precision: " + self.metric_container.get_class_precision(label) +
                        ", recall: " + self.metric_container.get_class_recall(label) +
                        ", f1: " + self.metric_container.get_class_f1(label))

        self.metric_container.generate_confusion_matrix_heat_map(self.output_folder)
