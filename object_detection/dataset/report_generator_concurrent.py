from multiprocessing import Pool
from os import getpid
from os import listdir
import dataset.xml_parser as xml_parser
import itertools

import dataset.dataset_generator as dataset_generator

INPUT_FOLDER = "../../../datasets/images/pascal-voc/separated/training/"

IMAGE_FOLDER = INPUT_FOLDER + "image/"
ANNOTATION_FOLDER = INPUT_FOLDER + "annotation/"

OUTPUT_FOLDER = "../reports/"

CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane",
           "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable",
           "pottedplant", "sofa", "tvmonitor"]


def calculate_number_rois(image_folder, annotation_folder, output_folder):
    """
    Generates a report (file) with information about the rois for each image, including:
    - Name of the file
    - Total number of rois
    - Number of foreground rois
    - Number of background rois

    :param
        image_folder: folder that contains all the images it will generate rois for
        annotation_folder: folder that contains all the annotations for the images
        output_folder: output folder where the report will be generated
    """
    def task(image_path, annotation_path, output_folder):
        """
          Subprocess task that finds the ROIs for a given image and writes the information to the
          thread's report file
        """
        output_file = "{}roi_info_per_image_{}".format(output_folder, getpid())

        with open(output_file, 'a') as f:
            data = dataset_generator.get_image_data_training(image_path, annotation_path)
            foreground_rois = len(data["rois"])
            background_rois = len(data["rois_background"])
            roi_info_image = {"image": image_path, "rois": foreground_rois + background_rois,
                              "foreground": foreground_rois, "background": background_rois}
            f.write(str(roi_info_image) + "\n")

            print("Done processing image: " + image_path)

    pool = Pool(processes=10)
    # Generate report info for each combination of image and annotation
    images = sorted(listdir(image_folder))
    annotations = sorted(listdir(annotation_folder))

    for file_pair in zip(images, annotations):
        # Finding paths to image and annotation
        image_path = image_folder + file_pair[0]
        annotation_path = annotation_folder + file_pair[1]
        # Submitting tasks to process pool
        pool.apply_async(task, (image_path, annotation_path, output_folder))

    pool.close()
    pool.join()


def get_number_images_with_class(annotation_folder):
    """
    This function generates a dictionary with the following information:
       - Key: Combination of 1, 2 or 3 classes
       - Value: Number of images with that combination of classes

    Example:
       If classes =
           ['person', 'dog', 'cat']
       Possible result =
           {'person': 4, 'dog': 2, 'cat': 2, "cat-person": 1, "cat-dog": 1, "dog-person": 2}

    :param annotation_folder: folder with all the annotation files
    :return: dictionary with class counts
    """
    annotations = listdir(annotation_folder)

    # Creating list with all combinations of 3 elements + all combinations of 2
    # elements + single classes.
    # We'll find counts of images with objects belonging to those class combinations
    class_combinations = \
        {key: 0 for key in CLASSES}
    class_combinations.update(
        {"-".join(sorted(key)): 0 for key in itertools.combinations(CLASSES, 2)})
    class_combinations.update(
        {"-".join(sorted(key)): 0 for key in itertools.combinations(CLASSES, 3)})

    for annotation in annotations:
        annotation_path = annotation_folder + annotation
        # Get all different classes in the current annotation file
        object_classes = \
            {xml_object["class"] for xml_object in xml_parser.parse_xml(annotation_path)}

        # We are only interested in images with up to three different classes
        if len(object_classes) <= 3:
            key = "-".join(sorted(object_classes))
            class_combinations[key] = class_combinations[key] + 1

    for combination in class_combinations.items():
        if combination[1] != 0:
            print(combination)

    return class_combinations

if __name__ == '__main__':
    get_number_images_with_class(ANNOTATION_FOLDER)
    #calculate_number_rois(IMAGE_FOLDER, ANNOTATION_FOLDER, OUTPUT_FOLDER)
