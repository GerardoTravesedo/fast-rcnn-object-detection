import math
import pickle
from multiprocessing import Pool
from os import listdir

import dataset_generator

TRAINING_INPUT_FOLDER = "../../../datasets/images/pascal-voc/separated/training-reduced/"

TEST_INPUT_FOLDER = "../../../datasets/images/pascal-voc/separated/test-reduced/"

TRAINING_IMAGE_FOLDER = TRAINING_INPUT_FOLDER + "image/"
TRAINING_ANNOTATION_FOLDER = TRAINING_INPUT_FOLDER + "annotation/"

TEST_IMAGE_FOLDER = TEST_INPUT_FOLDER + "image/"
TEST_ANNOTATION_FOLDER = TEST_INPUT_FOLDER + "annotation/"

TRAINING_OUTPUT_FOLDER = "../../../datasets/images/pascal-voc/transformed/training-reduced/"

TEST_OUTPUT_FOLDER = "../../../datasets/images/pascal-voc/transformed/test-reduced/"

NUMBER_THREADS = 5
NUMBER_OUTPUT_FILES = 5


def task(paths, output_folder, task_id):
    """
    Subprocess task that finds the rcnn input data for each pair (image, annotation) and writes
    it into a file in pickle format
    """
    # print("Processing task {}".format(task_id))

    try:
        # Generate the rcnn input data for each image
        data = []
        for path in paths:
            print("Processing image {} in task {}".format(path[0], task_id))
            data.append(dataset_generator.get_image_data_training(path[0], path[1]))
            print("Done processing image {} in task {}".format(path[0], task_id))

        output_file = "{}rcnn_dataset_{}".format(output_folder, task_id)

        # Write the entire list of image data into a pickle file
        with open(output_file, 'wb') as f:
            print ("Ready to write result of task {}".format(task_id))
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Task {} is done. Number of images processed: {}".format(task_id, len(data)))
    except Exception:
        print("There was an exception while processing task {}".format(task_id))


def main(image_folder, annotation_folder, output_folder):
    """
    Creates a specific number of threads defined by NUMBER_THREADS to generate the rcnn input data
    from the original PASCAL VOC images and annotations

    :param
        image_folder: folder that contains all the images it will generate rois for
        annotation_folder: folder that contains all the annotations for the images
        output_folder: output folder where the report will be generated
    """
    # Creating pool of subprocesses
    pool = Pool(processes=NUMBER_THREADS)

    images = sorted(listdir(image_folder))
    number_images = len(images)
    print("Number of images to process {}".format(number_images))

    annotations = sorted(listdir(annotation_folder))
    number_annotations = len(annotations)
    print("Number of annotations to process {}".format(number_annotations))

    images_per_file = int(math.ceil(number_images / float(NUMBER_OUTPUT_FILES)))
    print("Each file will have a max of {} images".format(images_per_file))

    # Gets list of output batch files that were already generated
    already_generated_files = listdir(output_folder)
    # Gets the indices of the batch output files that were generated in previous executions
    # For example: [output_3, output_5] -> [3, 5]
    already_generated_indices = \
        [int(existing_file.split("_")[2]) for existing_file in already_generated_files]
    already_generated_indices.sort()
    print("Already generated file: {}".format(str(already_generated_indices)))
    # Pointer to the next batch that was already processed so we can use it later to know if the
    # batch to generate is already there and we can skip it
    next_batch_already_processed = 0
    print("Number of already generated output files: {}".format(len(already_generated_indices)))

    # Keeping track of the images that are grouped together so far. Once its length gets to
    # images_per_thread we can submit the task
    images_annotations_group = []
    task_counter = 0

    file_annotation_pairs = zip(images, annotations)

    # Generate rcnn input data for each combination of image and annotation
    for output_file_index in range(0, NUMBER_OUTPUT_FILES):
        if not already_generated_indices \
            or next_batch_already_processed >= len(already_generated_indices) \
                or output_file_index != already_generated_indices[next_batch_already_processed]:
            print("Output file with id {} was NOT generated previously"
                  .format(output_file_index))

            # Finding the range of images that fall under the current batch
            # For example, if current batch is 2 and the # of images per batch is 50, then the
            # range will be [2 * 50, 2 * 50 + 50] = [100, 150]
            # We include 150 in the example because when slicing by index, the last index is not
            # included
            first_file_index = output_file_index * images_per_file
            last_file_index = output_file_index * images_per_file + images_per_file

            if last_file_index <= len(file_annotation_pairs):
                print("Generating output file for images from {} to {}"
                      .format(first_file_index, last_file_index - 1))
            else:
                print("Generating output file for images from {} to {}"
                      .format(first_file_index, len(file_annotation_pairs) - 1))

            files_current_batch = file_annotation_pairs[first_file_index:last_file_index] \
                if last_file_index <= len(file_annotation_pairs) \
                else file_annotation_pairs[first_file_index:]

            for file_pair in files_current_batch:
                image_path = image_folder + file_pair[0]
                annotation_path = annotation_folder + file_pair[1]

                # Adding them to the current group as a tuple (image_path, annotation_path)
                # print("Image/Annotation pair: {} - {}".format(image_path, annotation_path))
                images_annotations_group.append((image_path, annotation_path))

            print("Submitting task with {} images and id {}".format(
                len(images_annotations_group), task_counter))
            pool.apply_async(task, (images_annotations_group, output_folder, task_counter))
            images_annotations_group = []
        else:
            print("Output file already generated with id {}".format(output_file_index))
            next_batch_already_processed += 1

        task_counter += 1

    print("Done submitting all tasks to threads")

    pool.close()
    pool.join()

if __name__ == '__main__':
    #training_image_folder = "../test/data/test-batch-reader-dataset/images/"
    #training_annotation_folder = "../test/data/test-batch-reader-dataset/annotations/"
    #training_output_folder = "../test/data/test-batch-reader-dataset/batch/"
    #main(TRAINING_IMAGE_FOLDER, TRAINING_ANNOTATION_FOLDER, TRAINING_OUTPUT_FOLDER)
    main(TEST_IMAGE_FOLDER, TEST_ANNOTATION_FOLDER, TEST_OUTPUT_FOLDER)
