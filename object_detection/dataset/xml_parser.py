import xml.etree.ElementTree as ET


def parse_xml(xml_file):
    """
    This function extracts information from a PASCAL VOC image annotation file

    :param xml_file: path to the annotation file

    :return: dictionary with the extracted information, which includes the bboxes of the different
    objects in the image and their classes
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []

    for object in root.findall("object"):
        xml_info = {}

        type_tag = object.find("name")
        xml_info["class"] = type_tag.text

        bbox_tag = object.find("bndbox")
        bbox_info = {"xmin": int(bbox_tag.find("xmin").text),
                     "xmax": int(bbox_tag.find("xmax").text),
                     "ymin": int(bbox_tag.find("ymin").text),
                     "ymax": int(bbox_tag.find("ymax").text)}

        xml_info["bbox"] = bbox_info

        objects.append(xml_info)

    return objects


def contains_valid_classes(xml_file, filter_classes):
    """
    This function indicates if a given image annotation contains only valid classes

    :param xml_file: path to the annotation file
    :param filter_classes: list with valid classes

    :return: true if the annotation only contains objects of valid classes
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for object in root.findall("object"):
        if object.find("name").text not in filter_classes:
            return False

    return True
