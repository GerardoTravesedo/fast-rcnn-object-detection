import dataset.xml_parser as xml_parser


class TestXmlParser(object):

    def test_parse_xml_one_object(self):
        input_file = "test/data/xml_annotation_one_object.xml"
        data = xml_parser.parse_xml(input_file)

        assert len(data) == 1
        self._verify_xml_annotation(data[0], "person", 174, 101, 349, 351)

    def test_parse_xml_multiple_objects(self):
        input_file = "test/data/xml_annotation_multiple_objects.xml"
        data = xml_parser.parse_xml(input_file)

        assert len(data) == 2
        self._verify_xml_annotation(data[0], "person", 388, 194, 419, 339)
        self._verify_xml_annotation(data[1], "person", 415, 192, 447, 338)

    def test_is_person_valid(self):
        input_file = "test/data/xml_annotation_multiple_objects.xml"
        assert xml_parser.contains_valid_classes(input_file, ["person"])
        assert not xml_parser.contains_valid_classes(input_file, ["cat"])

    def _verify_xml_annotation(self, object_data, class_name, xmin, ymin, xmax, ymax):
        assert object_data["class"] == class_name
        assert object_data["bbox"]["xmin"] == xmin
        assert object_data["bbox"]["xmax"] == xmax
        assert object_data["bbox"]["ymin"] == ymin
        assert object_data["bbox"]["ymax"] == ymax
