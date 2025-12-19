"""Tests for XML generator."""

import xml.etree.ElementTree as ET

import pytest

from fuzz_generator.config import OutputSettings
from fuzz_generator.generators.xml_generator import XMLGenerator
from fuzz_generator.models.xml_models import (
    BlobElement,
    BlockElement,
    ChoiceElement,
    DataModel,
    NumberElement,
    StringElement,
)


class TestXMLGenerator:
    """Tests for XMLGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return XMLGenerator()

    def test_generator_creation(self):
        """Test generator can be created."""
        generator = XMLGenerator()
        assert generator.encoding == "utf-8"
        assert generator.indent == 4
        assert generator.include_comments is True

    def test_generator_with_settings(self):
        """Test generator with output settings."""
        settings = OutputSettings(
            encoding="utf-8",
            indent=2,
            include_comments=False,
        )
        generator = XMLGenerator(settings=settings)
        assert generator.indent == 2
        assert generator.include_comments is False

    def test_generate_simple_datamodel(self, generator):
        """Test generating simple DataModel."""
        model = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method", value="GET"),
                StringElement(name="Space", value=" ", token=True),
            ],
        )

        xml_str = generator.generate([model])

        # Verify XML validity
        root = ET.fromstring(xml_str)
        assert root.tag == "Secray"

        datamodel = root.find("DataModel")
        assert datamodel is not None
        assert datamodel.get("name") == "Request"

        strings = datamodel.findall("String")
        assert len(strings) == 2

    def test_generate_with_block(self, generator):
        """Test generating DataModel with Block."""
        models = [
            DataModel(
                name="CrLf",
                elements=[StringElement(name="End", value="\r\n", token=True)],
            ),
            DataModel(
                name="Request",
                elements=[
                    StringElement(name="Method"),
                    BlockElement(name="End", ref="CrLf"),
                ],
            ),
        ]

        xml_str = generator.generate(models)
        root = ET.fromstring(xml_str)

        datamodels = root.findall("DataModel")
        assert len(datamodels) == 2

        # Verify Block element
        request_dm = root.find(".//DataModel[@name='Request']")
        block = request_dm.find("Block")
        assert block is not None
        assert block.get("ref") == "CrLf"

    def test_generate_with_choice(self, generator):
        """Test generating DataModel with Choice."""
        model = DataModel(
            name="LineEnd",
            elements=[
                ChoiceElement(
                    name="EndChoice",
                    options=[
                        StringElement(name="CRLF", value="\r\n", token=True),
                        StringElement(name="LF", value="\n", token=True),
                    ],
                ),
            ],
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        choice = root.find(".//Choice")
        assert choice is not None
        assert len(choice.findall("String")) == 2

    def test_generate_with_number(self, generator):
        """Test generating DataModel with Number."""
        model = DataModel(
            name="Header",
            elements=[
                NumberElement(name="Length", size=32, endian="big"),
            ],
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        number = root.find(".//Number")
        assert number is not None
        assert number.get("name") == "Length"
        assert number.get("size") == "32"

    def test_generate_with_blob(self, generator):
        """Test generating DataModel with Blob."""
        model = DataModel(
            name="Body",
            elements=[
                BlobElement(name="Data", length=1024),
            ],
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        blob = root.find(".//Blob")
        assert blob is not None
        assert blob.get("name") == "Data"
        assert blob.get("length") == "1024"

    def test_xml_formatting(self, generator):
        """Test XML formatting."""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field")],
        )

        xml_str = generator.generate([model], indent=4)

        # Verify indentation
        lines = xml_str.split("\n")
        assert any(line.startswith("    ") for line in lines)

    def test_xml_no_formatting(self, generator):
        """Test XML without formatting."""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field")],
        )

        xml_str = generator.generate([model], indent=0)

        # Should be single line (no newlines in content)
        assert "<Secray>" in xml_str
        assert "</Secray>" in xml_str

    def test_xml_encoding(self, generator):
        """Test XML encoding."""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field", value="中文")],
        )

        xml_str = generator.generate([model])

        assert 'encoding="utf-8"' in xml_str
        assert "中文" in xml_str

    def test_xml_without_declaration(self, generator):
        """Test generating XML without declaration."""
        model = DataModel(name="Test", elements=[])

        xml_str = generator.generate([model], include_xml_declaration=False)

        assert not xml_str.startswith("<?xml")
        assert "<Secray>" in xml_str

    def test_generate_single(self, generator):
        """Test generating single DataModel."""
        model = DataModel(name="Test", elements=[])

        xml_str = generator.generate_single(model)

        root = ET.fromstring(xml_str)
        assert len(root.findall("DataModel")) == 1

    def test_generate_empty_models(self, generator):
        """Test generating with empty model list."""
        xml_str = generator.generate([])

        root = ET.fromstring(xml_str)
        assert root.tag == "Secray"
        assert len(root.findall("DataModel")) == 0

    def test_generate_with_comments(self, generator):
        """Test generating with comments."""
        model = DataModel(
            name="Test",
            description="Test description",
            elements=[],
        )

        xml_str = generator.generate([model], include_comments=True)

        assert "Generated by Fuzz Generator" in xml_str
        assert "Test description" in xml_str

    def test_generate_without_comments(self, generator):
        """Test generating without comments."""
        model = DataModel(
            name="Test",
            description="Test description",
            elements=[],
        )

        xml_str = generator.generate([model], include_comments=False)

        assert "Generated by Fuzz Generator" not in xml_str

    def test_generate_block_with_children(self, generator):
        """Test generating Block with inline children."""
        model = DataModel(
            name="Request",
            elements=[
                BlockElement(
                    name="Headers",
                    children=[
                        StringElement(name="Name"),
                        StringElement(name="Value"),
                    ],
                ),
            ],
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        block = root.find(".//Block")
        assert block is not None
        assert len(block.findall("String")) == 2

    def test_generate_block_with_occurs(self, generator):
        """Test generating Block with minOccurs/maxOccurs."""
        model = DataModel(
            name="Request",
            elements=[
                BlockElement(
                    name="Header",
                    ref="HeaderLine",
                    min_occurs=0,
                    max_occurs="unbounded",
                ),
            ],
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        block = root.find(".//Block")
        assert block is not None
        assert block.get("minOccurs") == "0"
        assert block.get("maxOccurs") == "unbounded"

    def test_generate_to_file(self, generator, tmp_path):
        """Test generating XML to file."""
        model = DataModel(name="Test", elements=[])
        file_path = tmp_path / "output.xml"

        generator.generate_to_file([model], str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert "<DataModel" in content

    def test_generate_from_dict(self, generator):
        """Test generating from dictionary."""
        data = {
            "name": "Request",
            "elements": [
                {"element_type": "String", "name": "Method"},
            ],
        }

        xml_str = generator.generate_from_dict(data)

        root = ET.fromstring(xml_str)
        dm = root.find("DataModel")
        assert dm.get("name") == "Request"

    def test_generate_from_dict_list(self, generator):
        """Test generating from list of dictionaries."""
        data = [
            {"name": "Model1", "elements": []},
            {"name": "Model2", "elements": []},
        ]

        xml_str = generator.generate_from_dict(data)

        root = ET.fromstring(xml_str)
        assert len(root.findall("DataModel")) == 2

    def test_generate_element_direct(self, generator):
        """Test generating individual element."""
        elem = StringElement(name="Test", value="value")

        xml_elem = generator.generate_element(elem)

        assert xml_elem.tag == "String"
        assert xml_elem.get("name") == "Test"
        assert xml_elem.get("value") == "value"

    def test_generate_element_unknown_type(self, generator):
        """Test generating unknown element type."""

        class UnknownElement:
            pass

        with pytest.raises(ValueError, match="Unknown element type"):
            generator.generate_element(UnknownElement())

    def test_complete_example(self, generator):
        """Test generating complete example from docs."""
        models = [
            DataModel(
                name="CrLf",
                elements=[
                    ChoiceElement(
                        name="EndChoice",
                        options=[
                            StringElement(name="End1", value="\r\n", token=True, mutable=False),
                            StringElement(name="End2", value="\r", token=True, mutable=False),
                            StringElement(name="End3", value="\n", token=True, mutable=False),
                        ],
                    ),
                ],
            ),
            DataModel(
                name="Request",
                elements=[
                    StringElement(name="Method"),
                    StringElement(name="Space", value=" ", token=True),
                    StringElement(name="Url", value="/"),
                    StringElement(name="Space2", value=" ", token=True),
                    StringElement(name="Version", value="HTTP/1.1", mutable=False),
                    BlockElement(name="End", ref="CrLf"),
                ],
            ),
        ]

        xml_str = generator.generate(models)

        # Verify structure
        root = ET.fromstring(xml_str)
        assert len(root.findall("DataModel")) == 2

        # Verify CrLf model
        crlf = root.find(".//DataModel[@name='CrLf']")
        assert crlf is not None
        choice = crlf.find("Choice")
        assert choice is not None
        assert len(choice.findall("String")) == 3

        # Verify Request model
        request = root.find(".//DataModel[@name='Request']")
        assert request is not None
        assert len(request.findall("String")) == 5
        assert request.find("Block") is not None
