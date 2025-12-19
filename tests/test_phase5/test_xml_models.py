"""Tests for XML model structures."""

from fuzz_generator.models.xml_models import (
    BlobElement,
    BlockElement,
    ChoiceElement,
    DataModel,
    NumberElement,
    StringElement,
)


class TestStringElement:
    """Tests for StringElement."""

    def test_basic_creation(self):
        """Test basic element creation."""
        elem = StringElement(name="Method")
        assert elem.name == "Method"
        assert elem.value is None
        assert elem.token is False
        assert elem.mutable is True

    def test_with_value(self):
        """Test element with value."""
        elem = StringElement(
            name="Method",
            value="GET",
            token=True,
            mutable=False,
        )
        assert elem.name == "Method"
        assert elem.value == "GET"
        assert elem.token is True
        assert elem.mutable is False

    def test_with_length_constraints(self):
        """Test element with length constraints."""
        elem = StringElement(
            name="Buffer",
            min_length=1,
            max_length=256,
        )
        assert elem.min_length == 1
        assert elem.max_length == 256

    def test_to_xml_attrs(self):
        """Test conversion to XML attributes."""
        elem = StringElement(
            name="Method",
            value="GET",
            token=True,
        )
        attrs = elem.to_xml_attrs()
        assert attrs["name"] == "Method"
        assert attrs["value"] == "GET"
        assert attrs["token"] == "true"
        assert "mutable" not in attrs  # Default value, not included

    def test_to_xml_attrs_mutable_false(self):
        """Test XML attributes with mutable=False."""
        elem = StringElement(name="Fixed", mutable=False)
        attrs = elem.to_xml_attrs()
        assert attrs["mutable"] == "false"


class TestNumberElement:
    """Tests for NumberElement."""

    def test_basic_creation(self):
        """Test basic element creation."""
        elem = NumberElement(name="Length")
        assert elem.name == "Length"
        assert elem.size == 32
        assert elem.signed is True
        assert elem.endian == "big"

    def test_with_all_options(self):
        """Test element with all options."""
        elem = NumberElement(
            name="Port",
            size=16,
            signed=False,
            endian="little",
            value=80,
            min_value=0,
            max_value=65535,
        )
        assert elem.size == 16
        assert elem.signed is False
        assert elem.endian == "little"
        assert elem.value == 80

    def test_to_xml_attrs(self):
        """Test conversion to XML attributes."""
        elem = NumberElement(
            name="Port",
            size=16,
            endian="little",
            value=80,
        )
        attrs = elem.to_xml_attrs()
        assert attrs["name"] == "Port"
        assert attrs["size"] == "16"
        assert attrs["endian"] == "little"
        assert attrs["value"] == "80"


class TestBlobElement:
    """Tests for BlobElement."""

    def test_basic_creation(self):
        """Test basic element creation."""
        elem = BlobElement(name="Data")
        assert elem.name == "Data"
        assert elem.length is None
        assert elem.mutable is True

    def test_with_length(self):
        """Test element with fixed length."""
        elem = BlobElement(name="Data", length=1024)
        assert elem.length == 1024

    def test_with_reference_length(self):
        """Test element with length reference."""
        elem = BlobElement(name="Data", length="ContentLength")
        assert elem.length == "ContentLength"

    def test_to_xml_attrs(self):
        """Test conversion to XML attributes."""
        elem = BlobElement(name="Data", length=256)
        attrs = elem.to_xml_attrs()
        assert attrs["name"] == "Data"
        assert attrs["length"] == "256"


class TestBlockElement:
    """Tests for BlockElement."""

    def test_basic_creation(self):
        """Test basic element creation."""
        elem = BlockElement(name="Header")
        assert elem.name == "Header"
        assert elem.ref is None
        assert elem.min_occurs == 1
        assert elem.max_occurs == 1

    def test_with_ref(self):
        """Test element with reference."""
        elem = BlockElement(
            name="Header",
            ref="HeaderLine",
            min_occurs=0,
            max_occurs="unbounded",
        )
        assert elem.ref == "HeaderLine"
        assert elem.min_occurs == 0
        assert elem.max_occurs == "unbounded"

    def test_with_children(self):
        """Test element with inline children."""
        elem = BlockElement(
            name="Request",
            children=[
                StringElement(name="Method"),
                StringElement(name="Space", value=" "),
            ],
        )
        assert len(elem.children) == 2
        assert elem.children[0].name == "Method"

    def test_to_xml_attrs(self):
        """Test conversion to XML attributes."""
        elem = BlockElement(
            name="Header",
            ref="HeaderLine",
            min_occurs=0,
        )
        attrs = elem.to_xml_attrs()
        assert attrs["name"] == "Header"
        assert attrs["ref"] == "HeaderLine"
        assert attrs["minOccurs"] == "0"
        assert "maxOccurs" not in attrs  # Default value


class TestChoiceElement:
    """Tests for ChoiceElement."""

    def test_basic_creation(self):
        """Test basic element creation."""
        elem = ChoiceElement(name="EndChoice")
        assert elem.name == "EndChoice"
        assert elem.options == []

    def test_with_options(self):
        """Test element with options."""
        elem = ChoiceElement(
            name="EndChoice",
            options=[
                StringElement(name="CRLF", value="\r\n", token=True),
                StringElement(name="LF", value="\n", token=True),
            ],
        )
        assert len(elem.options) == 2
        assert elem.options[0].name == "CRLF"
        assert elem.options[1].name == "LF"

    def test_to_xml_attrs(self):
        """Test conversion to XML attributes."""
        elem = ChoiceElement(name="EndChoice")
        attrs = elem.to_xml_attrs()
        assert attrs == {"name": "EndChoice"}


class TestDataModel:
    """Tests for DataModel."""

    def test_basic_creation(self):
        """Test basic model creation."""
        model = DataModel(name="Request")
        assert model.name == "Request"
        assert model.elements == []
        assert model.description is None

    def test_with_elements(self):
        """Test model with elements."""
        model = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method"),
                StringElement(name="Space", value=" ", token=True),
            ],
        )
        assert len(model.elements) == 2

    def test_with_description(self):
        """Test model with description."""
        model = DataModel(
            name="Request",
            description="HTTP Request model",
        )
        assert model.description == "HTTP Request model"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        model = DataModel(
            name="Request",
            elements=[StringElement(name="Method")],
        )
        d = model.to_dict()
        assert d["name"] == "Request"
        assert len(d["elements"]) == 1

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "Request",
            "elements": [
                {"element_type": "String", "name": "Method"},
            ],
        }
        model = DataModel.from_dict(data)
        assert model.name == "Request"
        assert len(model.elements) == 1

    def test_nested_structure(self):
        """Test nested structure."""
        # CrLf model would be defined separately
        outer = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method"),
                BlockElement(name="End", ref="CrLf"),
            ],
        )
        assert outer.elements[1].ref == "CrLf"

    def test_complex_nested_model(self):
        """Test complex nested model with Choice in Block."""
        model = DataModel(
            name="Message",
            elements=[
                BlockElement(
                    name="Headers",
                    children=[
                        ChoiceElement(
                            name="LineEnd",
                            options=[
                                StringElement(name="CRLF", value="\r\n"),
                                StringElement(name="LF", value="\n"),
                            ],
                        ),
                    ],
                ),
            ],
        )
        assert model.elements[0].name == "Headers"
        assert len(model.elements[0].children) == 1
        assert model.elements[0].children[0].name == "LineEnd"
