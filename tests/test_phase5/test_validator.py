"""Tests for XML validator."""

import pytest

from fuzz_generator.generators.validator import (
    ValidationResult,
    XMLValidator,
    validate_xml,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_default_valid(self):
        """Test default result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        """Test adding error."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding warning."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True  # Warnings don't invalidate
        assert "Test warning" in result.warnings

    def test_merge_results(self):
        """Test merging results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 1")
        result2.info["key"] = "value"

        result1.merge(result2)

        assert result1.is_valid is False
        assert "Error 1" in result1.errors
        assert "Warning 1" in result1.warnings
        assert result1.info["key"] == "value"


class TestXMLValidator:
    """Tests for XMLValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return XMLValidator()

    def test_validator_creation(self):
        """Test validator can be created."""
        validator = XMLValidator()
        assert validator.strict is False

    def test_validator_strict_mode(self):
        """Test validator in strict mode."""
        validator = XMLValidator(strict=True)
        assert validator.strict is True

    def test_validate_valid_xml(self, validator):
        """Test validating valid XML."""
        xml_str = """<?xml version="1.0" encoding="utf-8"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" value="value" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_syntax(self, validator):
        """Test validating invalid XML syntax."""
        xml_str = "<Secray><DataModel></Secray>"

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("syntax" in err.lower() for err in result.errors)

    def test_validate_wrong_root(self, validator):
        """Test validating wrong root element."""
        xml_str = """<?xml version="1.0"?>
        <WrongRoot>
            <DataModel name="Test" />
        </WrongRoot>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("Secray" in err for err in result.errors)

    def test_validate_missing_name(self, validator):
        """Test validating missing name attribute."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel>
                <String value="value" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("name" in err.lower() for err in result.errors)

    def test_validate_missing_datamodel_name(self, validator):
        """Test validating DataModel without name."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel>
                <String name="Field" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("name" in err.lower() for err in result.errors)

    def test_validate_duplicate_datamodel_name(self, validator):
        """Test validating duplicate DataModel names."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field1" />
            </DataModel>
            <DataModel name="Test">
                <String name="Field2" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("duplicate" in err.lower() for err in result.errors)

    def test_validate_invalid_ref(self, validator):
        """Test validating invalid Block ref."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Invalid" ref="NonExistent" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("ref" in err.lower() or "reference" in err.lower() for err in result.errors)

    def test_validate_valid_ref(self, validator):
        """Test validating valid Block ref."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="CrLf">
                <String name="End" value="\\r\\n" />
            </DataModel>
            <DataModel name="Request">
                <Block name="End" ref="CrLf" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True

    def test_validate_invalid_element_type(self, validator):
        """Test validating invalid element type."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <InvalidElement name="Bad" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("invalid" in err.lower() for err in result.errors)

    def test_validate_empty_document(self, validator):
        """Test validating empty document."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
        </Secray>
        """

        result = validator.validate(xml_str)

        # Empty is valid but should have warning
        assert result.is_valid is True
        assert any("no datamodel" in w.lower() for w in result.warnings)

    def test_validate_choice_with_one_option(self, validator):
        """Test validating Choice with only one option."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Choice name="OnlyOne">
                    <String name="Option1" />
                </Choice>
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert any("fewer than 2" in w.lower() for w in result.warnings)

    def test_validate_number_invalid_size(self, validator):
        """Test validating Number with unusual size."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Number name="Field" size="12" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert any("unusual size" in w.lower() for w in result.warnings)

    def test_validate_number_invalid_endian(self, validator):
        """Test validating Number with invalid endian."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Number name="Field" endian="middle" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("endian" in err.lower() for err in result.errors)

    def test_validate_block_both_ref_and_children(self, validator):
        """Test validating Block with both ref and children."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="CrLf">
                <String name="End" />
            </DataModel>
            <DataModel name="Test">
                <Block name="Mixed" ref="CrLf">
                    <String name="Extra" />
                </Block>
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert any("both" in w.lower() for w in result.warnings)

    def test_validate_block_neither_ref_nor_children(self, validator):
        """Test validating Block with neither ref nor children."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Empty" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert any("neither" in w.lower() for w in result.warnings)

    def test_validate_block_invalid_min_occurs(self, validator):
        """Test validating Block with invalid minOccurs."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Bad" minOccurs="-1" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("minoccurs" in err.lower() for err in result.errors)

    def test_validate_block_invalid_max_occurs(self, validator):
        """Test validating Block with invalid maxOccurs."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Bad" maxOccurs="abc" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("maxoccurs" in err.lower() for err in result.errors)

    def test_validate_block_max_less_than_min(self, validator):
        """Test validating Block with maxOccurs < minOccurs."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Bad" minOccurs="5" maxOccurs="3" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is False
        assert any("less than" in err.lower() for err in result.errors)

    def test_validate_block_unbounded_max(self, validator):
        """Test validating Block with unbounded maxOccurs."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Good" minOccurs="0" maxOccurs="unbounded" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        # Has warning for neither ref nor children, but unbounded is valid
        assert (
            any("maxoccurs" not in err.lower() for err in result.errors) or len(result.errors) == 0
        )

    def test_validate_unknown_attribute(self, validator):
        """Test validating element with unknown attribute."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" unknownAttr="value" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert any("unknown" in w.lower() for w in result.warnings)

    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        validator = XMLValidator(strict=True)

        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" unknownAttr="value" />
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        # In strict mode, warnings become errors
        assert result.is_valid is False
        assert any("strict" in err.lower() for err in result.errors)

    def test_validate_file(self, validator, tmp_path):
        """Test validating XML file."""
        file_path = tmp_path / "test.xml"
        file_path.write_text("""<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" />
            </DataModel>
        </Secray>
        """)

        result = validator.validate_file(str(file_path))

        assert result.is_valid is True

    def test_validate_nonexistent_file(self, validator):
        """Test validating non-existent file."""
        result = validator.validate_file("/nonexistent/path.xml")

        assert result.is_valid is False
        assert any("read" in err.lower() or "file" in err.lower() for err in result.errors)

    def test_is_valid_xml(self, validator):
        """Test is_valid_xml convenience method."""
        valid_xml = """<?xml version="1.0"?><Secray></Secray>"""
        invalid_xml = "<Secray><DataModel></Secray>"

        assert validator.is_valid_xml(valid_xml) is True
        assert validator.is_valid_xml(invalid_xml) is False

    def test_validate_xml_convenience_function(self):
        """Test validate_xml convenience function."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" />
            </DataModel>
        </Secray>
        """

        result = validate_xml(xml_str)
        assert result.is_valid is True

        result_strict = validate_xml(xml_str, strict=True)
        assert result_strict.is_valid is True

    def test_validate_info_datamodel_count(self, validator):
        """Test that validation result includes DataModel count."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test1" />
            <DataModel name="Test2" />
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.info["datamodel_count"] == 2

    def test_validate_info_datamodel_names(self, validator):
        """Test that validation result includes DataModel names."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Request" />
            <DataModel name="Response" />
        </Secray>
        """

        result = validator.validate(xml_str)

        assert set(result.info["datamodel_names"]) == {"Request", "Response"}

    def test_validate_nested_block_children(self, validator):
        """Test validating nested Block children."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Outer">
                    <Block name="Inner">
                        <String name="Field" />
                    </Block>
                </Block>
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        # Should have warnings for empty blocks but be valid
        assert result.is_valid is True

    def test_validate_choice_nested_elements(self, validator):
        """Test validating Choice with nested elements."""
        xml_str = """<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Choice name="Options">
                    <String name="Option1" />
                    <Block name="Option2">
                        <String name="Inner" />
                    </Block>
                </Choice>
            </DataModel>
        </Secray>
        """

        result = validator.validate(xml_str)

        assert result.is_valid is True
