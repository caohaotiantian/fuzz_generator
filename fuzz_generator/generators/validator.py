"""XML Validator for Secray DataModel format.

Validates XML syntax and DataModel structure.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from fuzz_generator.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of XML validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)


# Valid element types in DataModel
VALID_ELEMENT_TYPES = {"String", "Number", "Blob", "Block", "Choice"}

# Required attributes for each element type
REQUIRED_ATTRS = {
    "DataModel": ["name"],
    "String": ["name"],
    "Number": ["name"],
    "Blob": ["name"],
    "Block": ["name"],
    "Choice": ["name"],
}

# Optional attributes for each element type
OPTIONAL_ATTRS = {
    "String": ["value", "token", "mutable", "length", "minLength", "maxLength"],
    "Number": ["size", "signed", "endian", "value", "mutable", "minValue", "maxValue"],
    "Blob": ["length", "minLength", "maxLength", "mutable", "valueType"],
    "Block": ["ref", "minOccurs", "maxOccurs"],
    "Choice": [],
}


class XMLValidator:
    """Validator for Secray XML DataModel format."""

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, treat warnings as errors.
        """
        self.strict = strict

    def validate(self, xml_content: str) -> ValidationResult:
        """Validate XML content.

        Args:
            xml_content: XML string to validate.

        Returns:
            ValidationResult with validation status and messages.
        """
        result = ValidationResult()

        # Parse XML
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            result.add_error(f"XML syntax error: {e}")
            return result

        # Validate structure
        self._validate_root(root, result)

        if result.is_valid:
            # Validate DataModels
            self._validate_datamodels(root, result)

            # Validate references
            self._validate_references(root, result)

        # Convert warnings to errors in strict mode
        if self.strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"(strict) {warning}")
            result.warnings.clear()

        return result

    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate an XML file.

        Args:
            file_path: Path to XML file.

        Returns:
            ValidationResult with validation status and messages.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return self.validate(content)
        except OSError as e:
            result = ValidationResult()
            result.add_error(f"Failed to read file: {e}")
            return result

    def _validate_root(self, root: ET.Element, result: ValidationResult) -> None:
        """Validate root element."""
        if root.tag != "Secray":
            result.add_error(f"Root element must be 'Secray', found '{root.tag}'")
            return

        # Check for DataModel children
        datamodels = root.findall("DataModel")
        if not datamodels:
            result.add_warning("No DataModel elements found in document")

        result.info["datamodel_count"] = len(datamodels)

    def _validate_datamodels(self, root: ET.Element, result: ValidationResult) -> None:
        """Validate all DataModel elements."""
        datamodels = root.findall("DataModel")
        names = set()

        for dm in datamodels:
            # Check required attributes
            name = dm.get("name")
            if not name:
                result.add_error("DataModel missing required 'name' attribute")
                continue

            # Check for duplicate names
            if name in names:
                result.add_error(f"Duplicate DataModel name: '{name}'")
            names.add(name)

            # Validate children
            self._validate_datamodel_children(dm, name, result)

        result.info["datamodel_names"] = list(names)

    def _validate_datamodel_children(
        self,
        datamodel: ET.Element,
        datamodel_name: str,
        result: ValidationResult,
    ) -> None:
        """Validate children of a DataModel element."""
        for child in datamodel:
            # Skip comments
            if not isinstance(child.tag, str):
                continue

            if child.tag not in VALID_ELEMENT_TYPES:
                result.add_error(
                    f"Invalid element type '{child.tag}' in DataModel '{datamodel_name}'"
                )
                continue

            # Validate element
            self._validate_element(child, datamodel_name, result)

    def _validate_element(
        self,
        element: ET.Element,
        context: str,
        result: ValidationResult,
    ) -> None:
        """Validate a single element."""
        tag = element.tag

        # Check required attributes
        required = REQUIRED_ATTRS.get(tag, [])
        for attr in required:
            if element.get(attr) is None:
                result.add_error(f"{tag} element missing required '{attr}' attribute in {context}")

        # Validate specific element types
        if tag == "Block":
            self._validate_block_element(element, context, result)
        elif tag == "Choice":
            self._validate_choice_element(element, context, result)
        elif tag == "Number":
            self._validate_number_element(element, context, result)

        # Check for unknown attributes
        name = element.get("name", "unnamed")
        known_attrs = set(REQUIRED_ATTRS.get(tag, [])) | set(OPTIONAL_ATTRS.get(tag, []))
        for attr in element.attrib:
            if attr not in known_attrs:
                result.add_warning(
                    f"Unknown attribute '{attr}' on {tag} element '{name}' in {context}"
                )

    def _validate_block_element(
        self,
        element: ET.Element,
        context: str,
        result: ValidationResult,
    ) -> None:
        """Validate Block element specifics."""
        name = element.get("name", "unnamed")
        ref = element.get("ref")
        children = list(element)

        # Block should have either ref or children, not both
        if ref and children:
            result.add_warning(f"Block '{name}' in {context} has both 'ref' and inline children")
        elif not ref and not children:
            result.add_warning(f"Block '{name}' in {context} has neither 'ref' nor inline children")

        # Validate minOccurs/maxOccurs
        min_occurs = element.get("minOccurs")
        max_occurs = element.get("maxOccurs")

        if min_occurs is not None:
            try:
                min_val = int(min_occurs)
                if min_val < 0:
                    result.add_error(f"Block '{name}' in {context}: minOccurs must be non-negative")
            except ValueError:
                result.add_error(f"Block '{name}' in {context}: minOccurs must be an integer")

        if max_occurs is not None and max_occurs != "unbounded":
            try:
                max_val = int(max_occurs)
                if max_val < 0:
                    result.add_error(f"Block '{name}' in {context}: maxOccurs must be non-negative")
                if min_occurs is not None:
                    min_val = int(min_occurs)
                    if max_val < min_val:
                        result.add_error(
                            f"Block '{name}' in {context}: maxOccurs cannot be less than minOccurs"
                        )
            except ValueError:
                result.add_error(
                    f"Block '{name}' in {context}: maxOccurs must be an integer or 'unbounded'"
                )

        # Validate children recursively
        for child in children:
            if isinstance(child.tag, str) and child.tag in VALID_ELEMENT_TYPES:
                self._validate_element(child, f"{context}.{name}", result)

    def _validate_choice_element(
        self,
        element: ET.Element,
        context: str,
        result: ValidationResult,
    ) -> None:
        """Validate Choice element specifics."""
        name = element.get("name", "unnamed")
        options = [c for c in element if isinstance(c.tag, str) and c.tag in VALID_ELEMENT_TYPES]

        if len(options) < 2:
            result.add_warning(f"Choice '{name}' in {context} has fewer than 2 options")

        # Validate options recursively
        for option in options:
            self._validate_element(option, f"{context}.{name}", result)

    def _validate_number_element(
        self,
        element: ET.Element,
        context: str,
        result: ValidationResult,
    ) -> None:
        """Validate Number element specifics."""
        name = element.get("name", "unnamed")

        # Validate size
        size = element.get("size")
        if size is not None:
            try:
                size_val = int(size)
                valid_sizes = {8, 16, 32, 64}
                if size_val not in valid_sizes:
                    result.add_warning(
                        f"Number '{name}' in {context}: unusual size {size_val} "
                        f"(expected one of {valid_sizes})"
                    )
            except ValueError:
                result.add_error(f"Number '{name}' in {context}: size must be an integer")

        # Validate endian
        endian = element.get("endian")
        if endian is not None and endian not in {"big", "little"}:
            result.add_error(f"Number '{name}' in {context}: endian must be 'big' or 'little'")

    def _validate_references(self, root: ET.Element, result: ValidationResult) -> None:
        """Validate Block ref attributes point to existing DataModels."""
        # Collect all DataModel names
        datamodel_names = {dm.get("name") for dm in root.findall("DataModel") if dm.get("name")}

        # Find all Block elements with ref
        for block in root.iter("Block"):
            ref = block.get("ref")
            if ref and ref not in datamodel_names:
                name = block.get("name", "unnamed")
                result.add_error(f"Block '{name}' references non-existent DataModel '{ref}'")

    def is_valid_xml(self, xml_content: str) -> bool:
        """Quick check if XML is valid.

        Args:
            xml_content: XML string to check.

        Returns:
            True if XML is valid.
        """
        return self.validate(xml_content).is_valid


def validate_xml(xml_content: str, strict: bool = False) -> ValidationResult:
    """Convenience function to validate XML.

    Args:
        xml_content: XML string to validate.
        strict: If True, treat warnings as errors.

    Returns:
        ValidationResult with validation status and messages.
    """
    validator = XMLValidator(strict=strict)
    return validator.validate(xml_content)
