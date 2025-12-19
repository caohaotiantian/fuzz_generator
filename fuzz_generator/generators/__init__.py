"""XML generation and validation module."""

from fuzz_generator.generators.validator import ValidationResult, XMLValidator, validate_xml
from fuzz_generator.generators.xml_generator import XMLGenerator

__all__ = [
    "XMLGenerator",
    "XMLValidator",
    "ValidationResult",
    "validate_xml",
]
