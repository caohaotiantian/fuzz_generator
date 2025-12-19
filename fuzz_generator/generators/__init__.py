"""XML generation module for Secray DataModel output.

This module provides functionality for generating and validating
Secray format XML DataModel files.
"""

from fuzz_generator.generators.validator import ValidationResult, XMLValidator
from fuzz_generator.generators.xml_generator import XMLGenerator

__all__ = [
    # Generator
    "XMLGenerator",
    # Validator
    "XMLValidator",
    "ValidationResult",
]
