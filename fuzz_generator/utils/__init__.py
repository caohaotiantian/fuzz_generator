"""Utility modules for fuzz_generator."""

from fuzz_generator.utils.logger import get_logger, setup_logger
from fuzz_generator.utils.validators import (
    sanitize_datamodel_name,
    validate_batch_task,
    validate_function_name,
    validate_project_structure,
    validate_source_file,
    validate_xml_output_path,
)
from fuzz_generator.utils.xml_utils import (
    count_datamodels,
    extract_xml_from_text,
    format_xml,
    get_datamodel_names,
    merge_datamodels,
    xml_to_dict,
)

__all__ = [
    # Logger
    "setup_logger",
    "get_logger",
    # Validators
    "validate_function_name",
    "validate_source_file",
    "validate_project_structure",
    "validate_xml_output_path",
    "validate_batch_task",
    "sanitize_datamodel_name",
    # XML utils
    "extract_xml_from_text",
    "format_xml",
    "merge_datamodels",
    "count_datamodels",
    "get_datamodel_names",
    "xml_to_dict",
]
