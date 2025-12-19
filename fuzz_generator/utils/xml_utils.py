"""XML utilities for handling DataModel XML content."""

import re
from typing import Any

from lxml import etree

from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


def extract_xml_from_text(text: str) -> str | None:
    """Extract XML DataModel content from text.

    Handles various formats:
    - Raw XML starting with <DataModel or <Secray
    - XML in markdown code blocks (```xml ... ```)
    - XML mixed with other text

    Args:
        text: Text potentially containing XML

    Returns:
        Extracted XML string or None
    """
    if not text:
        return None

    # Try to find XML in markdown code block
    xml_block_pattern = r"```(?:xml)?\s*([\s\S]*?)```"
    matches = re.findall(xml_block_pattern, text)

    for match in matches:
        content = match.strip()
        if "<DataModel" in content or "<Secray" in content:
            return _normalize_xml(content)

    # Try to find raw XML
    # Look for <Secray or <DataModel tags
    secray_match = re.search(r"(<Secray[\s\S]*?</Secray>)", text)
    if secray_match:
        return _normalize_xml(secray_match.group(1))

    datamodel_match = re.search(r"(<DataModel[\s\S]*?</DataModel>)", text)
    if datamodel_match:
        # Wrap in Secray root
        content = datamodel_match.group(1)
        return f'<?xml version="1.0" encoding="utf-8"?>\n<Secray>\n{content}\n</Secray>'

    return None


def _normalize_xml(xml_str: str) -> str:
    """Normalize XML string.

    Args:
        xml_str: Raw XML string

    Returns:
        Normalized XML with proper declaration
    """
    xml_str = xml_str.strip()

    # Add XML declaration if missing
    if not xml_str.startswith("<?xml"):
        xml_str = f'<?xml version="1.0" encoding="utf-8"?>\n{xml_str}'

    # Wrap in Secray if needed
    if not xml_str.startswith("<?xml") and "<Secray" not in xml_str:
        # Find where to insert
        if xml_str.startswith("<DataModel"):
            xml_str = f'<?xml version="1.0" encoding="utf-8"?>\n<Secray>\n{xml_str}\n</Secray>'

    return xml_str


def format_xml(xml_str: str, indent: int = 4) -> str:
    """Format XML string with proper indentation.

    Args:
        xml_str: XML string to format
        indent: Number of spaces for indentation

    Returns:
        Formatted XML string
    """
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_str.encode("utf-8"), parser)
        return etree.tostring(
            root,
            pretty_print=True,
            encoding="utf-8",
            xml_declaration=True,
        ).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to format XML: {e}")
        return xml_str


def merge_datamodels(xml_contents: list[str]) -> str:
    """Merge multiple DataModel XML contents into one.

    Args:
        xml_contents: List of XML strings

    Returns:
        Merged XML string with all DataModels
    """
    all_datamodels = []

    for xml_str in xml_contents:
        if not xml_str:
            continue

        try:
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.fromstring(xml_str.encode("utf-8"), parser)

            # Find DataModel elements
            if root.tag == "Secray":
                datamodels = root.findall("DataModel")
            elif root.tag == "DataModel":
                datamodels = [root]
            else:
                datamodels = root.findall(".//DataModel")

            all_datamodels.extend(datamodels)

        except Exception as e:
            logger.warning(f"Failed to parse XML for merging: {e}")

    if not all_datamodels:
        return ""

    # Create merged document
    merged_root = etree.Element("Secray")
    for dm in all_datamodels:
        # Deep copy to avoid issues with moving elements
        merged_root.append(etree.fromstring(etree.tostring(dm)))

    return etree.tostring(
        merged_root,
        pretty_print=True,
        encoding="utf-8",
        xml_declaration=True,
    ).decode("utf-8")


def count_datamodels(xml_str: str) -> int:
    """Count the number of DataModel elements in XML.

    Args:
        xml_str: XML string

    Returns:
        Number of DataModel elements
    """
    try:
        root = etree.fromstring(xml_str.encode("utf-8"))
        if root.tag == "DataModel":
            return 1
        return len(root.findall(".//DataModel"))
    except Exception:
        return 0


def get_datamodel_names(xml_str: str) -> list[str]:
    """Get names of all DataModel elements.

    Args:
        xml_str: XML string

    Returns:
        List of DataModel names
    """
    try:
        root = etree.fromstring(xml_str.encode("utf-8"))
        names = []

        if root.tag == "DataModel":
            name = root.get("name")
            if name:
                names.append(name)
        else:
            for dm in root.findall(".//DataModel"):
                name = dm.get("name")
                if name:
                    names.append(name)

        return names
    except Exception:
        return []


def xml_to_dict(xml_str: str) -> dict[str, Any]:
    """Convert XML to dictionary representation.

    Args:
        xml_str: XML string

    Returns:
        Dictionary representation of XML
    """
    try:
        root = etree.fromstring(xml_str.encode("utf-8"))
        return _element_to_dict(root)
    except Exception as e:
        logger.warning(f"Failed to convert XML to dict: {e}")
        return {}


def _element_to_dict(element: etree._Element) -> dict[str, Any]:
    """Convert an XML element to dictionary.

    Args:
        element: lxml Element

    Returns:
        Dictionary representation
    """
    result: dict[str, Any] = {
        "tag": element.tag,
        "attributes": dict(element.attrib),
    }

    children = list(element)
    if children:
        result["children"] = [_element_to_dict(child) for child in children]

    if element.text and element.text.strip():
        result["text"] = element.text.strip()

    return result
