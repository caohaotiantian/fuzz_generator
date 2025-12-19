"""XML DataModel structure definitions."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class StringElement(BaseModel):
    """String element in DataModel."""

    element_type: Literal["String"] = "String"
    name: str = Field(description="Element name")
    value: str | None = Field(
        default=None,
        description="Default value",
    )
    token: bool = Field(
        default=False,
        description="Whether this is a fixed token",
    )
    mutable: bool = Field(
        default=True,
        description="Whether value can be mutated",
    )
    length: int | None = Field(
        default=None,
        description="Fixed length if applicable",
    )
    min_length: int | None = Field(
        default=None,
        description="Minimum length",
    )
    max_length: int | None = Field(
        default=None,
        description="Maximum length",
    )
    null_terminated: bool = Field(
        default=True,
        description="Whether string is null-terminated",
    )

    def to_xml_attrs(self) -> dict[str, str]:
        """Convert to XML attributes."""
        attrs = {"name": self.name}
        if self.value is not None:
            attrs["value"] = self.value
        if self.token:
            attrs["token"] = "true"
        if not self.mutable:
            attrs["mutable"] = "false"
        if self.length is not None:
            attrs["length"] = str(self.length)
        return attrs


class NumberElement(BaseModel):
    """Number element in DataModel."""

    element_type: Literal["Number"] = "Number"
    name: str = Field(description="Element name")
    size: int = Field(
        default=32,
        description="Bit size (8, 16, 32, 64)",
    )
    signed: bool = Field(
        default=True,
        description="Whether number is signed",
    )
    endian: Literal["big", "little"] = Field(
        default="big",
        description="Byte order",
    )
    value: int | None = Field(
        default=None,
        description="Default value",
    )
    min_value: int | None = Field(
        default=None,
        description="Minimum value",
    )
    max_value: int | None = Field(
        default=None,
        description="Maximum value",
    )
    mutable: bool = Field(
        default=True,
        description="Whether value can be mutated",
    )

    def to_xml_attrs(self) -> dict[str, str]:
        """Convert to XML attributes."""
        attrs = {
            "name": self.name,
            "size": str(self.size),
        }
        if self.signed:
            attrs["signed"] = "true"
        if self.endian != "big":
            attrs["endian"] = self.endian
        if self.value is not None:
            attrs["value"] = str(self.value)
        if not self.mutable:
            attrs["mutable"] = "false"
        return attrs


class BlobElement(BaseModel):
    """Blob (binary data) element in DataModel."""

    element_type: Literal["Blob"] = "Blob"
    name: str = Field(description="Element name")
    length: int | str | None = Field(
        default=None,
        description="Fixed length or reference to length field",
    )
    min_length: int | None = Field(
        default=None,
        description="Minimum length",
    )
    max_length: int | None = Field(
        default=None,
        description="Maximum length",
    )
    value_type: str | None = Field(
        default=None,
        description="Value type hint",
    )
    mutable: bool = Field(
        default=True,
        description="Whether data can be mutated",
    )

    def to_xml_attrs(self) -> dict[str, str]:
        """Convert to XML attributes."""
        attrs = {"name": self.name}
        if self.length is not None:
            attrs["length"] = str(self.length)
        if not self.mutable:
            attrs["mutable"] = "false"
        return attrs


class BlockElement(BaseModel):
    """Block element for grouping or reference in DataModel."""

    element_type: Literal["Block"] = "Block"
    name: str = Field(description="Element name")
    ref: str | None = Field(
        default=None,
        description="Reference to another DataModel",
    )
    min_occurs: int = Field(
        default=1,
        description="Minimum occurrences",
    )
    max_occurs: int | Literal["unbounded"] = Field(
        default=1,
        description="Maximum occurrences",
    )
    children: list["DataModelElement"] = Field(
        default_factory=list,
        description="Child elements (for inline definition)",
    )

    def to_xml_attrs(self) -> dict[str, str]:
        """Convert to XML attributes."""
        attrs = {"name": self.name}
        if self.ref:
            attrs["ref"] = self.ref
        if self.min_occurs != 1:
            attrs["minOccurs"] = str(self.min_occurs)
        if self.max_occurs != 1:
            attrs["maxOccurs"] = str(self.max_occurs)
        return attrs


class ChoiceElement(BaseModel):
    """Choice element for alternatives in DataModel."""

    element_type: Literal["Choice"] = "Choice"
    name: str = Field(description="Element name")
    options: list["DataModelElement"] = Field(
        default_factory=list,
        description="Choice options",
    )

    def to_xml_attrs(self) -> dict[str, str]:
        """Convert to XML attributes."""
        return {"name": self.name}


# Union type for all element types
DataModelElement = StringElement | NumberElement | BlobElement | BlockElement | ChoiceElement

# Update forward references
BlockElement.model_rebuild()
ChoiceElement.model_rebuild()


class DataModel(BaseModel):
    """Complete DataModel definition."""

    name: str = Field(description="DataModel name")
    elements: list[DataModelElement] = Field(
        default_factory=list,
        description="DataModel elements",
    )
    description: str | None = Field(
        default=None,
        description="DataModel description",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "elements": [e.model_dump() for e in self.elements],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataModel":
        """Create DataModel from dictionary."""
        return cls.model_validate(data)
