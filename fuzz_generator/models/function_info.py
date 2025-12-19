"""Function information models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ParameterDirection(str, Enum):
    """Parameter direction indicator."""

    IN = "in"
    OUT = "out"
    INOUT = "inout"


class ParameterInfo(BaseModel):
    """Function parameter information."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type")
    direction: ParameterDirection = Field(
        default=ParameterDirection.IN,
        description="Parameter direction",
    )
    description: str = Field(
        default="",
        description="Parameter description",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Parameter constraints (e.g., length limits)",
    )
    format_hints: list[str] = Field(
        default_factory=list,
        description="Format hints for fuzz generation",
    )
    default_value: str | None = Field(
        default=None,
        description="Default value if any",
    )
    is_optional: bool = Field(
        default=False,
        description="Whether parameter is optional",
    )
    is_pointer: bool = Field(
        default=False,
        description="Whether parameter is a pointer",
    )
    is_array: bool = Field(
        default=False,
        description="Whether parameter is an array",
    )
    array_size: int | str | None = Field(
        default=None,
        description="Array size if applicable",
    )


class StructFieldInfo(BaseModel):
    """Structure field information."""

    name: str = Field(description="Field name")
    type: str = Field(description="Field type")
    offset: int | None = Field(
        default=None,
        description="Byte offset in struct",
    )
    size: int | None = Field(
        default=None,
        description="Field size in bytes",
    )
    description: str = Field(
        default="",
        description="Field description",
    )


class StructInfo(BaseModel):
    """Structure information."""

    name: str = Field(description="Structure name")
    fields: list[StructFieldInfo] = Field(
        default_factory=list,
        description="Structure fields",
    )
    total_size: int | None = Field(
        default=None,
        description="Total structure size in bytes",
    )
    is_packed: bool = Field(
        default=False,
        description="Whether struct is packed",
    )


class FunctionInfo(BaseModel):
    """Complete function information."""

    name: str = Field(description="Function name")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(description="Starting line number")
    return_type: str = Field(description="Return type")
    parameters: list[ParameterInfo] = Field(
        default_factory=list,
        description="Function parameters",
    )
    source_code: str = Field(
        default="",
        description="Function source code",
    )
    description: str = Field(
        default="",
        description="Function description",
    )
    signature: str = Field(
        default="",
        description="Full function signature",
    )
    is_static: bool = Field(
        default=False,
        description="Whether function is static",
    )
    is_inline: bool = Field(
        default=False,
        description="Whether function is inline",
    )
    related_structs: list[StructInfo] = Field(
        default_factory=list,
        description="Related structure definitions",
    )
    local_variables: list[ParameterInfo] = Field(
        default_factory=list,
        description="Important local variables",
    )
    complexity: int = Field(
        default=0,
        description="Cyclomatic complexity",
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional annotations",
    )

    @property
    def full_signature(self) -> str:
        """Get full function signature."""
        if self.signature:
            return self.signature
        params = ", ".join(f"{p.type} {p.name}" for p in self.parameters)
        return f"{self.return_type} {self.name}({params})"

    @property
    def input_parameters(self) -> list[ParameterInfo]:
        """Get input parameters only."""
        return [
            p
            for p in self.parameters
            if p.direction in (ParameterDirection.IN, ParameterDirection.INOUT)
        ]

    @property
    def output_parameters(self) -> list[ParameterInfo]:
        """Get output parameters only."""
        return [
            p
            for p in self.parameters
            if p.direction in (ParameterDirection.OUT, ParameterDirection.INOUT)
        ]
