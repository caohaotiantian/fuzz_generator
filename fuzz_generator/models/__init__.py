"""Data models module."""

from fuzz_generator.models.analysis_result import (
    AnalysisContext,
    CallGraphInfo,
    ControlFlowInfo,
    DataFlowPath,
    GenerationResult,
)
from fuzz_generator.models.function_info import (
    FunctionInfo,
    ParameterDirection,
    ParameterInfo,
)
from fuzz_generator.models.task import (
    AnalysisTask,
    BatchTask,
    IntermediateResult,
    TaskResult,
    TaskStatus,
)
from fuzz_generator.models.xml_models import (
    BlobElement,
    BlockElement,
    ChoiceElement,
    DataModel,
    NumberElement,
    StringElement,
)

__all__ = [
    # Task models
    "TaskStatus",
    "AnalysisTask",
    "BatchTask",
    "TaskResult",
    "IntermediateResult",
    # Function models
    "ParameterDirection",
    "ParameterInfo",
    "FunctionInfo",
    # Analysis models
    "DataFlowPath",
    "ControlFlowInfo",
    "CallGraphInfo",
    "AnalysisContext",
    "GenerationResult",
    # XML models
    "StringElement",
    "NumberElement",
    "BlobElement",
    "ChoiceElement",
    "BlockElement",
    "DataModel",
]
