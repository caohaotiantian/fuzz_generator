"""Orchestrator Agent for coordinating the analysis workflow.

This agent is responsible for:
- Coordinating the multi-agent workflow
- Managing intermediate results
- Handling errors and retries
- Saving state for resume capability
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import (
    AnalysisContext,
    AnalysisTask,
    FunctionInfo,
    GenerationResult,
    IntermediateResult,
    TaskResult,
    TaskStatus,
)
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.agents.code_analyzer import CodeAnalyzerAgent
    from fuzz_generator.agents.context_builder import ContextBuilderAgent
    from fuzz_generator.agents.model_generator import ModelGeneratorAgent
    from fuzz_generator.config import Settings
    from fuzz_generator.storage import StorageBackend

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """Agent for orchestrating the analysis workflow.

    This agent coordinates the execution of CodeAnalyzer,
    ContextBuilder, and ModelGenerator agents.
    """

    def __init__(
        self,
        code_analyzer: "CodeAnalyzerAgent",
        context_builder: "ContextBuilderAgent",
        model_generator: "ModelGeneratorAgent",
        storage: "StorageBackend | None" = None,
        settings: "Settings | None" = None,
        config: AgentConfig | None = None,
    ):
        """Initialize Orchestrator agent.

        Args:
            code_analyzer: CodeAnalyzer agent instance
            context_builder: ContextBuilder agent instance
            model_generator: ModelGenerator agent instance
            storage: Storage backend for saving intermediate results
            settings: Application settings
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="Orchestrator",
                system_prompt="",
                max_iterations=50,
            )

        super().__init__(config, settings)

        self.code_analyzer = code_analyzer
        self.context_builder = context_builder
        self.model_generator = model_generator
        self.storage = storage

        self._current_task: AnalysisTask | None = None
        self._intermediate_results: dict[str, Any] = {}

    async def run(
        self,
        project_name: str,
        source_file: str,
        function_name: str,
        output_name: str | None = None,
        custom_knowledge: str = "",
        task_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the complete analysis workflow.

        Args:
            project_name: Name of the project in Joern
            source_file: Source file path
            function_name: Function name to analyze
            output_name: Optional name for generated DataModel
            custom_knowledge: Custom knowledge to inject
            task_id: Optional task ID for tracking
            **kwargs: Additional arguments

        Returns:
            AgentResult with TaskResult in data field
        """
        # Create or use task
        if task_id is None:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{function_name}"

        self._current_task = AnalysisTask(
            task_id=task_id,
            source_file=source_file,
            function_name=function_name,
            output_name=output_name,
            status=TaskStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
        )

        logger.info(f"Starting analysis workflow for: {function_name} (task: {task_id})")

        try:
            # Phase 1: Code Analysis
            function_info = await self._phase_code_analysis(
                project_name=project_name,
                function_name=function_name,
                source_file=source_file,
                custom_knowledge=custom_knowledge,
            )

            # Phase 2: Context Building
            analysis_context = await self._phase_context_building(
                project_name=project_name,
                function_info=function_info,
            )

            # Phase 3: Model Generation
            generation_result = await self._phase_model_generation(
                context=analysis_context,
                custom_knowledge=custom_knowledge,
                output_name=output_name,
            )

            # Create task result
            self._current_task.status = TaskStatus.COMPLETED
            self._current_task.completed_at = datetime.now()

            task_result = TaskResult(
                task_id=self._current_task.task_id,
                success=True,
                xml_content=generation_result.xml_content,
                data_models=generation_result.data_models,
                duration_seconds=self._current_task.duration_seconds or 0.0,
            )

            # Save final results
            await self._save_results(task_result)

            logger.info(f"Analysis workflow completed for: {function_name}")

            return self._create_result(
                success=True,
                data=task_result,
            )

        except Exception as e:
            logger.error(f"Analysis workflow failed: {e}")

            self._current_task.status = TaskStatus.FAILED
            self._current_task.error_message = str(e)

            return self._create_result(
                success=False,
                error=str(e),
            )

    async def _phase_code_analysis(
        self,
        project_name: str,
        function_name: str,
        source_file: str,
        custom_knowledge: str = "",
    ) -> FunctionInfo:
        """Execute code analysis phase.

        Args:
            project_name: Project name
            function_name: Function name
            source_file: Source file
            custom_knowledge: Custom knowledge

        Returns:
            FunctionInfo from analysis

        Raises:
            AnalysisError: If analysis fails
        """
        logger.info(f"Phase 1: Code Analysis - {function_name}")

        result = await self.code_analyzer.run(
            project_name=project_name,
            function_name=function_name,
            source_file=source_file,
            custom_knowledge=custom_knowledge,
        )

        if not result.success:
            raise AnalysisError(f"Code analysis failed: {result.error}")

        function_info: FunctionInfo = result.data

        # Save intermediate result
        await self._save_intermediate(
            stage="code_analysis",
            data=function_info.model_dump(),
        )

        return function_info

    async def _phase_context_building(
        self,
        project_name: str,
        function_info: FunctionInfo,
    ) -> AnalysisContext:
        """Execute context building phase.

        Args:
            project_name: Project name
            function_info: Function information

        Returns:
            AnalysisContext with gathered information

        Raises:
            AnalysisError: If context building fails
        """
        logger.info(f"Phase 2: Context Building - {function_info.name}")

        result = await self.context_builder.run(
            project_name=project_name,
            function_info=function_info,
        )

        if not result.success:
            raise AnalysisError(f"Context building failed: {result.error}")

        context: AnalysisContext = result.data

        # Save intermediate result
        await self._save_intermediate(
            stage="context_building",
            data=context.model_dump(),
        )

        return context

    async def _phase_model_generation(
        self,
        context: AnalysisContext,
        custom_knowledge: str = "",
        output_name: str | None = None,
    ) -> GenerationResult:
        """Execute model generation phase.

        Args:
            context: Analysis context
            custom_knowledge: Custom knowledge
            output_name: Output model name

        Returns:
            GenerationResult with XML content

        Raises:
            AnalysisError: If generation fails
        """
        logger.info(f"Phase 3: Model Generation - {context.function_info.name}")

        result = await self.model_generator.run(
            context=context,
            custom_knowledge=custom_knowledge,
            output_name=output_name,
        )

        if not result.success:
            raise AnalysisError(f"Model generation failed: {result.error}")

        generation_result: GenerationResult = result.data

        # Save intermediate result
        await self._save_intermediate(
            stage="model_generation",
            data={
                "xml_content": generation_result.xml_content,
                "data_models": generation_result.data_models,
            },
        )

        return generation_result

    async def _save_intermediate(
        self,
        stage: str,
        data: dict[str, Any],
    ) -> None:
        """Save intermediate result.

        Args:
            stage: Analysis stage name
            data: Result data to save
        """
        if not self.storage or not self._current_task:
            return

        try:
            intermediate = IntermediateResult(
                task_id=self._current_task.task_id,
                stage=stage,
                data=data,
                timestamp=datetime.now(),
            )

            await self.storage.save(
                category="intermediate",
                key=f"{self._current_task.task_id}_{stage}",
                data=intermediate.model_dump(),
            )

            self._intermediate_results[stage] = data

        except Exception as e:
            logger.warning(f"Failed to save intermediate result: {e}")

    async def _save_results(self, task_result: TaskResult) -> None:
        """Save final task results.

        Args:
            task_result: Complete task result
        """
        if not self.storage:
            return

        try:
            await self.storage.save(
                category="results",
                key=task_result.task_id,
                data=task_result.model_dump(),
            )
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

    async def analyze_function(
        self,
        project_name: str,
        source_file: str,
        function_name: str,
        **kwargs: Any,
    ) -> TaskResult:
        """Convenience method to analyze a function.

        Args:
            project_name: Project name
            source_file: Source file
            function_name: Function name
            **kwargs: Additional arguments

        Returns:
            TaskResult with complete analysis

        Raises:
            AnalysisError: If analysis fails
        """
        result = await self.run(
            project_name=project_name,
            source_file=source_file,
            function_name=function_name,
            **kwargs,
        )

        if not result.success:
            raise AnalysisError(f"Analysis failed: {result.error}")

        return result.data

    def get_intermediate_result(self, stage: str) -> dict[str, Any] | None:
        """Get intermediate result for a stage.

        Args:
            stage: Stage name

        Returns:
            Intermediate result data or None
        """
        return self._intermediate_results.get(stage)
