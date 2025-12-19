"""Performance benchmark tests."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fuzz_generator.batch import BatchExecutor, TaskParser
from fuzz_generator.generators import XMLGenerator, XMLValidator
from fuzz_generator.models import TaskResult
from fuzz_generator.models.xml_models import (
    BlockElement,
    ChoiceElement,
    DataModel,
    StringElement,
)
from fuzz_generator.storage import CacheManager, JsonStorage


class TestParserPerformance:
    """Performance tests for task parser."""

    def test_parse_large_task_file(self, tmp_path: Path):
        """Test parsing large task file."""
        # Generate large task file
        task_file = tmp_path / "large_tasks.yaml"

        tasks = []
        for i in range(100):
            tasks.append(f'''  - source_file: "file{i}.c"
    function_name: "func{i}"
    output_name: "Model{i}"''')

        task_file.write_text(f"""project_path: "/path/to/project"
description: "Large batch"
tasks:
{chr(10).join(tasks)}
""")

        parser = TaskParser()

        start = time.perf_counter()
        batch = parser.parse(str(task_file))
        duration = time.perf_counter() - start

        assert len(batch.tasks) == 100
        assert duration < 1.0, f"Parsing took too long: {duration:.3f}s"

    def test_validation_performance(self, tmp_path: Path):
        """Test batch validation performance."""
        from fuzz_generator.models import AnalysisTask, BatchTask

        # Create large batch
        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file=f"file{i}.c",
                function_name=f"func{i}",
            )
            for i in range(200)
        ]

        batch = BatchTask(
            batch_id="large_batch",
            project_path=str(tmp_path),
            tasks=tasks,
        )

        parser = TaskParser()

        start = time.perf_counter()
        parser.validate_batch(batch)
        duration = time.perf_counter() - start

        # Should be fast
        assert duration < 0.5, f"Validation took too long: {duration:.3f}s"


class TestXMLGeneratorPerformance:
    """Performance tests for XML generator."""

    @pytest.fixture
    def complex_model(self) -> DataModel:
        """Create complex DataModel for testing."""
        elements = []
        for i in range(50):
            elements.append(StringElement(name=f"Field{i}", value=f"value{i}"))

        # Add some blocks
        for i in range(10):
            elements.append(
                BlockElement(
                    name=f"Block{i}",
                    children=[StringElement(name=f"Inner{j}", value=f"v{j}") for j in range(5)],
                )
            )

        # Add some choices
        for i in range(5):
            elements.append(
                ChoiceElement(
                    name=f"Choice{i}",
                    options=[StringElement(name=f"Opt{j}", value=f"val{j}") for j in range(3)],
                )
            )

        return DataModel(
            name="ComplexModel",
            description="Complex model for performance testing",
            elements=elements,
        )

    def test_single_model_generation(self, complex_model: DataModel):
        """Test single complex model generation."""
        generator = XMLGenerator()

        start = time.perf_counter()
        for _ in range(100):
            generator.generate_single(complex_model)
        duration = time.perf_counter() - start

        avg_time = duration / 100
        assert avg_time < 0.1, f"Average generation time too slow: {avg_time:.4f}s"

    def test_multiple_models_generation(self, complex_model: DataModel):
        """Test generating multiple models."""
        generator = XMLGenerator()
        models = [complex_model] * 20

        start = time.perf_counter()
        xml_str = generator.generate(models)
        duration = time.perf_counter() - start

        assert duration < 1.0, f"Multi-model generation too slow: {duration:.3f}s"
        assert xml_str  # Non-empty output

    def test_file_output_performance(self, complex_model: DataModel, tmp_path: Path):
        """Test file output performance."""
        generator = XMLGenerator()
        models = [complex_model] * 10

        output_file = tmp_path / "output.xml"

        start = time.perf_counter()
        generator.generate_to_file(models, str(output_file))
        duration = time.perf_counter() - start

        assert output_file.exists()
        assert duration < 1.0, f"File output too slow: {duration:.3f}s"


class TestXMLValidatorPerformance:
    """Performance tests for XML validator."""

    @pytest.fixture
    def large_xml(self) -> str:
        """Generate large XML for testing."""
        models = []
        for i in range(50):
            elements = "".join(f'<String name="Field{j}" value="value{j}" />' for j in range(20))
            models.append(f'<DataModel name="Model{i}">{elements}</DataModel>')

        return f"""<?xml version="1.0" encoding="utf-8"?>
<Secray>
{"".join(models)}
</Secray>"""

    def test_validate_large_xml(self, large_xml: str):
        """Test validating large XML document."""
        validator = XMLValidator()

        start = time.perf_counter()
        result = validator.validate(large_xml)
        duration = time.perf_counter() - start

        assert result.is_valid
        assert duration < 0.5, f"Validation too slow: {duration:.3f}s"

    def test_repeated_validation(self, large_xml: str):
        """Test repeated validation."""
        validator = XMLValidator()

        start = time.perf_counter()
        for _ in range(50):
            validator.validate(large_xml)
        duration = time.perf_counter() - start

        avg_time = duration / 50
        assert avg_time < 0.05, f"Average validation too slow: {avg_time:.4f}s"


class TestStoragePerformance:
    """Performance tests for storage operations."""

    @pytest.mark.asyncio
    async def test_save_load_performance(self, tmp_path: Path):
        """Test save and load performance."""
        storage = JsonStorage(base_dir=tmp_path)

        # Save many items
        start = time.perf_counter()
        for i in range(100):
            await storage.save(
                category="test",
                key=f"item_{i}",
                data={"index": i, "value": f"data_{i}" * 100},
            )
        save_duration = time.perf_counter() - start

        # Load all items
        start = time.perf_counter()
        for i in range(100):
            await storage.load(category="test", key=f"item_{i}")
        load_duration = time.perf_counter() - start

        assert save_duration < 2.0, f"Save too slow: {save_duration:.3f}s"
        assert load_duration < 1.0, f"Load too slow: {load_duration:.3f}s"

    @pytest.mark.asyncio
    async def test_cache_performance(self, tmp_path: Path):
        """Test cache performance."""
        cache = CacheManager(base_dir=tmp_path)

        # Test cache miss then hit
        key = "test_key"
        value = {"data": "x" * 1000}

        # Cache miss
        start = time.perf_counter()
        result = await cache.get(key, category="analysis")
        time.perf_counter() - start
        assert result is None

        # Set value (key, data, category)
        await cache.set(key, value, category="analysis")

        # Cache hit
        start = time.perf_counter()
        for _ in range(100):
            result = await cache.get(key, category="analysis")
        hit_duration = time.perf_counter() - start

        avg_hit = hit_duration / 100
        assert avg_hit < 0.01, f"Cache hit too slow: {avg_hit:.4f}s"


class TestBatchExecutorPerformance:
    """Performance tests for batch executor."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self, tmp_path: Path):
        """Test sequential task execution performance."""
        from fuzz_generator.models import AnalysisTask, BatchTask

        # Create batch with many tasks
        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file=f"file{i}.c",
                function_name=f"func{i}",
            )
            for i in range(20)
        ]

        batch = BatchTask(
            batch_id="perf_batch",
            project_path=str(tmp_path),
            tasks=tasks,
        )

        # Mock fast orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(
            return_value=MagicMock(
                success=True,
                data=TaskResult(task_id="test", success=True),
            )
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator)

        start = time.perf_counter()
        result = await executor.execute(batch, max_concurrent=1)
        duration = time.perf_counter() - start

        assert result.completed_tasks == 20
        # With mocked orchestrator, should be very fast
        assert duration < 2.0, f"Sequential execution too slow: {duration:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, tmp_path: Path):
        """Test concurrent task execution performance."""
        from fuzz_generator.models import AnalysisTask, BatchTask

        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file=f"file{i}.c",
                function_name=f"func{i}",
            )
            for i in range(20)
        ]

        batch = BatchTask(
            batch_id="perf_batch",
            project_path=str(tmp_path),
            tasks=tasks,
        )

        mock_orchestrator = AsyncMock()

        async def slow_run(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            return MagicMock(
                success=True,
                data=TaskResult(task_id="test", success=True),
            )

        mock_orchestrator.run = slow_run

        executor = BatchExecutor(orchestrator=mock_orchestrator)

        # Concurrent should be faster
        start = time.perf_counter()
        result = await executor.execute(batch, max_concurrent=5)
        concurrent_duration = time.perf_counter() - start

        assert result.completed_tasks == 20
        # Concurrent should complete faster than sequential
        assert concurrent_duration < 1.0, f"Concurrent too slow: {concurrent_duration:.3f}s"
