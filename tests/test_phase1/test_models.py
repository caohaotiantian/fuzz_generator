"""Test data models."""

from fuzz_generator.models import (
    AnalysisContext,
    AnalysisTask,
    BatchTask,
    BlobElement,
    BlockElement,
    ChoiceElement,
    ControlFlowInfo,
    DataFlowPath,
    DataModel,
    FunctionInfo,
    GenerationResult,
    NumberElement,
    ParameterDirection,
    ParameterInfo,
    StringElement,
    TaskResult,
    TaskStatus,
)


class TestTaskModels:
    """Test task models."""

    def test_analysis_task_creation(self):
        """Test creating analysis task."""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
        )
        assert task.status == TaskStatus.PENDING
        assert task.output_name is None
        assert task.priority == 0
        assert task.depends_on == []

    def test_analysis_task_serialization(self):
        """Test task serialization."""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
        )
        json_str = task.model_dump_json()
        loaded = AnalysisTask.model_validate_json(json_str)

        assert loaded.task_id == task.task_id
        assert loaded.source_file == task.source_file
        assert loaded.function_name == task.function_name

    def test_analysis_task_state_transitions(self):
        """Test task state transitions."""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
        )

        assert task.status == TaskStatus.PENDING
        assert task.started_at is None

        task.start()
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        task.complete()
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.duration_seconds is not None

    def test_analysis_task_failure(self):
        """Test task failure."""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
        )

        task.start()
        task.fail("Something went wrong")

        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Something went wrong"

    def test_batch_task_creation(self):
        """Test batch task creation."""
        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file=f"file{i}.c",
                function_name=f"func{i}",
            )
            for i in range(3)
        ]

        batch = BatchTask(
            batch_id="batch_001",
            project_path="/path/to/project",
            tasks=tasks,
        )

        assert len(batch.tasks) == 3
        assert batch.completed_count == 0
        assert batch.pending_count == 3
        assert not batch.is_complete

    def test_batch_task_progress(self):
        """Test batch task progress tracking."""
        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file="main.c",
                function_name=f"func{i}",
            )
            for i in range(3)
        ]

        batch = BatchTask(
            batch_id="batch_001",
            project_path="/path",
            tasks=tasks,
        )

        tasks[0].complete()
        tasks[1].fail("error")

        assert batch.completed_count == 1
        assert batch.failed_count == 1
        assert batch.pending_count == 1
        assert not batch.is_complete

        tasks[2].complete()
        assert batch.is_complete

    def test_batch_get_next_task(self):
        """Test getting next task with dependencies."""
        task1 = AnalysisTask(
            task_id="task_1",
            source_file="main.c",
            function_name="func1",
        )
        task2 = AnalysisTask(
            task_id="task_2",
            source_file="main.c",
            function_name="func2",
            depends_on=["task_1"],
        )

        batch = BatchTask(
            batch_id="batch_001",
            project_path="/path",
            tasks=[task1, task2],
        )

        # task2 depends on task1, so task1 should be next
        next_task = batch.get_next_task()
        assert next_task.task_id == "task_1"

        # After task1 completes, task2 should be available
        task1.complete()
        next_task = batch.get_next_task()
        assert next_task.task_id == "task_2"

    def test_task_result(self):
        """Test task result model."""
        result = TaskResult(
            task_id="task_001",
            success=True,
            xml_content="<DataModel>...</DataModel>",
            data_models=[{"name": "TestModel"}],
            duration_seconds=5.5,
        )

        assert result.success
        assert result.xml_content is not None
        assert len(result.data_models) == 1


class TestFunctionModels:
    """Test function information models."""

    def test_parameter_info(self):
        """Test parameter info model."""
        param = ParameterInfo(
            name="buffer",
            type="char*",
            direction=ParameterDirection.IN,
            description="Input buffer",
        )

        assert param.name == "buffer"
        assert param.constraints == []
        assert not param.is_optional

    def test_function_info(self):
        """Test function info model."""
        func = FunctionInfo(
            name="process",
            file_path="main.c",
            line_number=10,
            return_type="int",
            parameters=[
                ParameterInfo(
                    name="buf",
                    type="char*",
                    direction=ParameterDirection.IN,
                ),
                ParameterInfo(
                    name="len",
                    type="int",
                    direction=ParameterDirection.IN,
                ),
            ],
            source_code="int process(char* buf, int len) { return 0; }",
        )

        assert func.name == "process"
        assert len(func.parameters) == 2
        assert func.description == ""

    def test_function_info_signature(self):
        """Test function signature property."""
        func = FunctionInfo(
            name="test",
            file_path="test.c",
            line_number=1,
            return_type="void",
            parameters=[
                ParameterInfo(name="x", type="int", direction=ParameterDirection.IN),
            ],
        )

        signature = func.full_signature
        assert "void" in signature
        assert "test" in signature
        assert "int x" in signature

    def test_input_output_parameters(self):
        """Test input/output parameter filtering."""
        func = FunctionInfo(
            name="func",
            file_path="test.c",
            line_number=1,
            return_type="int",
            parameters=[
                ParameterInfo(name="in_param", type="int", direction=ParameterDirection.IN),
                ParameterInfo(name="out_param", type="int*", direction=ParameterDirection.OUT),
                ParameterInfo(name="inout_param", type="int*", direction=ParameterDirection.INOUT),
            ],
        )

        inputs = func.input_parameters
        outputs = func.output_parameters

        assert len(inputs) == 2  # in and inout
        assert len(outputs) == 2  # out and inout


class TestAnalysisModels:
    """Test analysis result models."""

    def test_dataflow_path(self):
        """Test data flow path model."""
        from fuzz_generator.models.analysis_result import DataFlowNode

        path = DataFlowPath(
            source=DataFlowNode(code="gets(buf)", file="main.c", line=10),
            sink=DataFlowNode(code="system(buf)", file="main.c", line=20),
            path_length=3,
        )

        assert path.source.code == "gets(buf)"
        assert path.sink.line == 20

    def test_control_flow_info(self):
        """Test control flow info model."""
        cf = ControlFlowInfo(
            has_loops=True,
            has_conditions=True,
            complexity=5,
        )

        assert cf.has_loops
        assert cf.complexity == 5

    def test_analysis_context(self):
        """Test analysis context model."""
        func = FunctionInfo(
            name="test",
            file_path="test.c",
            line_number=1,
            return_type="void",
        )

        context = AnalysisContext(
            function_info=func,
            data_flows=[],
            control_flow=ControlFlowInfo(),
        )

        assert context.function_info.name == "test"
        assert not context.has_data_flows

    def test_generation_result(self):
        """Test generation result model."""
        result = GenerationResult(
            success=True,
            xml_content="<DataModel>...</DataModel>",
            model_count=2,
        )

        assert result.success
        assert result.model_count == 2


class TestXMLModels:
    """Test XML DataModel structures."""

    def test_string_element(self):
        """Test String element."""
        elem = StringElement(
            name="Method",
            value="GET",
            token=True,
            mutable=False,
        )

        assert elem.name == "Method"
        assert elem.token is True

        attrs = elem.to_xml_attrs()
        assert attrs["name"] == "Method"
        assert attrs["value"] == "GET"
        assert attrs["token"] == "true"
        assert attrs["mutable"] == "false"

    def test_number_element(self):
        """Test Number element."""
        elem = NumberElement(
            name="Length",
            size=32,
            signed=False,
            endian="little",
        )

        assert elem.name == "Length"
        assert elem.size == 32

        attrs = elem.to_xml_attrs()
        assert attrs["size"] == "32"
        assert attrs["endian"] == "little"

    def test_blob_element(self):
        """Test Blob element."""
        elem = BlobElement(
            name="Data",
            length=100,
        )

        assert elem.name == "Data"

        attrs = elem.to_xml_attrs()
        assert attrs["length"] == "100"

    def test_block_element(self):
        """Test Block element."""
        elem = BlockElement(
            name="Header",
            ref="HeaderLine",
            min_occurs=0,
            max_occurs="unbounded",
        )

        assert elem.ref == "HeaderLine"

        attrs = elem.to_xml_attrs()
        assert attrs["ref"] == "HeaderLine"
        assert attrs["minOccurs"] == "0"
        assert attrs["maxOccurs"] == "unbounded"

    def test_choice_element(self):
        """Test Choice element."""
        elem = ChoiceElement(
            name="EndChoice",
            options=[
                StringElement(name="CRLF", value="\\r\\n", token=True),
                StringElement(name="LF", value="\\n", token=True),
            ],
        )

        assert len(elem.options) == 2

    def test_datamodel(self):
        """Test DataModel."""
        model = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method"),
                StringElement(name="Space", value=" ", token=True),
                BlockElement(name="End", ref="CrLf"),
            ],
        )

        assert model.name == "Request"
        assert len(model.elements) == 3

    def test_datamodel_serialization(self):
        """Test DataModel serialization."""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field")],
        )

        dict_repr = model.to_dict()
        assert dict_repr["name"] == "Test"
        assert len(dict_repr["elements"]) == 1

        # Test round-trip
        loaded = DataModel.from_dict(dict_repr)
        assert loaded.name == model.name
