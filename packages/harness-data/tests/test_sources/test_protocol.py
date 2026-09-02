from harness.data.sources.protocol import Source, SourceMetadata


class TestSourceProtocol:
    def test_source_metadata_creation(self):
        meta = SourceMetadata(
            name="test_source",
            source_type="file",
            row_count=100,
            columns=["id", "name", "score"],
            column_types={"id": "int64", "name": "object", "score": "float64"},
        )
        assert meta.name == "test_source"
        assert meta.source_type == "file"
        assert meta.row_count == 100
        assert len(meta.columns) == 3

    def test_source_metadata_defaults(self):
        meta = SourceMetadata(name="minimal", source_type="file")
        assert meta.row_count is None
        assert meta.columns == []
        assert meta.column_types == {}
