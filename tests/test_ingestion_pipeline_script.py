from pathlib import Path
from types import SimpleNamespace

from scripts import run_ingestion_pipeline


def test_ingestion_pipeline_writes_only_to_ingestion_output_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    active_retrieval_directory = tmp_path / "indexes" / "functional_specs_v4"
    ingestion_output_directory = tmp_path / "ingestion-work"
    settings = SimpleNamespace(
        log_level="INFO",
        raw_specs_dir=tmp_path / "raw",
        processed_dir=active_retrieval_directory,
        ingestion_output_dir=ingestion_output_directory,
    )
    discovered = SimpleNamespace(
        file_path=tmp_path / "raw" / "example.docx",
        file_name="example.docx",
    )
    artifact = SimpleNamespace(
        extracted_text=SimpleNamespace(non_empty_paragraph_count=1),
        extracted_tables=SimpleNamespace(table_count=1),
    )
    chunked = SimpleNamespace(total_chunks=1)
    retrieval_ready = SimpleNamespace(total_units=2)
    written_directories: list[Path] = []

    monkeypatch.setattr(run_ingestion_pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(run_ingestion_pipeline, "configure_logging", lambda level: None)
    monkeypatch.setattr(
        run_ingestion_pipeline,
        "discover_docx_files",
        lambda directory: [discovered],
    )
    monkeypatch.setattr(run_ingestion_pipeline, "ingest_docx_file", lambda path: artifact)
    monkeypatch.setattr(run_ingestion_pipeline, "build_normalized_artifact", lambda value: object())
    monkeypatch.setattr(run_ingestion_pipeline, "chunk_normalized_artifact", lambda value: chunked)
    monkeypatch.setattr(run_ingestion_pipeline, "chunk_tables_from_artifact", lambda value: [])
    monkeypatch.setattr(
        run_ingestion_pipeline,
        "build_retrieval_ready_artifact",
        lambda normalized, paragraphs, tables: retrieval_ready,
    )

    def record_write(value, directory: Path) -> Path:
        written_directories.append(directory)
        return directory / "artifact.json"

    monkeypatch.setattr(run_ingestion_pipeline, "write_ingested_artifact_to_json", record_write)
    monkeypatch.setattr(run_ingestion_pipeline, "write_chunked_document_to_json", record_write)
    monkeypatch.setattr(run_ingestion_pipeline, "write_retrieval_ready_artifact_to_json", record_write)

    run_ingestion_pipeline.main()

    assert written_directories == [ingestion_output_directory] * 3
    assert active_retrieval_directory not in written_directories
