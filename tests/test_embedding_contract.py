from pathlib import Path

from docx import Document

from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.table_chunker import chunk_tables_from_artifact
from app.embeddings.embedding_contract import build_embedding_batch_contract


def test_build_embedding_batch_contract(tmp_path: Path) -> None:
    file_path = tmp_path / "FS_FCIS_14.7.0.0.0$ASNB_R24_Teller_Branch_Reports_Realignment_v1.0.docx"

    document = Document()
    document.add_paragraph("Paragraph 1")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Column"
    table.cell(0, 1).text = "Description"
    table.cell(1, 0).text = "Branch Code"
    table.cell(1, 1).text = "Unique branch identifier"
    document.save(file_path)

    raw_artifact = ingest_docx_file(file_path)
    normalized_artifact = build_normalized_artifact(raw_artifact)
    paragraph_chunks = chunk_normalized_artifact(normalized_artifact, max_paragraphs_per_chunk=1)
    table_chunks = chunk_tables_from_artifact(normalized_artifact)
    retrieval_ready = build_retrieval_ready_artifact(
        normalized_artifact,
        paragraph_chunks,
        table_chunks,
    )

    batch = build_embedding_batch_contract(retrieval_ready, embedding_model="text-embedding-3-large")

    assert batch.document_name == file_path.name
    assert batch.total_records == 2
    assert batch.records[0].embedding_model == "text-embedding-3-large"
    assert batch.records[0].embedding_status == "pending"
    assert batch.records[0].vector is None
    assert batch.records[1].source_kind == "table"
