import json
from pathlib import Path

from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord


def test_write_embedding_batch_to_json(tmp_path: Path) -> None:
    batch = EmbeddingBatch(
        document_name="FS_FCIS_14.7.0.0.0$ASNB_R24_Test.docx",
        total_records=2,
        records=[
            EmbeddingRecord(
                unit_id="doc::chunk_0",
                unit_index=0,
                source_kind="paragraph",
                document_family="FS_FCIS_14.7.0.0.0$ASNB",
                release_label="R24",
                text="Paragraph text",
                embedding_model="text-embedding-3-large",
                embedding_status="embedded",
                vector=[0.1, 0.2, 0.3],
            ),
            EmbeddingRecord(
                unit_id="doc::table_chunk_0",
                unit_index=1,
                source_kind="table",
                document_family="FS_FCIS_14.7.0.0.0$ASNB",
                release_label="R24",
                text="Column | Description",
                embedding_model="text-embedding-3-large",
                embedding_status="embedded",
                vector=[0.4, 0.5, 0.6],
            ),
        ],
    )

    output_file = write_embedding_batch_to_json(batch, tmp_path / "embeddings")

    assert output_file.exists()
    assert output_file.name == "FS_FCIS_14.7.0.0.0$ASNB_R24_Test.embeddings.json"

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["document_name"] == "FS_FCIS_14.7.0.0.0$ASNB_R24_Test.docx"
    assert payload["total_records"] == 2
    assert payload["records"][0]["source_kind"] == "paragraph"
    assert payload["records"][1]["source_kind"] == "table"
    assert payload["records"][1]["vector"] == [0.4, 0.5, 0.6]
