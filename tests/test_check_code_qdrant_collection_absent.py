from __future__ import annotations

import pytest

from scripts.check_code_qdrant_collection_absent import main
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def test_collection_preflight_accepts_new_name_and_rejects_existing_name(tmp_path) -> None:
    qdrant_path = tmp_path / "qdrant"

    main([
        "--qdrant-path", str(qdrant_path), "--collection-name", "code_custom_r3_v2"
    ])

    client = create_persistent_qdrant_client(qdrant_path)
    try:
        client.create_collection("code_custom_r3_v2", vectors_config={"size": 3, "distance": "Cosine"})
    finally:
        client.close()

    with pytest.raises(FileExistsError, match="already exists"):
        main([
            "--qdrant-path", str(qdrant_path), "--collection-name", "code_custom_r3_v2"
        ])
