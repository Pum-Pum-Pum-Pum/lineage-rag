from app.ingestion.filename_parser import parse_document_filename


def test_parse_document_filename_valid_case() -> None:
    parsed = parse_document_filename("FS_FCIS_14.4.0.0.0$ASNB_R12.docx")

    assert parsed.document_name == "FS_FCIS_14.4.0.0.0$ASNB_R12.docx"
    assert parsed.document_family == "FS_FCIS_14.4.0.0.0$ASNB"
    assert parsed.release_label == "R12"
    assert parsed.release_number == 12
    assert parsed.variant_suffix is None
    assert parsed.source_type == "docx"


def test_parse_document_filename_with_suffix() -> None:
    parsed = parse_document_filename(
        "FS_FCIS_14.4.0.0.0$ASNB_R2_PNB_Branch Online Reports(BOR)_v1.2.docx"
    )

    assert parsed.document_family == "FS_FCIS_14.4.0.0.0$ASNB"
    assert parsed.release_label == "R2"
    assert parsed.release_number == 2
    assert parsed.variant_suffix == "PNB_Branch Online Reports(BOR)_v1.2"


def test_parse_document_filename_with_another_suffix_pattern() -> None:
    parsed = parse_document_filename(
        "FS_FCIS_14.7.0.0.0$ASNB_R24_Teller_Branch_Reports_Realignment_v1.0.docx"
    )

    assert parsed.document_family == "FS_FCIS_14.7.0.0.0$ASNB"
    assert parsed.release_label == "R24"
    assert parsed.release_number == 24
    assert parsed.variant_suffix == "Teller_Branch_Reports_Realignment_v1.0"


def test_parse_document_filename_invalid_case() -> None:
    try:
        parse_document_filename("invalid_filename.docx")
    except ValueError as exc:
        assert "expected release-aware pattern" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid filename pattern")
