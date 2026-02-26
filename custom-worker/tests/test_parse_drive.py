from src.ingest import parse_client_order_id, parse_drive_metadata, parse_quantity, parse_scent


def test_parse_drive_client_order_id_from_path() -> None:
    order_id = parse_client_order_id(
        "123456789/2x-Dog Tag/my-file.png", "123456789-file-head-2.png"
    )
    assert order_id == "123456789"


def test_parse_drive_client_order_id_from_uploadkit_path() -> None:
    order_id = parse_client_order_id(
        "UploadKit/#1004/1x-head - Black Ice/#1004-head - Black Ice-x1.png",
        "#1004-head - Black Ice-x1.png",
    )
    assert order_id == "1004"


def test_parse_drive_client_order_id_from_client_hash_path() -> None:
    order_id = parse_client_order_id(
        "UploadKit/8332656-#1004/1x-head - Black Ice/8332656-#1004-head - Black Ice-x1.png",
        "8332656-#1004-head - Black Ice-x1.png",
    )
    assert order_id == "8332656-1004"


def test_parse_drive_client_order_id_from_filename_fallback() -> None:
    order_id = parse_client_order_id("", "ABC-123-portrait-car-1.png")
    assert order_id == "ABC"


def test_parse_drive_quantity() -> None:
    quantity = parse_quantity("12345/4x-Product/file.png", "12345-file-head-4.png")
    assert quantity == 4


def test_parse_drive_quantity_from_uploadkit_nested_path() -> None:
    quantity = parse_quantity(
        "UploadKit/#1004/3x-body - Black Ice/#1004-body - Black Ice-x3.jpg",
        "#1004-body - Black Ice-x3.jpg",
    )
    assert quantity == 3


def test_parse_scent_from_uploadkit_path() -> None:
    scent = parse_scent(
        "UploadKit/#1004/1x-head - Black Ice/#1004-head - Black Ice-x1.png",
        "#1004-head - Black Ice-x1.png",
    )
    assert scent == "Black Ice"


def test_parse_scent_from_filename() -> None:
    scent = parse_scent("", "#1004-body - Ocean Breeze-x2.jpg")
    assert scent == "Ocean Breeze"


def test_parse_drive_metadata() -> None:
    parsed = parse_drive_metadata("12345/1x-car - Citrus", "12345-awesome-car - Citrus-x1.png")
    assert parsed.client_order_id == "12345"
    assert parsed.quantity == 1
    assert parsed.cut_option == "car"
    assert parsed.scent == "Citrus"
