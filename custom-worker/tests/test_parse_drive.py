from src.ingest import parse_client_order_id, parse_drive_metadata, parse_quantity


def test_parse_drive_client_order_id_from_path() -> None:
    order_id = parse_client_order_id(
        "123456789/2x-Dog Tag/my-file.png", "123456789-file-head-2.png"
    )
    assert order_id == "123456789"


def test_parse_drive_client_order_id_from_filename_fallback() -> None:
    order_id = parse_client_order_id("", "ABC-123-portrait-car-1.png")
    assert order_id == "ABC"


def test_parse_drive_quantity() -> None:
    quantity = parse_quantity("12345/4x-Product/file.png", "12345-file-head-4.png")
    assert quantity == 4


def test_parse_drive_metadata() -> None:
    parsed = parse_drive_metadata("12345/1x-Car", "12345-awesome-car-1.png")
    assert parsed.client_order_id == "12345"
    assert parsed.quantity == 1
    assert parsed.cut_option == "car"
