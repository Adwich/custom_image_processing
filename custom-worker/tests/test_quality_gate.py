import numpy as np

from src.quality import evaluate_quality_gate


def test_quality_gate_metrics_mask_area_ratio() -> None:
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[20:80, 30:70] = 255

    result = evaluate_quality_gate(alpha, "body")

    assert "mask_area_ratio" in result.metrics
    assert result.metrics["mask_area_ratio"] > 0
    assert result.passed is True


def test_quality_gate_detects_multiple_large_components() -> None:
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[5:35, 5:35] = 255
    alpha[35:65, 35:65] = 255
    alpha[65:95, 65:95] = 255

    result = evaluate_quality_gate(alpha, "body")

    assert result.passed is False
    assert "too_many_large_components" in result.reasons
