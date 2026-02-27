import numpy as np

from src.head_mode import FaceDetection, evaluate_head_specific_quality
from src.human_parsing import compose_head_part_mask


def test_compose_head_part_mask_includes_and_excludes_labels() -> None:
    hair = np.zeros((6, 6), dtype=bool)
    hair[1:3, 1:3] = True
    face = np.zeros((6, 6), dtype=bool)
    face[2:5, 2:5] = True
    torso = np.zeros((6, 6), dtype=bool)
    torso[4:6, 0:6] = True

    mask = compose_head_part_mask(
        {
            "hair": hair,
            "face": face,
            "torso": torso,
        },
        include_labels=("hair", "face", "head", "neck"),
        exclude_labels=("torso", "upper_clothes"),
    )

    assert mask is not None
    assert bool(mask[2, 2]) is True
    assert bool(mask[5, 2]) is False


def test_head_specific_quality_flags_face_coverage_and_torso_leakage() -> None:
    alpha = np.zeros((120, 120), dtype=np.uint8)
    alpha[1:120, 20:95] = 255
    face = FaceDetection(box_xyxy=(35, 15, 75, 55), score=0.9)

    reasons, metrics = evaluate_head_specific_quality(
        alpha=alpha,
        face=face,
        require_face_detection=True,
        face_coverage_min_ratio=0.70,
        top_clip_margin_ratio=0.02,
        torso_leakage_max_ratio=0.35,
    )

    assert "head_top_clipped" in reasons
    assert "head_torso_leakage" in reasons
    assert metrics["head_alpha_below_neck_ratio"] > 0.35


def test_head_specific_quality_requires_face_when_missing() -> None:
    alpha = np.zeros((80, 80), dtype=np.uint8)
    alpha[15:60, 20:65] = 255
    reasons, metrics = evaluate_head_specific_quality(
        alpha=alpha,
        face=None,
        require_face_detection=True,
        face_coverage_min_ratio=0.65,
        top_clip_margin_ratio=0.01,
        torso_leakage_max_ratio=0.42,
    )

    assert "head_face_not_detected" in reasons
    assert metrics["head_face_coverage_ratio"] == 0.0
