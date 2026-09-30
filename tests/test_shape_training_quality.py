from visual_system.services.neural_shape_model import OUTLINE_CLASSES, SOLID_CLASSES
from visual_system.train_shapes import _deployment_report


def test_training_quality_gate_rejects_incomplete_class_coverage():
    report = _deployment_report(
        {
            "oval": 25,
            "hexagon": 25,
            "sphere": 25,
            "cylinder": 25,
            "cube": 25,
            "cuboid": 25,
            "cone": 25,
        },
        {
            "outline": {
                "accuracy": 1.0,
                "classes": {
                    "oval": {"count": 6, "precision": 1.0, "recall": 1.0},
                    "hexagon": {"count": 6, "precision": 1.0, "recall": 1.0},
                },
            },
            "solid": {
                "accuracy": 1.0,
                "classes": {
                    name: {"count": 6, "precision": 1.0, "recall": 1.0}
                    for name in SOLID_CLASSES
                },
            },
        },
    )

    assert "outline" not in report["deployable_heads"]
    assert "solid" in report["deployable_heads"]
    assert any("circle:" in item for item in report["heads"]["outline"]["failures"])


def test_training_quality_gate_requires_high_metrics_for_every_class():
    counts = {label: 30 for label in (*OUTLINE_CLASSES, *SOLID_CLASSES)}
    metrics = {
        head: {
            "accuracy": 0.9,
            "classes": {
                label: {"count": 6, "precision": 0.9, "recall": 0.9}
                for label in classes
            },
        }
        for head, classes in (("outline", OUTLINE_CLASSES), ("solid", SOLID_CLASSES))
    }
    metrics["solid"]["classes"]["cone"]["precision"] = 0.6

    report = _deployment_report(counts, metrics)

    assert report["deployable_heads"] == ["outline"]
    assert any("cone: precision" in item for item in report["heads"]["solid"]["failures"])
