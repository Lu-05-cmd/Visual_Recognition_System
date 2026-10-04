from visual_system.services.neural_shape_model import OUTLINE_CLASSES, SOLID_CLASSES
from visual_system.train_shapes import _deployment_report, _print_solid_metrics_table


def test_training_quality_gate_rejects_incomplete_class_coverage():
    report = _deployment_report(
        {
            "oval": 25,
            "hexagon": 25,
            "sphere": 25,
            "cylinder": 25,
            "box": 50,
            "cone": 25,
            "hexagonal_prism": 25,
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
    metrics["solid"]["classes"]["cone"]["precision"] = 0.625

    report = _deployment_report(counts, metrics)

    assert report["deployable_heads"] == ["outline"]
    assert any("cone: precision" in item for item in report["heads"]["solid"]["failures"])


def test_solid_quality_gate_accepts_metrics_at_seventy_percent():
    counts = {label: 25 for label in (*OUTLINE_CLASSES, *SOLID_CLASSES)}
    metrics = {
        "solid": {
            "accuracy": 0.7,
            "classes": {
                label: {"count": 5, "precision": 0.7, "recall": 0.7}
                for label in SOLID_CLASSES
            },
        }
    }

    report = _deployment_report(counts, metrics)

    assert "solid" in report["deployable_heads"]


def test_quality_gate_rejects_negative_class_below_train_minimum():
    train_counts = {
        "hexagon": 18,
        **{label: 25 for label in SOLID_CLASSES if label != "none"},
    }
    metrics = {
        "solid": {
            "accuracy": 0.9,
            "classes": {
                label: {"count": 5, "precision": 0.9, "recall": 0.9}
                for label in SOLID_CLASSES
            },
        }
    }

    report = _deployment_report(train_counts, metrics)

    assert "solid" not in report["deployable_heads"]
    assert "none: train mẫu 18 < 20" in report["heads"]["solid"]["failures"]


def test_solid_metrics_table_shows_per_class_counts_rates_and_gate(capsys):
    metrics = {
        "solid_accuracy": 0.8,
        "solid_classes": {
            label: {"count": 5, "precision": 0.9, "recall": 0.9}
            for label in SOLID_CLASSES
        },
    }
    report = _deployment_report(
        {
            "sphere": 30,
            "cylinder": 30,
            "cone": 30,
            "box": 30,
            "hexagonal_prism": 30,
            "oval": 25,
        },
        {"solid": {"accuracy": 0.8, "classes": metrics["solid_classes"]}},
    )

    _print_solid_metrics_table(
        {
            "sphere": 30,
            "cylinder": 30,
            "cone": 30,
            "box": 30,
            "hexagonal_prism": 30,
            "oval": 25,
        },
        metrics,
        report,
    )

    output = capsys.readouterr().out
    assert "Class      | Train |   Val | Precision |  Recall | Gate" in output
    assert f"{'sphere':<10} | {25:>5} | {5:>5} | {0.9:>8.1%} | {0.9:>6.1%} | {'PASS':>4}" in output
    assert f"{'none':<10} | {20:>5} | {5:>5} | {0.9:>8.1%} | {0.9:>6.1%} | {'PASS':>4}" in output
    assert "Solid accuracy: 80.0% (required 70%) | Deployment gate: PASS" in output
