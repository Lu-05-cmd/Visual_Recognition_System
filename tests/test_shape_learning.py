from shape_synthetic import FLAT_SHAPES, make_scene, scene_box
from visual_system.services.shape_library import ShapeFeatureLibrary
from visual_system.services.shape_recognizer import ShapeRecognizer


def test_shape_library_learns_separated_2d_classes_from_samples(tmp_path) -> None:
    library = ShapeFeatureLibrary(tmp_path / "trained-shapes.sqlite3")
    failures = []
    for label in FLAT_SHAPES:
        for seed in range(4):
            image = make_scene(label, seed)
            x1, y1, x2, y2 = scene_box(label, seed)
            added, _, _ = library.add_image(image[y1:y2, x1:x2], label)
            assert added

    for label in FLAT_SHAPES:
        for seed in range(4, 8):
            image = make_scene(label, seed)
            x1, y1, x2, y2 = scene_box(label, seed)
            extracted = ShapeRecognizer._extract_contour(image[y1:y2, x1:x2], "blue")
            prediction = library.classify(extracted, labels=set(FLAT_SHAPES))
            if prediction != label:
                failures.append((label, seed, prediction))

    assert not failures, f"Few-shot 2D shape errors: {failures}"
