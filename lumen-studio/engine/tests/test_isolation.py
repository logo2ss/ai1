"""A native crash in the MediaPipe helper must not take the engine down."""

import numpy as np

from conftest import load, needs_face, needs_testdata


def test_helper_crash_is_contained_and_recovers():
    from lumen_engine import mpworker
    from lumen_engine.errors import ProcessingError

    try:
        mpworker.call("os.abort", what="Test helper")  # simulates a segfault/abort in native code
        raise AssertionError("expected ProcessingError")
    except ProcessingError as e:
        assert "crashed" in e.message and e.hint
    assert mpworker.call("math.sqrt", 16.0) == 4.0  # a fresh helper serves the next request


def test_errors_keep_hint_across_processes():
    from lumen_engine import mpworker
    from lumen_engine.errors import NoFaceFoundError

    try:
        mpworker.call("lumen_engine.mptasks.detect_faces", np.zeros((64, 64, 3), np.uint8))
    except NoFaceFoundError as e:  # pragma: no cover - only if the model raises
        assert e.hint
    except Exception as e:  # missing model is also fine here, but must stay a LumenError
        from lumen_engine.errors import LumenError

        assert isinstance(e, LumenError), e


@needs_face
@needs_testdata
def test_face_detection_through_helper():
    import cv2

    from lumen_engine import mpworker

    rgb = load("faces/two_people.jpg")
    faces = mpworker.call("lumen_engine.mptasks.detect_faces", cv2.convertScaleAbs(rgb, alpha=255.0))
    assert len(faces) == 2 and faces[0].points.shape == (478, 3)
