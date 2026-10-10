import pytest

from kgl import tracker


def test_none_tracker_accepts_the_full_lifecycle():
    t = tracker.make_tracker("none", project="kgl-test")
    assert isinstance(t, tracker.NoneTracker)
    t.start(run_name="exp000/default", config={"seed": 1})
    t.log({"cv": 0.5}, step=1)
    t.finish()


def test_unknown_tracker_raises():
    with pytest.raises(KeyError):
        tracker.make_tracker("tensorboard", project="kgl-test")
