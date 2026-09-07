import pytest

from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import PerSceneStep


class _RecordingPerSceneStep(PerSceneStep):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run_scene(self, *, image_repository, scene, input, state, scene_state):
        ids = sorted(image_repository.image_ids(scene=scene))
        self.calls.append((scene, ids))
        scene_state["ran"] = True
        return ids


class _FailingForOneScene(PerSceneStep):
    def __init__(self, bad_scene):
        super().__init__()
        self.bad_scene = bad_scene
        self.seen = []

    def run_scene(self, *, image_repository, scene, input, state, scene_state):
        self.seen.append(scene)
        if scene == self.bad_scene:
            raise RuntimeError("boom")
        return scene


@pytest.fixture
def repo():
    repository = ImageRepository()
    repository.add_images(["a.jpg", "b.jpg"], scene="s1")
    repository.add_images(["c.jpg"], scene="s2")
    return repository


def test_per_scene_step_runs_once_per_scene(repo):
    step = _RecordingPerSceneStep()
    state = {}
    result = step.run(image_repository=repo, input=None, state=state)

    assert set(result) == {"s1", "s2"}
    assert dict(step.calls) == {"s1": [0, 1], "s2": [2]}
    assert result["s1"] == [0, 1]
    assert result["s2"] == [2]


def test_per_scene_step_isolates_scene_state(repo):
    step = _RecordingPerSceneStep()
    state = {}
    step.run(image_repository=repo, input=None, state=state)

    assert state["scenes"]["s1"] == {"ran": True}
    assert state["scenes"]["s2"] == {"ran": True}


def test_per_scene_step_isolates_failures(repo):
    step = _FailingForOneScene(bad_scene="s1")
    result = step.run(image_repository=repo, input=None, state={})

    assert step.seen == ["s1", "s2"]
    assert result == {"s2": "s2"}
