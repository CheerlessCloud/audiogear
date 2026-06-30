"""BUG-8: tqdm must stay quiet when not attached to a TTY (file logs)."""

import importlib

import audiogear.utils.progress as progress


def test_disabled_when_stderr_not_a_tty(monkeypatch):
    monkeypatch.delenv("AUDIOGEAR_PROGRESS", raising=False)

    class _NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr(progress.sys, "stderr", _NotATty())
    assert progress._progress_disabled() is True

    bar = progress.tqdm(range(3))
    assert bar.disable is True
    list(bar)  # must still iterate fully


def test_enabled_when_stderr_is_a_tty(monkeypatch):
    monkeypatch.delenv("AUDIOGEAR_PROGRESS", raising=False)

    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(progress.sys, "stderr", _Tty())
    assert progress._progress_disabled() is False


def test_env_var_forces_off(monkeypatch):
    monkeypatch.setenv("AUDIOGEAR_PROGRESS", "0")
    assert progress._progress_disabled() is True


def test_env_var_forces_on(monkeypatch):
    monkeypatch.setenv("AUDIOGEAR_PROGRESS", "1")
    assert progress._progress_disabled() is False


def test_iterates_and_preserves_values(monkeypatch):
    monkeypatch.setenv("AUDIOGEAR_PROGRESS", "0")  # disabled, but must still yield
    importlib.reload(progress)
    assert list(progress.tqdm([1, 2, 3])) == [1, 2, 3]
