"""Batch input must terminate without a Python prompt or welcome banner."""
import io
import sys

import pytest
import chatrepl


@pytest.mark.parametrize("prompt,stdin_text,expected", [
    (["one", "shot"], "unused", "one shot"),
    ([], "piped request\n", "piped request\n"),
    ([], "  indented\n", "  indented\n"),
    ([], " \n", None),
    ([], "", None),
])
def test_batch_never_starts_console(monkeypatch, prompt, stdin_text, expected):
    monkeypatch.setattr(sys, "argv", ["chatrepl", "-k", "test", "-u", "http://example.invalid/v1", "-m", "test", "--no-context-files"] + prompt)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    turns = []
    monkeypatch.setattr(chatrepl, "run_agent_turn", lambda conv, stream, text, **kwargs: turns.append(text))
    def unexpected_console(*args, **kwargs):
        raise AssertionError("Batch mode must not create a console")
    monkeypatch.setattr(chatrepl, "InteractiveConsole", unexpected_console)
    chatrepl.main()
    assert turns == ([] if expected is None else [expected])


def test_batch_failure_has_nonzero_exit_and_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["chatrepl", "-k", "test", "-u", "http://example.invalid/v1", "-m", "test", "--no-context-files", "hello"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    def fail(*args, **kwargs):
        raise RuntimeError("Incomplete completion (finish_reason='length')")
    monkeypatch.setattr(chatrepl, "run_agent_turn", fail)
    with pytest.raises(SystemExit) as caught:
        chatrepl.main()
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert "finish_reason='length'" in output.err
    assert "Traceback" not in output.err
