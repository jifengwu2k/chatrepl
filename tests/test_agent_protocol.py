# -*- coding: utf-8 -*-
"""Exercise the real agent loop across tool rounds and incomplete responses."""
from __future__ import unicode_literals

import io
import json
import sys

import pytest
import chatrepl
import chat_completions_conversation_with_tools as client


def conversation():
    return client.ChatCompletionsConversationWithTools(
        "test", "http://example.invalid/v1", "test", chatrepl.SYSTEM_PROMPT, chatrepl.TOOLS_BY_NAME)


def body_for(message, reason, stream, complete=True):
    choice = {"finish_reason": reason}
    if not stream:
        choice["message"] = message
        return json.dumps({"choices": [choice]}).encode("utf-8")
    message = dict(message)
    if "tool_calls" in message:
        message["tool_calls"] = [dict(call, index=i) for i, call in enumerate(message["tool_calls"])]
    choice["delta"] = message
    body = b"data: " + json.dumps({"choices": [choice]}).encode("utf-8") + b"\n\n"
    return body + (b"data: [DONE]\n\n" if complete else b"")


def install_responses(monkeypatch, bodies):
    requests = []
    remaining = list(bodies)
    def post(url, api_key, payload):
        requests.append(payload)
        return io.BytesIO(remaining.pop(0))
    monkeypatch.setattr(client, "open_post_response", post)
    monkeypatch.setattr(chatrepl, "SYS_STDOUT_BUFFER", io.BytesIO())
    return requests


@pytest.mark.parametrize("stream", [True, False])
def test_multiple_tool_calls_do_not_insert_blank_user_messages(monkeypatch, tmpdir, stream):
    path = str(tmpdir.join("read.txt"))
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("hello\n")
    calls = [client.ToolCall("read-%s" % i, "read", {"path": path}).to_payload() for i in range(2)]
    requests = install_responses(monkeypatch, [
        body_for({"content": "", "tool_calls": calls}, "tool_calls", stream),
        body_for({"content": "done"}, "stop", stream),
    ])
    conv = conversation()
    chatrepl.run_agent_turn(conv, stream, "Read twice")
    assert len(requests) == 2
    assert [m["role"] for m in requests[1]["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert [m for m in conv.messages if m["role"] == "user"] == [{"role": "user", "content": "Read twice"}]


@pytest.mark.parametrize("stream", [True, False])
def test_send_without_text_continues_appended_history(monkeypatch, stream):
    requests = install_responses(monkeypatch, [body_for({"content": "ok"}, "stop", stream)])
    conv = conversation()
    namespace = chatrepl.build_namespace(conv, stream, chatrepl.SYSTEM_PROMPT, False, [])
    namespace["append"]("already appended")
    namespace["send"]()
    assert [m["role"] for m in requests[0]["messages"]] == ["system", "user"]
    assert "\b" not in namespace["help_text"]


@pytest.mark.parametrize("stream,reason,complete", [
    (True, None, False), (True, "tool_calls", False), (True, None, True),
    (True, "length", True), (False, "length", True), (False, None, True),
])
def test_incomplete_write_call_never_executes(monkeypatch, tmpdir, stream, reason, complete):
    path = str(tmpdir.join("must-not-exist.txt"))
    call = client.ToolCall("write-1", "write", {"path": path, "content": "unexpected"})
    install_responses(monkeypatch, [body_for({"content": "", "tool_calls": [call.to_payload()]}, reason, stream, complete)])
    conv = conversation()
    with pytest.raises(client.CompletionError):
        chatrepl.run_agent_turn(conv, stream, "Write a file")
    assert not tmpdir.join("must-not-exist.txt").exists()
    assert len(conv.messages) == 1


def test_image_read_attaches_image_after_tool_messages(monkeypatch, tmpdir):
    path = str(tmpdir.join("pixel.png"))
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
    call = client.ToolCall("read-1", "read", {"path": path}).to_payload()
    requests = install_responses(monkeypatch, [
        body_for({"content": "", "tool_calls": [call]}, "tool_calls", False),
        body_for({"content": "done"}, "stop", False),
    ])
    chatrepl.run_agent_turn(conversation(), False, "Look at this")

    messages = requests[1]["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "user"]
    assert messages[3]["content"] == "Read image file [image/png]"
    image_part = messages[4]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_stream_reasoning_is_separated_from_answer(monkeypatch):
    body = body_for({"reasoning_content": "thought"}, None, True, complete=False)
    body += body_for({"content": "answer"}, "stop", True)
    install_responses(monkeypatch, [body])
    chatrepl.run_agent_turn(conversation(), True, "Hello")
    assert chatrepl.SYS_STDOUT_BUFFER.getvalue() == b"[reasoning]\nthought\nanswer\n"


def test_shell_result_preserves_separate_streams(monkeypatch, capfd):
    source = "import os; os.write(1,b'OUT1\\n'); os.write(2,b'ERR1\\n'); os.write(1,b'OUT2\\n'); os.write(2,b'ERR2\\n'); raise SystemExit(7)"
    monkeypatch.setattr(chatrepl, "build_live_shell_command", lambda command: [sys.executable, "-c", source])
    monkeypatch.setattr(chatrepl, "SYS_STDOUT_BUFFER", io.BytesIO())
    result = chatrepl.tool_shell({"command": "test child"})
    assert result == "Exit code: 7\n\nstdout:\nOUT1\nOUT2\n\n\nstderr:\nERR1\nERR2\n"
    out, err = capfd.readouterr()
    assert out == "OUT1\nOUT2\n"
    assert err == "ERR1\nERR2\n"


@pytest.mark.parametrize("source,expected,out,err", [
    ("import os; os.write(1, b'out')", "Exit code: 0\n\nstdout:\nout", "out", ""),
    ("import os; os.write(2, b'err'); raise SystemExit(3)",
     "Exit code: 3\n\nstderr:\nerr", "", "err"),
    ("pass", "Exit code: 0", "", ""),
])
def test_shell_result_handles_single_or_empty_streams(monkeypatch, capfd, source, expected, out, err):
    monkeypatch.setattr(chatrepl, "build_live_shell_command", lambda command: [sys.executable, "-c", source])
    monkeypatch.setattr(chatrepl, "SYS_STDOUT_BUFFER", io.BytesIO())
    assert chatrepl.tool_shell({"command": "test child"}) == expected
    captured_out, captured_err = capfd.readouterr()
    assert captured_out == out
    assert captured_err == err


@pytest.mark.parametrize("arguments", [None, [], "{}", {"path": 42}, {}])
def test_malformed_write_arguments_are_errors(arguments):
    result = chatrepl.run_tool("write", arguments)
    assert result.text.startswith(("Invalid arguments", "Error running"))
