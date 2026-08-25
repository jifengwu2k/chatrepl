# -*- coding: utf-8 -*-
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tests for the REPL save(path) / load(path) helpers."""

import json
import os

from chat_completions_conversation_with_tools import ChatCompletionsConversationWithTools

from chatrepl import SYSTEM_PROMPT, build_namespace


def build_conversation():
    # type: () -> ChatCompletionsConversationWithTools
    return ChatCompletionsConversationWithTools(
        api_key="test-key",
        base_url="http://example.invalid/v1",
        model="test-model",
        system_prompt=SYSTEM_PROMPT,
        tools_by_name={},
    )


def build_namespace_for(conversation):
    # type: (ChatCompletionsConversationWithTools) -> dict
    return build_namespace(
        conversation,
        True,
        SYSTEM_PROMPT,
        False,
        [],
    )


def test_save_writes_transcript_including_system_message(tmpdir):
    conversation = build_conversation()
    conversation.append_user_message("hello")
    namespace = build_namespace_for(conversation)

    path = os.path.join(str(tmpdir), "conv.json")
    namespace["save"](path)

    with open(path, "rb") as handle:
        saved = json.loads(handle.read().decode("utf-8"))

    assert saved[0]["role"] == "system"
    assert saved[1] == {"role": "user", "content": "hello"}


def test_load_appends_saved_transcript(tmpdir):
    source = build_conversation()
    source.append_user_message("hello")
    source.append_assistant_message("hi there")
    namespace = build_namespace_for(source)

    path = os.path.join(str(tmpdir), "conv.json")
    namespace["save"](path)

    target = build_conversation()
    target.append_user_message("existing")
    build_namespace_for(target)["load"](path)

    roles = [message["role"] for message in target.messages]
    assert roles == ["system", "user", "user", "assistant"]
    assert target.messages[-1] == {"role": "assistant", "content": "hi there"}


def test_load_skips_system_message(tmpdir):
    source = build_conversation()
    source.append_user_message("hello")
    namespace = build_namespace_for(source)

    path = os.path.join(str(tmpdir), "conv.json")
    namespace["save"](path)

    target = build_conversation()
    build_namespace_for(target)["load"](path)

    assert [message["role"] for message in target.messages] == ["system", "user"]


def test_load_missing_file_reports_error(tmpdir, capsys):
    conversation = build_conversation()
    namespace = build_namespace_for(conversation)

    path = os.path.join(str(tmpdir), "missing.json")
    namespace["load"](path)

    captured = capsys.readouterr()
    assert "Error loading" in captured.err


def test_load_invalid_json_reports_error(tmpdir, capsys):
    conversation = build_conversation()
    namespace = build_namespace_for(conversation)

    path = os.path.join(str(tmpdir), "bad.json")
    with open(path, "wb") as handle:
        handle.write(b"not json")

    namespace["load"](path)

    captured = capsys.readouterr()
    assert "Error loading" in captured.err


def test_save_and_load_helpers_are_documented_in_banner():
    conversation = build_conversation()
    namespace = build_namespace_for(conversation)
    assert "Save the current conversation transcript" in namespace["help_text"]
    assert "Load a saved JSON transcript" in namespace["help_text"]
