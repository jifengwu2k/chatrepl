# -*- coding: utf-8 -*-
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tests for AGENTS.md / CLAUDE.md discovery and system prompt composition."""

import os

from chatrepl import SYSTEM_PROMPT, compose_system_prompt


def test_compose_system_prompt_disabled():
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, False)
    assert prompt == SYSTEM_PROMPT
    assert paths == []


def test_compose_system_prompt_enabled_without_files(tmpdir, monkeypatch):
    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)
    assert prompt == SYSTEM_PROMPT
    assert paths == []


def test_compose_system_prompt_includes_agents_md(tmpdir, monkeypatch):
    context_path = os.path.join(str(tmpdir), "AGENTS.md")
    with open(context_path, "wb") as handle:
        handle.write(b"Follow the project rules.")

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [context_path]
    assert "AGENTS.md" in prompt
    assert "Follow the project rules." in prompt
    assert prompt.startswith(SYSTEM_PROMPT)


def test_compose_system_prompt_includes_claude_md(tmpdir, monkeypatch):
    context_path = os.path.join(str(tmpdir), "CLAUDE.md")
    with open(context_path, "wb") as handle:
        handle.write(b"Claude instructions.")

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [context_path]
    assert "Claude instructions." in prompt


def test_compose_system_prompt_includes_both_files_in_listing_order(tmpdir, monkeypatch):
    agents_path = os.path.join(str(tmpdir), "AGENTS.md")
    claude_path = os.path.join(str(tmpdir), "CLAUDE.md")
    for path in (agents_path, claude_path):
        with open(path, "wb") as handle:
            handle.write(b"rules\n")

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert sorted(paths) == sorted([agents_path, claude_path])


def test_compose_system_prompt_unreadable_context_file(tmpdir, monkeypatch):
    # A directory named AGENTS.md cannot be read as a text file.
    context_path = os.path.join(str(tmpdir), "AGENTS.md")
    os.makedirs(context_path)

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [context_path]
    assert "Failed to read context file" in prompt
