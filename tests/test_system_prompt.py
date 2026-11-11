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


def write_context_file(directory, name, content):
    path = os.path.join(str(directory), name)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


def test_compose_system_prompt_includes_both_files_in_listing_order(tmpdir, monkeypatch):
    agents_path = write_context_file(tmpdir, "AGENTS.md", b"agents rules\n")
    write_context_file(tmpdir, "CLAUDE.md", b"claude rules\n")

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    # One context file per directory: AGENTS.md shadows CLAUDE.md.
    assert paths == [agents_path]
    assert "agents rules" in prompt
    assert "claude rules" not in prompt


def test_compose_system_prompt_prefers_override_file(tmpdir, monkeypatch):
    override_path = write_context_file(tmpdir, "AGENTS.override.md", b"override rules\n")
    write_context_file(tmpdir, "AGENTS.md", b"agents rules\n")

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [override_path]
    assert "override rules" in prompt
    assert "agents rules" not in prompt


def test_compose_system_prompt_includes_parent_directories(tmpdir, monkeypatch):
    parent_path = write_context_file(tmpdir, "AGENTS.md", b"parent rules\n")
    child_directory = tmpdir.mkdir("project")

    monkeypatch.chdir(str(child_directory))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [parent_path]
    assert "parent rules" in prompt


def test_compose_system_prompt_orders_outermost_first(tmpdir, monkeypatch):
    outer_path = write_context_file(tmpdir, "AGENTS.md", b"outer rules\n")
    child_directory = tmpdir.mkdir("project")
    inner_path = write_context_file(child_directory, "CLAUDE.md", b"inner rules\n")
    deep_directory = child_directory.mkdir("src")

    monkeypatch.chdir(str(deep_directory))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == [outer_path, inner_path]
    assert prompt.index("outer rules") < prompt.index("inner rules")


def test_compose_system_prompt_ignores_context_directories(tmpdir, monkeypatch):
    # A directory named AGENTS.md is not a context file.
    os.makedirs(os.path.join(str(tmpdir), "AGENTS.md"))

    monkeypatch.chdir(str(tmpdir))
    prompt, paths = compose_system_prompt(SYSTEM_PROMPT, True)

    assert paths == []
    assert prompt == SYSTEM_PROMPT
