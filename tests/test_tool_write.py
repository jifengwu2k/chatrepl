# -*- coding: utf-8 -*-
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tests for the write tool and parent-directory creation."""

import os

from chatrepl import ensure_parent_dir, run_tool, tool_write


def test_run_tool_rejects_non_object_arguments():
    # A model can return valid JSON such as an array for a tool call.  This
    # must be reported to the model rather than crashing on arguments["path"].
    result = run_tool("write", [{"path": "out.txt", "content": "hello"}])
    assert result == "Invalid arguments for write: expected a JSON object, got list"


def test_tool_write_creates_file_with_parent_directories(tmpdir):
    path = os.path.join(str(tmpdir), "out", "nested", "file.txt")
    result = tool_write({"path": path, "content": "hello"})

    assert result == "Successfully wrote 5 characters to %s" % path
    with open(path, "rb") as handle:
        assert handle.read() == b"hello"


def test_tool_write_overwrites_existing_file(tmpdir):
    path = os.path.join(str(tmpdir), "file.txt")
    tool_write({"path": path, "content": "first"})
    result = tool_write({"path": path, "content": "second"})

    assert result.startswith("Successfully wrote 6 characters")
    with open(path, "rb") as handle:
        assert handle.read() == b"second"


def test_tool_write_unicode_content(tmpdir):
    path = os.path.join(str(tmpdir), "uni.txt")
    result = tool_write({"path": path, "content": u"\u4f60\u597d"})

    assert result.startswith("Successfully wrote 2 characters")
    with open(path, "rb") as handle:
        assert handle.read() == u"\u4f60\u597d".encode("utf-8")


def test_tool_write_error_when_path_is_directory(tmpdir):
    directory = os.path.join(str(tmpdir), "dir")
    os.makedirs(directory)
    result = tool_write({"path": directory, "content": "x"})
    assert result.startswith("Error writing")


def test_ensure_parent_dir_creates_nested_directories(tmpdir):
    target = os.path.join(str(tmpdir), "a", "b", "c", "file.txt")
    ensure_parent_dir(target)
    assert os.path.isdir(os.path.join(str(tmpdir), "a", "b", "c"))


def test_ensure_parent_dir_skips_existing_directories(tmpdir):
    existing = os.path.join(str(tmpdir), "a", "b")
    os.makedirs(existing)
    ensure_parent_dir(os.path.join(existing, "file.txt"))
    assert os.path.isdir(existing)


def test_ensure_parent_dir_top_level_file(tmpdir):
    ensure_parent_dir(os.path.join(str(tmpdir), "file.txt"))
    assert os.path.isdir(str(tmpdir))
