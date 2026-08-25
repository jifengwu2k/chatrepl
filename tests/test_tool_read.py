# -*- coding: utf-8 -*-
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tests for the read tool's offset/limit line selection logic."""

import os

from chatrepl import tool_read


def write_bytes(directory, name, content):
    path = os.path.join(str(directory), name)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


def test_tool_read_full_file(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\nline2\nline3\n")
    assert tool_read({"path": path}) == "line1\nline2\nline3\n"


def test_tool_read_with_limit_returns_remaining_hint(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\nline2\nline3\n")
    result = tool_read({"path": path, "offset": 2, "limit": 1})
    assert result == "line2\n\n\n[1 more lines in file. Use offset=3 to continue.]"


def test_tool_read_offset_to_end_without_limit(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\nline2\nline3\n")
    result = tool_read({"path": path, "offset": 2})
    assert result == "line2\nline3\n"


def test_tool_read_offset_beyond_eof(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\nline2\n")
    result = tool_read({"path": path, "offset": 5})
    assert result == "Error: offset 5 is beyond end of file (2 lines total)"


def test_tool_read_invalid_offset(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\n")
    assert tool_read({"path": path, "offset": 0}) == "Error: offset must be >= 1"


def test_tool_read_invalid_limit(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\n")
    assert tool_read({"path": path, "limit": -1}) == "Error: limit must be >= 0"


def test_tool_read_missing_file():
    result = tool_read({"path": "/nonexistent/path/file.txt"})
    assert result.startswith("Error reading")


def test_tool_read_empty_file(tmpdir):
    path = write_bytes(tmpdir, "empty.txt", b"")
    assert tool_read({"path": path}) == ""


def test_tool_read_empty_file_offset_beyond(tmpdir):
    path = write_bytes(tmpdir, "empty.txt", b"")
    result = tool_read({"path": path, "offset": 2})
    assert result == "Error: offset 2 is beyond end of file (0 lines total)"


def test_tool_read_unicode(tmpdir):
    path = write_bytes(tmpdir, "uni.txt", u"\u4f60\u597d\n".encode("utf-8"))
    assert tool_read({"path": path}) == u"\u4f60\u597d\n"


def test_tool_read_invalid_utf8_is_replaced(tmpdir):
    path = write_bytes(tmpdir, "bad.txt", b"ok\xff\n")
    result = tool_read({"path": path})
    assert u"\ufffd" in result


def test_tool_read_uses_offset_default_of_one(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"only\n")
    result = tool_read({"path": path, "offset": 1, "limit": 10})
    assert result == "only\n"


def test_tool_read_zero_limit_returns_hint(tmpdir):
    path = write_bytes(tmpdir, "hello.txt", b"line1\nline2\nline3\n")
    result = tool_read({"path": path, "offset": 1, "limit": 0})
    assert "more lines in file" in result
