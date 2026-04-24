#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
from __future__ import print_function, unicode_literals

import argparse
import codecs
import difflib
import json
import os
import sys
from code import InteractiveConsole
from enum import Enum
from pydoc import TextDoc
from typing import IO, List, Text

from chat_completions_conversation_with_tools import (
    ChatCompletionsConversationWithTools,
    Tool,
)
from create_inspect_typeddict import create_typeddict
from file_to_unicode_base64_data_uri import file_to_unicode_base64_data_uri
from get_unicode_multiline_input_with_editor import (
    get_unicode_multiline_input_with_editor,
)
from get_unicode_shell import get_unicode_shell
from live_tee_and_capture import live_tee_and_capture
from posix_or_nt import posix_or_nt
from textcompat import (
    get_stdout_encoding,
    stdin_str_to_text,
    text_to_filesystem_str,
    text_to_stdout_str,
)

POSIX_OR_NT = posix_or_nt()

if sys.version_info < (3,):
    SYS_STDOUT_BUFFER = sys.stdout
else:
    SYS_STDOUT_BUFFER = sys.stdout.buffer


def fputs(file, text):
    # type: (IO, Text) -> None
    encoded = text.encode(get_stdout_encoding(), errors="ignore")
    file.write(encoded)
    try:
        file.flush()
    except Exception:
        pass


CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


class RunAgentTurnState(Enum):
    SEND_STREAMING_RESPONSE = "send_streaming_response"
    SEND_NON_STREAMING_RESPONSE = "send_non_streaming_response"
    HANDLE_ASSISTANT_MESSAGE = "handle_assistant_message"


SYSTEM_PROMPT = (
    """You are an agent with four tools: read, write, edit, shell (%s).

Tool rules:
- Use read for inspecting files.
- If a file is large, use read with offset/limit.
- Use write for creating new files or completely rewriting files.
- Use edit for precise changes to existing files.
- For edit, every edits[].oldText must match exactly once in the original file.
- All edits in one call are matched against the original file, not incrementally.
- Do not create overlapping edits.
- Use shell for shell commands.
- Prefer short, safe shell commands.
""" % get_unicode_shell()
)

ReadParameters = create_typeddict(
    "ReadParameters",
    {
        "path": Text,
        "offset": int,
        "limit": int,
    },
    optional_keys=["offset", "limit"],
)

WriteParameters = create_typeddict(
    "WriteParameters",
    {
        "path": Text,
        "content": Text,
    },
)

EditReplacement = create_typeddict(
    "EditReplacement",
    {
        "oldText": Text,
        "newText": Text,
    },
)

EditParameters = create_typeddict(
    "EditParameters",
    {
        "path": Text,
        "edits": List[EditReplacement],
    },
)

ShellParameters = create_typeddict(
    "ShellParameters",
    {
        "command": Text,
    },
)


TOOLS_BY_NAME = {
    "read": Tool(
        description=(
            "Read the contents of a file. Supports text files. "
            "Optional offset/limit can be used to read part of a file."
        ),
        parameters_typeddict=ReadParameters,
    ),
    "write": Tool(
        description=(
            "Write content to a file. Creates the file if it does not exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters_typeddict=WriteParameters,
    ),
    "edit": Tool(
        description=(
            "Edit a single file using exact text replacement. "
            "Each edits[].oldText must match a unique, non-overlapping region of the original file."
        ),
        parameters_typeddict=EditParameters,
    ),
    "shell": Tool(
        description=(
            "Execute a shell command in the current working directory. "
            "Returns stdout and stderr combined."
        ),
        parameters_typeddict=ShellParameters,
    ),
}


def write_text_file(path, content):
    # type: (Text, Text) -> None
    with codecs.open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def read_text_file(path):
    # type: (Text) -> Text
    with codecs.open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def compose_system_prompt(base_system_prompt, context_files_enabled):
    # type: (Text, bool) -> tuple
    if not context_files_enabled:
        return base_system_prompt, []
        
    context_file_paths = []
    file_names = os.listdir(os.getcwd())
    for file_name in file_names:
        if file_name in CONTEXT_FILE_NAMES:
            context_file_paths.append(os.path.join(os.getcwd(), file_name))

    if not context_file_paths:
        return base_system_prompt, []

    sections = [
        base_system_prompt,
        "",
        "The following project instruction files were discovered under the current working directory.",
        "Follow them in addition to the base system prompt.",
        "",
    ]
    for path in context_file_paths:
        try:
            content = read_text_file(path)
        except Exception as exc:
            content = "[Failed to read context file: %s]" % exc
        sections.extend(
            [
                "--- %s ---" % path,
                content,
                "",
            ]
        )

    return "\n".join(sections), context_file_paths


def tool_read(arguments):
    # type: (dict) -> Text
    path = arguments["path"]

    if "offset" in arguments:
        offset = arguments["offset"] or 1
    else:
        offset = 1

    if "limit" in arguments:
        limit = arguments["limit"]
    else:
        limit = None

    if offset < 1:
        return "Error: offset must be >= 1"

    if limit is not None and limit < 0:
        return "Error: limit must be >= 0"

    selected = []
    selected_count = 0
    total_lines = 0

    try:
        with codecs.open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                total_lines = line_number
                if line_number >= offset and (limit is None or selected_count < limit):
                    selected.append(line)
                    selected_count += 1
    except Exception as exc:
        return "Error reading %s: %s" % (path, exc)

    if total_lines == 0:
        if offset > 1:
            return "Error: offset %s is beyond end of file (0 lines total)" % offset
        return ""

    if offset > total_lines:
        return "Error: offset %s is beyond end of file (%s lines total)" % (
            offset,
            total_lines,
        )

    selected_text = "".join(selected)
    end_line = (offset - 1) + len(selected)

    if limit is not None and end_line < total_lines:
        return "%s\n\n[%s more lines in file. Use offset=%s to continue.]" % (
            selected_text,
            total_lines - end_line,
            end_line + 1,
        )

    return selected_text


def ensure_parent_dir(path):
    # type: (Text) -> None
    directory = os.path.dirname(path)
    if not directory or os.path.isdir(directory):
        return

    parent_directory = os.path.dirname(directory)
    if parent_directory and parent_directory != directory:
        ensure_parent_dir(directory)

    if not os.path.isdir(directory):
        os.mkdir(directory)


def tool_write(arguments):
    # type: (dict) -> Text
    path = arguments["path"]
    content = arguments["content"]
    try:
        ensure_parent_dir(path)
        write_text_file(path, content)
        return "Successfully wrote %s characters to %s" % (len(content), path)
    except Exception as exc:
        return "Error writing %s: %s" % (path, exc)


def find_all_occurrences(haystack, needle):
    # type: (Text, Text) -> list
    if needle == "":
        return []
    positions = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            break
        positions.append(index)
        start = index + 1
    return positions


def apply_exact_edits(original, edits):
    # type: (Text, list) -> Text
    spans = []
    edit_index = 0
    for edit in edits:
        old_text = edit["oldText"]
        new_text = edit["newText"]
        if old_text == "":
            raise ValueError("Edit %s: oldText must not be empty" % edit_index)
        positions = find_all_occurrences(original, old_text)
        if len(positions) == 0:
            raise ValueError("Edit %s: oldText not found" % edit_index)
        if len(positions) > 1:
            raise ValueError(
                "Edit %s: oldText matched %s times, must match exactly once"
                % (edit_index, len(positions))
            )
        start = positions[0]
        end = start + len(old_text)
        spans.append((start, end, new_text))
        edit_index += 1

    spans.sort(key=lambda item: item[0])
    index = 1
    while index < len(spans):
        previous_end = spans[index - 1][1]
        current_start = spans[index][0]
        if current_start < previous_end:
            raise ValueError("Edits overlap. Merge nearby changes into a single edit.")
        index += 1

    output = []
    cursor = 0
    for start, end, new_text in spans:
        output.append(original[cursor:start])
        output.append(new_text)
        cursor = end
    output.append(original[cursor:])
    return "".join(output)


def unified_diff(old_text, new_text, path):
    # type: (Text, Text, Text) -> Text
    diff_lines = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=path,
        tofile=path,
        lineterm="",
    )
    return "\n".join(diff_lines)


def tool_edit(arguments):
    # type: (dict) -> Text
    path = arguments["path"]
    edits = arguments["edits"]

    try:
        original = read_text_file(path)
    except Exception as exc:
        return "Error reading %s for edit: %s" % (path, exc)

    try:
        updated = apply_exact_edits(original, edits)
    except Exception as exc:
        return "Edit failed: %s" % exc

    if updated == original:
        return "No changes made."

    try:
        write_text_file(path, updated)
    except Exception as exc:
        return "Error writing edited file %s: %s" % (path, exc)

    diff = unified_diff(original, updated, path)
    return "Applied %s edit(s) to %s\n\n%s" % (len(edits), path, diff)


def build_live_shell_command(command):
    # type: (Text) -> list
    if POSIX_OR_NT == "nt":
        return ["cmd.exe", "/c", command]
    else:
        return [get_unicode_shell(), "-lc", command]


def tool_shell(arguments):
    # type: (dict) -> Text
    command = arguments["command"]

    fputs(SYS_STDOUT_BUFFER, "$ %s\n" % command)

    try:
        exit_code, stdout_buffer, stderr_buffer = live_tee_and_capture(
            build_live_shell_command(command), tee_stdout=True, tee_stderr=True
        )
        output = bytearray()
        output.extend(stdout_buffer)
        output.extend(stderr_buffer)
        text_output = output.decode("utf-8", "replace")

        result = ["Exit code: %s" % exit_code]
        if text_output:
            result.extend(["", text_output])
        return "\n".join(result)
    except Exception as exc:
        return "Error executing shell command: %s" % exc


def run_tool(name, arguments):
    # type: (Text, dict) -> Text
    if name == "read":
        return tool_read(arguments)
    if name == "write":
        return tool_write(arguments)
    if name == "edit":
        return tool_edit(arguments)
    if name == "shell":
        return tool_shell(arguments)
    return "Unknown tool: %s" % name


def refresh_conversation_system_prompt(
    conversation,
    base_system_prompt,
    context_files_enabled,
    update_messages=True,
):
    # type: (ChatCompletionsConversationWithTools, Text, bool, bool) -> list
    system_prompt, context_file_paths = compose_system_prompt(
        base_system_prompt,
        context_files_enabled,
    )
    conversation.set_system_prompt(system_prompt, update_messages=update_messages)
    return context_file_paths


def run_agent_turn(conversation, stream_enabled, text="", image_url=None):
    # type: (ChatCompletionsConversationWithTools, bool, Text, object) -> None
    pending_text = text
    pending_image_url = image_url
    assistant_message = None
    state = (
        RunAgentTurnState.SEND_STREAMING_RESPONSE
        if stream_enabled
        else RunAgentTurnState.SEND_NON_STREAMING_RESPONSE
    )

    while True:
        if state == RunAgentTurnState.SEND_STREAMING_RESPONSE:
            streamed_state = {
                "text_seen": False,
                "tool_header_printed": False,
            }

            def on_stream_text(text_piece):
                # type: (Text) -> None
                streamed_state["text_seen"] = True
                fputs(SYS_STDOUT_BUFFER, text_piece)

            def on_stream_tool_call_delta(tool_call_delta):
                # type: (dict) -> None
                if streamed_state["text_seen"]:
                    fputs(SYS_STDOUT_BUFFER, "\n")
                    streamed_state["text_seen"] = False
                if not streamed_state["tool_header_printed"]:
                    fputs(SYS_STDOUT_BUFFER, "[assistant is preparing tool call(s)]\n")
                    streamed_state["tool_header_printed"] = True
                fputs(SYS_STDOUT_BUFFER, "%s\n" % json.dumps(tool_call_delta))

            assistant_message = conversation.send_and_stream_response(
                pending_text,
                image_url=pending_image_url,
                on_content_delta=on_stream_text,
                on_tool_call_delta=on_stream_tool_call_delta,
            )
            if assistant_message.content or streamed_state["tool_header_printed"]:
                fputs(SYS_STDOUT_BUFFER, "\n")
            state = RunAgentTurnState.HANDLE_ASSISTANT_MESSAGE
        elif state == RunAgentTurnState.SEND_NON_STREAMING_RESPONSE:
            assistant_message = conversation.send_and_receive_response(
                pending_text,
                image_url=pending_image_url,
            )
            state = RunAgentTurnState.HANDLE_ASSISTANT_MESSAGE
        elif state == RunAgentTurnState.HANDLE_ASSISTANT_MESSAGE:
            pending_text = ""
            pending_image_url = None

            content = assistant_message.content
            tool_calls = assistant_message.tool_calls

            if not stream_enabled:
                fputs(SYS_STDOUT_BUFFER, content)
                fputs(SYS_STDOUT_BUFFER, "\n")

            if not tool_calls:
                return

            fputs(
                SYS_STDOUT_BUFFER,
                "\n[assistant is using %s tool(s)]\n" % len(tool_calls),
            )

            for call in tool_calls:
                fputs(SYS_STDOUT_BUFFER, "\n[tool %s]\n" % call.name)
                tool_result = run_tool(call.name, call.arguments)

                fputs(SYS_STDOUT_BUFFER, tool_result)
                fputs(SYS_STDOUT_BUFFER, "\n")
                conversation.append_tool_message(call.id, tool_result)
                fputs(SYS_STDOUT_BUFFER, "\n")

            state = (
                RunAgentTurnState.SEND_STREAMING_RESPONSE
                if stream_enabled
                else RunAgentTurnState.SEND_NON_STREAMING_RESPONSE
            )
        else:
            raise ValueError("Unknown run_agent_turn state: %s" % state)


def get_multiline_input(initial_lines=None):
    # type: (object) -> Text
    if initial_lines is None:
        initial_lines = []
    return "\n".join(get_unicode_multiline_input_with_editor(initial_lines, None))


def print_banner(conversation, functions, context_files_enabled, context_file_paths):
    # type: (ChatCompletionsConversationWithTools, list, bool, list) -> Text
    banner_lines = [
        "Welcome to chatrepl. Use one of the following commands to interact with %s:"
        % conversation.model,
        "",
    ]

    if context_files_enabled:
        if context_file_paths:
            banner_lines.append("Loaded context files:")
            for path in context_file_paths:
                banner_lines.append("- %s" % path)
        else:
            banner_lines.append("Loaded context files: none")
        banner_lines.append("")
    else:
        banner_lines.extend(["Context files disabled.", ""])

    text_doc = TextDoc()
    for function in functions:
        banner_lines.append(text_doc.document(function))
    return "\n".join(banner_lines)


def build_namespace(
    conversation,
    default_stream_enabled,
    base_system_prompt,
    context_files_enabled,
    context_file_paths,
):
    # type: (ChatCompletionsConversationWithTools, bool, Text, bool, list) -> dict
    def send(text="", image_path=None, stream=default_stream_enabled):
        # type: (Text, object, bool) -> None
        """Send a message and let the agent complete tool calls. Optionally include a local image path and control streaming."""
        image_url = None
        if image_path is not None:
            image_url = file_to_unicode_base64_data_uri(
                text_to_filesystem_str(Text(image_path))
            )
        run_agent_turn(
            conversation,
            bool(stream),
            text,
            image_url=image_url,
        )

    def append(text):
        # type: (Text) -> None
        """Append text to the conversation as a user message without sending it."""
        conversation.append_user_message(text)

    def multiline():
        # type: () -> None
        """Append multiline input via your editor without sending it."""
        conversation.append_user_message(get_multiline_input())

    def txt(txt_file_path):
        # type: (Text) -> None
        """Append a UTF-8 text file as a user message without sending it."""
        conversation.append_user_message(read_text_file(txt_file_path))

    def img(image_path):
        # type: (Text) -> None
        """Append a local image as a user message without sending it. Files are embedded as data URLs."""
        conversation.append_user_message(
            "",
            image_url=file_to_unicode_base64_data_uri(
                text_to_filesystem_str(Text(image_path))
            ),
        )

    def reset():
        # type: () -> None
        """Reset the conversation to the system prompt only."""
        refresh_conversation_system_prompt(
            conversation,
            base_system_prompt,
            context_files_enabled,
            update_messages=False,
        )
        conversation.reset()

    functions = [
        send,
        append,
        multiline,
        txt,
        img,
        reset,
    ]
    namespace = {}
    for function in functions:
        namespace[function.__name__] = function
    namespace["conversation"] = conversation
    namespace["TOOLS_BY_NAME"] = TOOLS_BY_NAME
    namespace["help_text"] = print_banner(
        conversation,
        functions,
        context_files_enabled,
        context_file_paths,
    )
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--api-key",
        required=True,
        help="API key for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "-u",
        "--base-url",
        required=True,
        help="Base API URL or full chat completions endpoint, e.g. http://localhost:11434/v1",
    )
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and use regular responses",
    )
    parser.add_argument(
        "--no-context-files",
        action="store_true",
        help="Disable AGENTS.md and CLAUDE.md discovery",
    )
    parser.add_argument("prompt", nargs="*", help="Optional initial prompt")
    args = parser.parse_args()

    api_key = stdin_str_to_text(args.api_key)
    base_url = stdin_str_to_text(args.base_url)
    model = stdin_str_to_text(args.model)
    no_stream = args.no_stream
    no_context_files = args.no_context_files
    prompt = map(lambda component: stdin_str_to_text(component), args.prompt)

    context_files_enabled = not no_context_files
    conversation = ChatCompletionsConversationWithTools(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools_by_name=TOOLS_BY_NAME,
    )
    context_file_paths = refresh_conversation_system_prompt(
        conversation,
        SYSTEM_PROMPT,
        context_files_enabled,
        update_messages=False,
    )
    conversation.reset()

    initial_prompt = " ".join(prompt).strip()

    default_stream_enabled = not no_stream

    if not sys.stdin.isatty() and not initial_prompt:
        run_agent_turn(
            conversation,
            default_stream_enabled,
            stdin_str_to_text(sys.stdin.read()),
        )

    if initial_prompt:
        run_agent_turn(conversation, default_stream_enabled, initial_prompt)

    is_interactive = sys.stdin.isatty()
    readline = None
    history_path = None

    # Try to import readline under interactive mode
    if is_interactive:
        try:
            import readline
        except ImportError:
            readline = None

            print(
                "Failed to import `readline`. This will affect the command-line interface functionality:\n",
                file=sys.stderr,
            )

            print(
                "- Line editing features (arrow keys, cursor movement) will be disabled",
                file=sys.stderr,
            )

            print(
                "- Command history (up/down keys) will not be available",
                file=sys.stderr,
            )

            print(
                "\nWhile the program will still run, the text input will be basic and limited.",
                file=sys.stderr,
            )

            print(
                "\nYou can install readline with `pip install pyreadline`.\n",
                file=sys.stderr,
            )
    else:
        readline = None

    if readline is not None:
        history_path = os.path.join(os.path.expanduser("~"), ".chatrepl_history")
        try:
            readline.read_history_file(history_path)
        except Exception:
            pass

    namespace = build_namespace(
        conversation,
        default_stream_enabled,
        SYSTEM_PROMPT,
        context_files_enabled,
        context_file_paths,
    )
    interactive_console = InteractiveConsole(namespace)
    interactive_console.runsource(
        text_to_stdout_str("from __future__ import print_function, unicode_literals")
    )
    interactive_console.interact(text_to_stdout_str(namespace["help_text"]))

    if readline is not None and history_path is not None:
        try:
            readline.write_history_file(history_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
