#!/usr/bin/env python3

import argparse
import json
import os
import readline
import sys
from typing import List, Optional, Union, BinaryIO, Iterable

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

# Type aliases
Message = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
]


# Setup unbuffered output streams
STDOUT = os.fdopen(sys.stdout.fileno(), 'wb+', 0)
STDERR = os.fdopen(sys.stderr.fileno(), 'wb+', 0)


def fputs(string: Union[str, bytes], stream: BinaryIO) -> None:
    """Write string to a binary stream, encoding if necessary."""
    if isinstance(string, str):
        string = string.encode()
    stream.write(string)
    stream.flush()


def perror(e: Exception) -> None:
    fputs(f'{type(e).__name__}: {e}\n', STDERR)


def save_messages_to_file(messages: List[Message], filename: str) -> None:
    """Save chat messages to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        perror(e)


def load_messages_from_file(filename: str) -> Optional[List[Message]]:
    """Load chat messages from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                if all(isinstance(message, dict) and isinstance(message.get("role", None), str) and isinstance(message.get("content", None), str) for message in loaded):
                    return loaded
            raise ValueError(
                "Invalid JSON schema: Expected a list of dictionaries with keys 'role' (string) and 'content' (string). "
                f"Got: {loaded}"
            )
    except Exception as e:
        perror(e)
        return None


def print_messages(messages: List[Message]) -> None:
    """Print all messages in the conversation."""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        fputs(f'{role}: {content}\n', STDOUT)


def read_file_content(filename: str) -> Optional[str]:
    """Read text content from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        perror(e)
        return None


def chat_with_model(client: OpenAI, messages: List[Message], model: str) -> Iterable[str]:
    """Send messages to the model and yield streamed response chunks."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        full_response = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            yield content
            full_response.append(content)

        assistant_message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": ''.join(full_response)
        }
        messages.append(assistant_message)

    except Exception as e:
        perror(e)


def get_single_input(prompt: str = '> ') -> str:
    """Get a single-line input from the user."""
    # Do NOT use input().
    # When the user enters something and presses down BACKSPACE, the prompt is removed as well.
    return input(prompt)


def get_multiline_input() -> str:
    """Get multiline input from the user until EOF (Ctrl-D)."""
    fputs('Enter Ctrl-D on a blank line to finish input:\n', STDOUT)
    lines = []
    try:
        while True:
            line = get_single_input('> ')
            lines.append(line)
    except EOFError:
        pass
    return '\n'.join(lines)


def display_help() -> None:
    fputs('Enter a message to send to the model or use one of the following commands:\n', STDOUT)
    fputs(':save <filename>  Save the conversation to <filename>\n', STDOUT)
    fputs(':load <filename>  Load a conversation from <filename>\n', STDOUT)
    fputs(':send <filename>  Send the contents of <filename>\n', STDOUT)
    fputs(':multiline        Enter multiline input\n', STDOUT)
    fputs(':help             Display help\n', STDOUT)
    fputs(':quit             Exit the program\n', STDOUT)


def main() -> None:
    """Entry point for the chat interface."""
    parser = argparse.ArgumentParser(description="OpenAI Conversation API Terminal Chat")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI API base URL")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model = args.model

    histfile = os.path.join(os.path.expanduser("~"), ".chat_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass

    messages: List[Message] = []

    fputs(f'Welcome to Terminal Chat (Model: {model})\n', STDOUT)
    display_help()

    while True:
        try:
            user_input = get_single_input('\nUser: ').strip()

            if user_input.startswith(':'):
                cmd, *args = user_input.lower().split()
                if cmd == ":save" and len(args) == 1:
                    filename = args[0]
                    save_messages_to_file(messages, filename)
                    fputs(f"Conversation saved to {filename}\n", STDOUT)
                    continue
                elif cmd == ":load" and len(args) == 1:
                    filename = args[0]
                    loaded = load_messages_from_file(filename)
                    if loaded is not None:
                        messages.clear()
                        messages.extend(loaded)
                        fputs(f"Loaded conversation from {filename}\n", STDOUT)
                        print_messages(messages)
                    continue
                elif cmd == ":send" and len(args) == 1:
                    filename = args[0]
                    file_content = read_file_content(filename)
                    if file_content is not None:
                        user_input = file_content
                    else:
                        continue
                elif cmd == ":multiline" and not args:
                    user_input = get_multiline_input()
                elif cmd == ":help" and not args:
                    display_help()
                    continue
                elif cmd == ":quit" and not args:
                    break
                else:
                    fputs("Unknown command.\n", STDOUT)
                    display_help()
                    continue

            if not user_input:
                continue

            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": user_input
            }
            messages.append(user_message)

            fputs('\nAssistant: ', STDOUT)
            for chunk in chat_with_model(client, messages, model):
                fputs(chunk, STDOUT)
            fputs('\n', STDOUT)

        except (KeyboardInterrupt, EOFError):
            break

    readline.write_history_file(histfile)


if __name__ == "__main__":
    main()
