#!/usr/bin/env python3

import argparse
import json
import os
import readline
import sys
from typing import List, Optional, Union, BinaryIO, Iterable

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


def main() -> None:
    """Entry point for the chat interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("conversation", type=str, help="Conversation (JSON file) to print")
    args = parser.parse_args()

    messages = load_messages_from_file(args.conversation)
    if messages is not None:
        print_messages(messages)


if __name__ == '__main__':
    main()
