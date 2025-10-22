#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2025 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
from __future__ import print_function, unicode_literals

import argparse
import codecs
import os.path
import sys
from code import InteractiveConsole
from pydoc import TextDoc

from chat_completions_conversation import AssistantMessage, ChatCompletionsConversation
from file_to_unicode_base64_data_uri import file_to_unicode_base64_data_uri
from get_unicode_multiline_input_with_editor import get_unicode_multiline_input_with_editor
from textcompat import (
    filesystem_str_to_text,
    stdin_str_to_text,
    text_to_filesystem_str,
    text_to_stdout_str,
    text_to_utf_8_str,
)
from typing import Text


def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '-k', '--api-key',
        type=str,
        required=True,
        help='API key'
    )

    parser.add_argument(
        '-u', '--base-url',
        type=str,
        required=True,
        help='Base URL'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Model name'
    )

    parser.add_argument(
        '-l', '--load',
        metavar='JSON_FILE_PATH',
        type=str,
        required=False,
        help='Load a conversation from JSON_FILE_PATH'
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize conversation
    conversation = ChatCompletionsConversation(
        api_key=filesystem_str_to_text(args.api_key),
        base_url=filesystem_str_to_text(args.base_url),
        model=filesystem_str_to_text(args.model)
    )

    if args.load:
        conversation.load_from_json_file(filesystem_str_to_text(args.load))

    # Use streaming?
    stream = True

    # Are we under interactive mode?
    is_interactive = sys.stdin.isatty()

    # Try to import readline under interactive mode
    if is_interactive:
        try:
            import readline
        except ImportError:
            readline = None

            print(
                'Failed to import `readline`. This will affect the command-line interface functionality:\n',
                file=sys.stderr
            )

            print(
                '- Line editing features (arrow keys, cursor movement) will be disabled',
                file=sys.stderr
            )

            print(
                '- Command history (up/down keys) will not be available',
                file=sys.stderr
            )

            print(
                '\nWhile the program will still run, the text input will be basic and limited.',
                file=sys.stderr
            )

            print(
                '\nYou can install readline with `pip install pyreadline`.\n',
                file=sys.stderr
            )
    else:
        readline = None

    # Interface functions
    def send(
            text=''  # type: Text
    ):
        """Send a message and stream the response"""
        if stream:
            for chunk in conversation.send_and_stream_response(text):
                print(chunk, end='')
            print()
        else:
            print(conversation.send_and_receive_response(text))

    def append(
            text  # type: Text
    ):
        """Append text to conversation (don't send)"""
        conversation.append_user_message(text)

    def multiline():
        """Append multiline input via your editor"""
        multiline_input = u''.join(
            get_unicode_multiline_input_with_editor(
                [],
                None
            )
        )
        conversation.append_user_message(multiline_input)

    def img(
            img_file_path  # type: Text
    ):
        """Append an image file"""
        image_url = file_to_unicode_base64_data_uri(text_to_filesystem_str(img_file_path))
        conversation.append_user_message('', image_url)

    def txt(
            txt_file_path  # type: Text
    ):
        """Append a text file"""
        with codecs.open(
                text_to_filesystem_str(txt_file_path),
                mode=text_to_utf_8_str('r'),
                encoding=text_to_utf_8_str('utf-8')
        ) as txt_file:
            conversation.append_user_message(txt_file.read())

    def load(
            json_file_path  # type: Text
    ):
        """Load conversation from JSON"""
        conversation.load_from_json_file(json_file_path)

    def save(
            json_file_path  # type: Text
    ):
        """Save conversation to JSON"""
        conversation.save_to_json_file(json_file_path)

    def export(
            md_file_path  # type: Text
    ):
        """Export conversation to Markdown"""
        with codecs.open(
                text_to_filesystem_str(md_file_path),
                mode=text_to_utf_8_str('w'),
                encoding=text_to_utf_8_str('utf-8')
        ) as md_file:
            md_file.write(conversation.export_to_text())

    def correct():
        """Correct (edit) last model response"""
        messages = conversation.messages
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AssistantMessage):
                corrected = u''.join(
                    get_unicode_multiline_input_with_editor(
                        [last_message.text],
                        None
                    )
                )
                conversation.correct_last_response(corrected)
                return True
        return False

    # Non-interactive mode
    if not is_interactive:
        send(stdin_str_to_text(sys.stdin.read()))
    # Interactive mode
    else:
        # Read readline history file
        if readline is not None:
            chatrepl_history_file_path = os.path.join(os.path.expanduser('~'), '.chatrepl_history')
            try:
                readline.read_history_file(chatrepl_history_file_path)
            except Exception:
                pass
        else:
            chatrepl_history_file_path = None

        # Initialize interactive environment
        all_functions = [send, correct, append, multiline, img, txt, load, save, export]

        namespace = {
            function.__name__: function
            for function in all_functions
        }

        banner_lines = [
            'Welcome to ChatREPL! Use one of the following commands to interact with %s:\n' % conversation.model
        ]

        text_doc = TextDoc()
        for function in all_functions:
            banner_lines += [
                text_doc.document(function),
            ]

        interactive_console = InteractiveConsole(namespace)

        interactive_console.runsource(
            text_to_utf_8_str('from __future__ import print_function, unicode_literals')
        )

        interactive_console.interact(text_to_stdout_str('\n'.join(banner_lines)))

        # Write readline history file before exiting
        if readline is not None and chatrepl_history_file_path is not None:
            readline.write_history_file(chatrepl_history_file_path)


if __name__ == '__main__':
    main()
