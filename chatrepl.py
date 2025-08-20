#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import codecs
import json
import os
import os.path
import sys

# Python version compatibility setup
if sys.version_info < (3,):
    from collections import Iterator
    from urllib2 import Request, urlopen, URLError

    def unicode_input(prompt):
        return raw_input(prompt).decode(sys.stdin.encoding)

    open_with_encoding = codecs.open

    def post_request_instance(url, data, headers):
        return Request(url, data=data, headers=headers)
else:
    from collections.abc import Iterator
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    unicode = str
    unicode_input = input
    open_with_encoding = open

    def post_request_instance(url, data, headers):
        return Request(url, data=data, headers=headers, method='POST')


def sanitize(text):
    return text.encode(sys.stdout.encoding, errors='ignore').decode(sys.stdout.encoding)


def fputs(text, stream):
    """Write text to a stream."""
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        fputs(sanitize(text), stream)


def perror(exception):
    fputs(u'%s: %s\n' % (type(exception).__name__, exception), sys.stderr)


class Conversation:
    __slots__ = (
        'api_key',
        'base_url',
        'model',
        'messages',
        'message_counter'
    )

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.messages = []
        self.message_counter = 0

    def save_messages_to_file(self, filename):
        # type: (str) -> None
        """Save chat messages to a JSON file."""
        with open_with_encoding(filename, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)

    def load_messages_from_file(self, filename):
        # type: (str) -> None
        """Load chat messages from a JSON file."""
        with open_with_encoding(filename, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list) and all(
                (
                    isinstance(message, dict)
                    and message.get(u'role', None) in (u'user', u'assistant')
                    and isinstance(message.get(u'content', None), unicode)
                )
                for message in loaded
            ):
                self.messages = loaded

                num_user_messages_encountered = 0
                for message in self.messages:
                    role = message[u'role']
                    if role == u'user':
                        num_user_messages_encountered += 1
                self.message_counter = num_user_messages_encountered
            else:
                raise ValueError(
                    u"Invalid JSON schema: Expected a list of dictionaries with keys 'role' ('user' or 'assistant') and 'content' (string). Got: %s" % loaded
                )

    def print_messages(self):
        # type: () -> None
        """Print messages."""
        num_user_messages_encountered = 0
        for message in self.messages:
            role = message[u'role']
            content = message[u'content']

            if role == 'user':
                num_user_messages_encountered += 1

            fputs(
                u'\n%s [%i]: %s\n' % (
                    role.capitalize(),
                    num_user_messages_encountered,
                    content
                ),
                sys.stdout
            )

    def add_message(
            self,
            text,
            image_url=u''
    ):
        # type: (unicode, unicode) -> None
        """Add a message to the model's message list without obtaining a response."""
        if image_url:
            message = {
                u'role': u'user',
                u'content': [
                    { u'type': u'text', u'text': text },
                    { u'type': u'image_url', u'image_url': { u'url': image_url } }
                ]
            }
        else:
            message = {
                u'role': u'user',
                u'content': text
            }

        self.messages.append(message)

    def correct_last_response(
            self,
            corrected_response
    ):
        # type: (unicode) -> bool
        """Correct the model's last response. Returns True upon success and False upon failure."""
        if self.messages:
            last_message = self.messages[-1]
            if last_message[u'role'] == u'assistant':
                last_message[u'content'] = corrected_response
                return True
        return False

    def send_message_to_model_and_stream_response(
            self,
            text,
            image_url=u''
    ):
        # type: (unicode, unicode) -> Iterator[unicode]
        """Send a message to the model using standard library HTTP requests.
        If an exception occurs, the message is guaranteed to be removed from the conversation."""
        num_messages = len(self.messages)

        if image_url:
            message = {
                u'role': u'user',
                u'content': [
                    { u'type': u'text', u'text': text },
                    { u'type': u'image_url', u'image_url': { u'url': image_url } }
                ]
            }
        else:
            message = {
                u'role': u'user',
                u'content': text
            }

        self.messages.append(message)

        exception_occurred = False
        try:
            url = u'%s/chat/completions' % self.base_url
            headers = {
                u'Content-Type': u'application/json',
                u'Authorization': u'Bearer %s' % self.api_key
            }
            data = json.dumps({
                u'model': self.model,
                u'messages': self.messages,
                u'stream': True,
            }).encode('utf-8')

            req = post_request_instance(url, data=data, headers=headers)
            response = urlopen(req)

            contents = []
            for line in response:
                if line.startswith(b'data: '):
                    chunk_data = line[6:].decode('utf-8').strip()  # Remove "data: " prefix
                    if chunk_data == u'[DONE]':
                        break

                    chunk = json.loads(chunk_data)
                    content = chunk.get(u'choices', [{}])[0].get(u'delta', {}).get(u'content', u'') or u''
                    contents.append(content)
                    yield content

            self.messages.append(
                {
                    u'role': u'assistant',
                    u'content': u''.join(contents)
                }
            )

            self.message_counter += 1

        except Exception:
            exception_occurred = True
            raise
        finally:
            if exception_occurred:
                self.messages.pop()


# Executed when run as a module
if __name__ == "__main__":
    # Are we under interactive mode?
    IS_INTERACTIVE = sys.stdin.isatty()

    # Try to import readline under interactive mode
    if IS_INTERACTIVE:
        try:
            import readline
        except ImportError as ie:
            readline = None
            perror(ie)
            fputs(u'\nFailed to import `readline`. This will affect the command-line interface functionality:\n', sys.stderr)
            fputs(u' - Line editing features (arrow keys, cursor movement) will be disabled\n', sys.stderr)
            fputs(u' - Command history (up/down keys) will not be available\n', sys.stderr)
            fputs(u'\nWhile the program will still run, the text input will be basic and limited.\n', sys.stderr)
            fputs(u'\nYou can install readline with `pip install pyreadline`.\n', sys.stderr)
    else:
        readline = None


    def read_file_content(filename):
        # type: (str) -> unicode
        """Read text_type content from a file."""
        with open_with_encoding(filename, 'r', encoding='utf-8') as f:
            return f.read()


    def get_single_line_input(prompt=u'> '):
        # type: (unicode) -> unicode
        """Get a single-line input from the user."""
        # Do NOT use unicode_input().
        # When the user enters something and presses down BACKSPACE, the prompt is removed as well.
        return unicode_input(prompt)


    def get_multi_line_input():
        # type: () -> unicode
        """Get multi-line input from the user until EOF."""
        fputs(u'Enter EOF on a blank line to finish input:\n', sys.stdout)
        lines = []
        try:
            while True:
                line = get_single_line_input(u'> ')
                lines.append(line)
        except EOFError:
            pass

        return u'\n'.join(lines)


    def single_interaction(conversation):
        # type: (Conversation) -> None
        content = None

        while True:
            user_input = get_single_line_input(
                u'\nUser [%d]: ' % (conversation.message_counter + 1)
            ).strip()

            if user_input.startswith(u':'):
                tokens = user_input.split()
                cmd = tokens[0] if tokens else u''
                args = tokens[1:]

                # :multiline
                if cmd == u':multiline' and not args:
                    content = get_multi_line_input()
                # :send <textfile>
                elif cmd == u':send' and len(args) == 1:
                    try:
                        content = read_file_content(args[0])
                    except Exception as exp:
                        perror(exp)
                        continue
                # :load <jsonfile>
                elif cmd == u':load' and len(args) == 1:
                    try:
                        conversation.load_messages_from_file(args[0])
                        fputs(u'Loaded conversation from %s:\n' % args[0], sys.stdout)
                        conversation.print_messages()
                        continue
                    except Exception as exp:
                        perror(exp)
                        continue
                # :save <jsonfile>
                elif cmd == u':save' and len(args) == 1:
                    try:
                        conversation.save_messages_to_file(args[0])
                        fputs(u'Conversation saved to %s\n' % args[0], sys.stdout)
                    except Exception as exp:
                        perror(exp)
                    continue
                # :help
                elif cmd == u':help' and not args:
                    display_help()
                    continue
                # :quit
                elif cmd == u':quit' and not args:
                    # Same as pressing Ctrl-D (sending EOF)
                    raise EOFError
                else:
                    fputs(u'Unknown command.\n', sys.stdout)
                    display_help()
                    continue
            else:
                content = user_input

            # We do not allow content to be empty
            if content:
                break

        if content is not None:
            try:
                fputs(u'\nAssistant [%d]: ' % (conversation.message_counter + 1), sys.stdout)
                for response in conversation.send_message_to_model_and_stream_response(content):
                    fputs(response, sys.stdout)
                fputs(u'\n', sys.stdout)
            except URLError as ue:
                perror(ue)


    def display_help():
        fputs(u'\nEnter a message to send to the model or use one of the following commands:\n', sys.stdout)
        fputs(u':multiline        Enter multiline input\n', sys.stdout)
        fputs(u':send TEXTFILE    Send the contents of TEXTFILE\n', sys.stdout)
        fputs(u':load JSONFILE    Load a conversation from JSONFILE\n', sys.stdout)
        fputs(u':save JSONFILE    Save the conversation to JSONFILE\n', sys.stdout)
        fputs(u':help             Display help\n', sys.stdout)
        fputs(u':quit (or Ctrl-D) Exit the program\n', sys.stdout)


    def main():
        """Entry point for the chat interface."""
        # Create the parser
        parser = argparse.ArgumentParser()

        # Add arguments
        parser.add_argument('--api-key', type=str, required=False, help='API key')
        parser.add_argument('--base-url', type=str, required=False, help='Base URL')
        parser.add_argument('--model', type=str, required=False, help='Model name')
        parser.add_argument('--load', metavar='JSONFILE', type=str, required=False, help='Load a conversation from JSONFILE')
        parser.add_argument('--print', metavar='JSONFILE', type=str, required=False, help='Print a saved conversation from JSONFILE and exit')

        # Parse arguments
        args = parser.parse_args()

        # Print a saved conversation from JSONFILE and exit
        if args.print:
            conversation = Conversation(args.api_key, args.base_url, args.model)
            conversation.load_messages_from_file(args.print)
            conversation.print_messages()
        else:
            if (
                not args.api_key
                or not args.base_url
                or not args.model
            ):
                parser.error('Must provide --api-key, --base-url, and --model when not printing a conversation using --print')

            conversation = Conversation(args.api_key, args.base_url, args.model)

            # Initialize messages
            if args.load:
                conversation.load_messages_from_file(args.load)

            # Non-interactive mode
            if not IS_INTERACTIVE:
                text_message = sys.stdin.read()

                if text_message:
                    for response in conversation.send_message_to_model_and_stream_response(
                        text_message
                    ):
                        fputs(response, sys.stdout)
                    fputs(u'\n', sys.stdout)
            # Interactive mode
            else:
                # Read readline history file
                if readline is not None:
                    histfile = os.path.join(os.path.expanduser("~"), ".chat_history")
                    try:
                        readline.read_history_file(histfile)
                    except Exception:
                        pass
                else:
                    histfile = None

                # Display greeting
                fputs(
                    u'\nWelcome to ChatREPL (Model: %s)\n' % args.model,
                    sys.stdout
                )
                display_help()

                # Main loop
                while True:
                    try:
                        single_interaction(conversation)
                    except EOFError:
                        break

                # Write readline history file before exiting
                if readline is not None and histfile is not None:
                    readline.write_history_file(histfile)

    main()
