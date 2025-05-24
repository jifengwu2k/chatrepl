#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import getopt
import io
import json
import os
import os.path
import sys


# Python version compatibility setup
if sys.version_info < (3,):
    from urllib2 import Request, urlopen, URLError
    text_type = unicode
    binary_type = str

    def unicode_input(prompt):
        return raw_input(prompt).decode(sys.stdin.encoding)

    open_with_encoding = codecs.open

    def post_request_instance(url, data, headers):
        return Request(url, data=data, headers=headers)
else:
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    text_type = str
    binary_type = bytes
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

def perror(e):
    fputs(u'%s: %s\n' % (type(e).__name__, e), sys.stderr)


try:
    import readline
except ImportError as e:
    readline = None
    perror(e)
    fputs(u"\nFailed to import readline. This will affect the command-line interface functionality:\n", sys.stderr)
    fputs(u" - Line editing features (arrow keys, cursor movement) will be disabled\n", sys.stderr)
    fputs(u" - Command history (up/down keys) will not be available\n", sys.stderr)
    fputs(u"\nWhile the program will still run, the text input will be basic and limited.\n", sys.stderr)


def save_messages_to_file(messages, filename):
    """Save chat messages to a JSON file."""
    try:
        with open_with_encoding(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        perror(e)


def load_messages_from_file(filename):
    """Load chat messages from a JSON file."""
    try:
        with open_with_encoding(filename, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                if all(isinstance(message, dict) and isinstance(message.get(u'role', None), text_type) and isinstance(message.get(u'content', None), text_type) for message in loaded):
                    return loaded
            raise ValueError(u"Invalid JSON schema: Expected a list of dictionaries with keys 'role' (string) and 'content' (string). Got: %s" % loaded)
    except Exception as e:
        perror(e)
        return None


def print_messages(messages):
    """Print all messages in the conversation."""
    for msg in messages:
        role = msg[u'role'].capitalize()
        content = msg[u'content']
        fputs(u'\n%s: %s\n' % (role, content), sys.stdout)


def read_file_content(filename):
    """Read text_type content from a file."""
    try:
        with open_with_encoding(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        perror(e)
        return None


def chat_with_model(
    api_key,
    base_url,
    model,
    messages
):
    """Send messages to the model using standard library HTTP requests."""
    url = u"%s/chat/completions" % base_url
    headers = {
        u"Content-Type": u"application/json",
        u"Authorization": u"Bearer %s" % api_key
    }
    data = json.dumps({
        u"model": model,
        u"messages": messages,
        u"stream": True
    }).encode('utf-8')

    full_response = []
    try:
        req = post_request_instance(url, data=data, headers=headers)
        response = urlopen(req)
        for line in response:
            if line.startswith(b"data: "):
                chunk_data = line[6:].decode('utf-8').strip()  # Remove "data: " prefix
                if chunk_data == u"[DONE]":
                    break
                
                chunk = json.loads(chunk_data)
                content = chunk.get(u"choices", [{}])[0].get(u"delta", {}).get(u"content", "") or u""
                yield content
                full_response.append(content)

        # Append assistant's response to messages
        if full_response:
            assistant_message = {
                u"role": u"assistant",
                u"content": u''.join(full_response)
            }
            messages.append(assistant_message)
    except Exception as e:
        perror(e)


def get_single_input(prompt=u'> '):
    """Get a single-line input from the user."""
    # Do NOT use unicode_input().
    # When the user enters something and presses down BACKSPACE, the prompt is removed as well.
    return unicode_input(prompt)


def get_multiline_input():
    """Get multiline input from the user until EOF."""
    fputs(u'Enter EOF on a blank line to finish input:\n', sys.stdout)
    lines = []
    try:
        while True:
            line = get_single_input(u'> ')
            lines.append(line)
    except EOFError as e:
        pass

    return u'\n'.join(lines)


def display_help():
    fputs(u'\nEnter a message to send to the model or use one of the following commands:\n', sys.stdout)
    fputs(u':save <filename>  Save the conversation to <filename>\n', sys.stdout)
    fputs(u':load <filename>  Load a conversation from <filename>\n', sys.stdout)
    fputs(u':send <filename>  Send the contents of <filename>\n', sys.stdout)
    fputs(u':multiline        Enter multiline input\n', sys.stdout)
    fputs(u':help             Display help\n', sys.stdout)
    fputs(u':quit             Exit the program\n', sys.stdout)


def main():
    """Entry point for the chat interface."""
    # Default values
    api_key = None
    base_url = None
    model = None

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["api-key=", "base-url=", "model="])
    except getopt.GetoptError as e:
        perror(e)
        sys.exit(2)

    # Process options
    for opt, arg in opts:
        if opt == "--api-key":
            api_key = arg
        elif opt == "--base-url":
            base_url = arg
        elif opt == "--model":
            model = arg

    # Check required arguments
    if not api_key or not base_url or not model:
        print("Error: --api-key, --base-url, and --model are required arguments")
        sys.exit(2)

    if readline is not None:
        histfile = os.path.join(os.path.expanduser("~"), ".chat_history")
        try:
            readline.read_history_file(histfile)
        except Exception:
            pass

    messages = []

    fputs(u'\nWelcome to Terminal Chat (Model: %s)\n' % model, sys.stdout)
    display_help()

    while True:
        try:
            user_input = get_single_input(u'\nUser: ').strip()

            if user_input.startswith(u':'):
                tokens = user_input.split()
                cmd = tokens[0] if tokens else u""
                args = tokens[1:]

                if cmd == u":save" and len(args) == 1:
                    filename = args[0]
                    save_messages_to_file(messages, filename)
                    fputs(u"Conversation saved to %s\n" % filename, sys.stdout)
                    continue
                elif cmd == u":load" and len(args) == 1:
                    filename = args[0]
                    loaded = load_messages_from_file(filename)
                    if loaded is not None:
                        messages = []
                        messages.extend(loaded)
                        fputs(u"Loaded conversation from %s\n" % filename, sys.stdout)
                        print_messages(messages)
                    continue
                elif cmd == u":send" and len(args) == 1:
                    filename = args[0]
                    file_content = read_file_content(filename)
                    if file_content is not None:
                        user_input = file_content
                    else:
                        continue
                elif cmd == u":multiline" and not args:
                    user_input = get_multiline_input()
                elif cmd == u":help" and not args:
                    display_help()
                    continue
                elif cmd == u":quit" and not args:
                    break
                else:
                    fputs(u"Unknown command.\n", sys.stdout)
                    display_help()
                    continue

            if not user_input:
                continue

            # Send message to model
            user_message = {
                u"role": u"user",
                u"content": user_input
            }
            messages.append(user_message)

            fputs(u'\nAssistant: ', sys.stdout)
            for chunk in chat_with_model(api_key, base_url, model, messages):
                fputs(chunk, sys.stdout)
            fputs(u'\n', sys.stdout)

        except (KeyboardInterrupt, EOFError):
            break

    # Save history before exiting
    if readline is not None:
        readline.write_history_file(histfile)


if __name__ == "__main__":
    main()
