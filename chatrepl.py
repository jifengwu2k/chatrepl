#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
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


# Are we under interactive mode?
IS_INTERACTIVE = sys.stdin.isatty()


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


# Try to import readline under interactive mode
if IS_INTERACTIVE:
    try:
        import readline
    except ImportError as e:
        readline = None
        perror(e)
        fputs(u"\nFailed to import readline. This will affect the command-line interface functionality:\n", sys.stderr)
        fputs(u" - Line editing features (arrow keys, cursor movement) will be disabled\n", sys.stderr)
        fputs(u" - Command history (up/down keys) will not be available\n", sys.stderr)
        fputs(u"\nWhile the program will still run, the text input will be basic and limited.\n", sys.stderr)
else:
    readline = None


def save_messages_to_file(messages, filename):
    """Save chat messages to a JSON file."""
    with open_with_encoding(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def load_messages_from_file(filename):
    """Load chat messages from a JSON file."""
    with open_with_encoding(filename, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        if isinstance(loaded, list) and all(
            (
                isinstance(message, dict)
                and message.get(u'role', None) in (u'user', u'assistant')
                and isinstance(message.get(u'content', None), text_type)
            )
            for message in loaded
        ):
            return loaded
        else:
            raise ValueError(u"Invalid JSON schema: Expected a list of dictionaries with keys 'role' ('user' or 'assistant') and 'content' (string). Got: %s" % loaded)


def print_loaded_messages(loaded_message_list):
    """Print all loaded messages.
    Return number of user messages."""
    num_user_messages_encountered = 0
    
    for message in loaded_message_list:
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

    return num_user_messages_encountered


def read_file_content(filename):
    """Read text_type content from a file."""
    with open_with_encoding(filename, 'r', encoding='utf-8') as f:
        return f.read()


# Global message counter
# Like IPython's message counter, incremented for non-empty user messages
# Read in `get_single_message_content_from_model`
# Written in `get_single_message_content_from_user`, `main`
global_message_counter = 1


# Model-facing functions
def send_messages_to_model_and_stream_response(
    api_key,
    base_url,
    model,
    message_list
):
    """Send messages to the model using standard library HTTP requests."""
    url = u"%s/chat/completions" % base_url
    headers = {
        u"Content-Type": u"application/json",
        u"Authorization": u"Bearer %s" % api_key
    }
    data = json.dumps({
        u"model": model,
        u"messages": message_list,
        u"stream": True
    }).encode('utf-8')

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


def get_single_message_content_from_model(
    api_key,
    base_url,
    model,
    message_list
):
    global global_message_counter
    
    fputs(
        u'\nAssistant [%d]: ' % global_message_counter,
        sys.stdout
    )
    
    full_response = []
    for response in send_messages_to_model_and_stream_response(
        api_key,
        base_url,
        model,
        message_list
    ):
        fputs(response, sys.stdout)
        full_response.append(response)
    
    fputs(u'\n', sys.stdout)
    
    return u''.join(full_response)


# User-facing functions
def display_help():
    fputs(u'\nEnter a message to send to the model or use one of the following commands:\n', sys.stdout)
    fputs(u':multiline        Enter multiline input\n', sys.stdout)
    fputs(u':send TEXTFILE    Send the contents of TEXTFILE\n', sys.stdout)
    fputs(u':load JSONFILE    Load a conversation from JSONFILE\n', sys.stdout)
    fputs(u':save JSONFILE    Save the conversation to JSONFILE\n', sys.stdout)
    fputs(u':help             Display help\n', sys.stdout)
    fputs(u':quit (or Ctrl-D) Exit the program\n', sys.stdout)


def get_single_line_input(prompt=u'> '):
    """Get a single-line input from the user."""
    # Do NOT use unicode_input().
    # When the user enters something and presses down BACKSPACE, the prompt is removed as well.
    return unicode_input(prompt)


def get_multi_line_input():
    """Get multi-line input from the user until EOF."""
    fputs(u'Enter EOF on a blank line to finish input:\n', sys.stdout)
    lines = []
    try:
        while True:
            line = get_single_line_input(u'> ')
            lines.append(line)
    except EOFError as e:
        pass

    return u'\n'.join(lines)


def get_single_message_content_from_user(mutable_message_list):
    global global_message_counter
    
    while True:
        user_input = get_single_line_input(
            u'\nUser [%d]: ' % global_message_counter
        ).strip()

        if user_input.startswith(u':'):
            tokens = user_input.split()
            cmd = tokens[0] if tokens else u""
            args = tokens[1:]

            # :multiline
            if cmd == u":multiline" and not args:
                content = get_multi_line_input()
            # :send <textfile>
            elif cmd == u":send" and len(args) == 1: 
                try:
                    content = read_file_content(args[0])
                except Exception as e:
                    perror(e)
                    
                    global_message_counter += 1
                    continue
            # :load <jsonfile>
            elif cmd == u":load" and len(args) == 1:
                try:
                    loaded_messages = load_messages_from_file(args[0])
                    fputs(u"Loaded conversation from %s:\n" % args[0], sys.stdout)
                    
                    last_message_counter = print_loaded_messages(loaded_messages)
                    
                    mutable_message_list[:] = loaded_messages
                    global_message_counter = last_message_counter + 1
                    continue
                except Exception as e:
                    perror(e)
                    
                    global_message_counter += 1
                    continue
            # :save <jsonfile>
            elif cmd == u":save" and len(args) == 1:
                try:
                    save_messages_to_file(mutable_message_list, args[0])
                    fputs(u"Conversation saved to %s\n" % args[0], sys.stdout)
                except Exception as e:
                    perror(e)
                
                global_message_counter += 1
                continue
            # :help
            elif cmd == u":help" and not args:
                display_help()

                global_message_counter += 1
                continue
            # :quit
            elif cmd == u":quit" and not args:
                # Same as pressing Ctrl-D (sending EOF)
                raise EOFError
            else:
                fputs(u"Unknown command.\n", sys.stdout)
                display_help()
                
                global_message_counter += 1
                continue
        else:
            content = user_input

        # We do not allow content to be empty
        if not content:
            continue
        
        return content


def main():
    """Entry point for the chat interface."""
    global global_message_counter
    
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
        print_loaded_messages(load_messages_from_file(args.print))
    else:
        if (
            not args.api_key
            or not args.base_url
            or not args.model
        ):
            parser.error('Must provide --api-key, --base-url, and --model when not printing a conversation using --print')
        
        # Initialize messages
        if args.load:
            messages = load_messages_from_file(args.load)
        else:
            messages = []
    
        # Non-interactive mode
        if not IS_INTERACTIVE:
            user_message_content = sys.stdin.read()
            
            if user_message_content:
                messages.append(
                    {
                        u'role': 'user',
                        u'content': user_message_content
                    }
                )
                
                for response in send_messages_to_model_and_stream_response(
                    args.api_key,
                    args.base_url,
                    args.model,
                    messages
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
        
            # Display greeting
            fputs(
                u'\nWelcome to ChatREPL (Model: %s)\n' % args.model,
                sys.stdout
            )
            display_help()
            
            # Main loop
            while True:
                try:
                    user_message_content = get_single_message_content_from_user(
                        messages
                    )
                    
                    messages.append(
                        {
                            u'role': 'user',
                            u'content': user_message_content
                        }
                    )
                except EOFError:
                    break
                
                try:
                    assistant_message_content = get_single_message_content_from_model(
                        args.api_key,
                        args.base_url,
                        args.model,
                        messages
                    )
                    
                    messages.append(
                        {
                            u'role': u'assistant',
                            u'content': assistant_message_content
                        }
                    )
                except URLError as e:
                    perror(e)
                    
                    # Also remove the last user message
                    messages.pop()
                    
                    # And skip incrementing global_message_counter
                    continue
                
                global_message_counter += 1
                
            # Write readline history file before exiting
            if readline is not None:
                readline.write_history_file(histfile)


if __name__ == "__main__":
    main()