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


def main():
    args = sys.argv[1:]
    
    # Check for required argument
    if len(args) != 1:
        fputs("Error: Conversation file argument is required\n", sys.stderr)
        sys.exit(2)

    messages = load_messages_from_file(args[0])
    if messages is not None:
        print_messages(messages)


if __name__ == '__main__':
    main()
