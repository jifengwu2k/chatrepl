A Python 2.7+ REPL for interacting with LLMs with an OpenAI Chat Completions-compatible API.

## Features

- **Zero Dependencies:** Works with stock Python 2.7+ and 3.x
- **Interactive Chat:** Natural REPL interface with conversation tracking
- **Streaming Responses:** See responses as they're generated
- **Multiline Input:** Support for complex prompts with `:multiline`
- **File Integration:** Load prompts from text files with `:send <textfile>`
- **Conversation Persistence:** Save/load complete conversations in JSON format
- **Enhanced Input:** Optional readline support for history and line editing
- **Dual Modes:** Both interactive REPL and pipe-friendly CLI
- **API Ready:** Can be imported as a module for programmatic use

## Installation

```bash
pip install chatrepl
```

## Interactive Mode (CLI)

```bash
$ python -m chatrepl \
  --api-key "your-api-key" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o"
```

### Basic Conversation

```text
User [1]: Explain recursion to a 5-year-old

Assistant [1]: Imagine you're holding a doll that has...
```

### Using Files
```text
User [2]: :send code.py
Assistant [2]: I notice this Python code could be improved...

User [3]: :save review_chat.json
```

### Multiline Input

```text
User [4]: :multiline
Enter EOF on a blank line to finish input:
> Compare these programming languages:
> 1. Python
> 2. Rust
> 3. Go
> [Ctrl-D]

Assistant [4]: Here's a comparison:
1. Python - High-level, interpreted...
2. Rust - Systems programming...
3. Go - Compiled, concurrent...
```

### Non-interactive Mode (Piped Input)

```bash
$ uname -a | python -m chatrepl --api-key <your_api_key> --base-url <your_base_url> --model <model_name>
The output you've provided appears to be system information from ... [output streamed to STDOUT]
```

### Print Saved Conversations

```bash
$ python -m chatrepl --print conversation.json
User [1]: ...

Assistant [1]: ...
```

### Interactive Commands

- `:multiline` - Enter multiline input mode (end with blank line + Ctrl-D)
- `:send TEXTFILE` - Send contents of TEXTFILE
- `:load JSONFILE` - Load conversation from JSONFILE
- `:save JSONFILE` - Save conversation to JSONFILE
- `:help` - Show help
- `:quit` or `Ctrl-D` - Exit the program

### Best Practices

1. For long sessions, periodically save with `:save`
2. Use `:multiline` for structured prompts (lists, code, etc.)
3. JSON files can be edited manually for prompt engineering

## Programmatic Usage (API)

```python
from chatrepl import Conversation

# Initialize conversation
conv = Conversation(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o"
)

# Load conversation
conv.load_messages_from_file("conversation.json")

# Access message history
for msg in conv.messages:
    print(f"{msg['role']}: {msg['content']}")

# Add a message to the model's message list without obtaining a response
conv.add_message('Help me with the following tasks:')

# Send message and stream response
print("Assistant: ", end="")
for chunk in conv.send_message_to_model_and_stream_response("Hello!"):
    print(chunk, end="")
print()

# Save conversation
conv.save_messages_to_file("conversation.json")
```

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
