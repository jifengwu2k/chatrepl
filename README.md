A Python 2.7+ REPL for interacting with LLMs with an OpenAI Chat Completions-compatible API.

## Features

- **Minimal Dependencies:** Works with stock Python 2.7+ and 3.x
- **Python-Powered REPL:** Native interactive shell with special chat commands as Python functions
- **Streaming Responses:** See model replies as they're generated
- **Multiline Editing:** Use your default editor for long/structured prompts
- **File & Image Support:** Attach text files or images to your conversation
- **Conversation Persistence:** Save/load full conversations as JSON
- **Markdown Export:** Archive conversations in Markdown for easy reference
- **Enhanced Input:** Optional readline support for history and line editing
- **Dual Modes:** Both interactive REPL and pipe-friendly CLI

## Installation

```bash
pip install chatrepl
```

## Interactive Mode (CLI)

```bash
python -m chatrepl \
  --api-key "your-api-key" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o"
```

You'll get a Python shell preloaded with chat helper functions:

| Function               | Description                              |
|------------------------|------------------------------------------|
| `send(text)`           | Send a message and stream the response   |
| `append(text)`         | Append text to conversation (don't send) |
| `multiline()`          | Append multiline input via your editor   |
| `img(img_file_path)`   | Append an image file                     |
| `txt(txt_file_path)`   | Append a text file                       |
| `load(json_file_path)` | Load conversation from JSON              |
| `save(json_file_path)` | Save conversation to JSON                |
| `export(md_file_path)` | Export conversation to Markdown          |
| `correct()`            | Correct (edit) last model response       |

 `exit()` or EOF (`Ctrl-D` on Unix) leaves REPL.

### Basic Example

```text
Welcome to ChatREPL! Use the following commands to interact with gpt-4o:

>>> send('Explain recursion.')
Assistant: Sure! Recursion is...

>>> multiline()
(opens your editor for multiline input)

>>> img('diagram.png')
(appends the given image to the conversation)

>>> send('Describe this image.')
Assistant: ...

>>> save('chat.json')
(saves the entire chat so far)
```

## Non-interactive Mode (Piped Input)

```bash
$ uname -a | python -m chatrepl --api-key <your_api_key> --base-url <your_base_url> --model <model_name>
The output you've provided appears to be system information from ... [output streamed to STDOUT]
```

## Programmatic Usage (API)

See [chat-completions-conversation](https://github.com/jifengwu2k/chat-completions-conversation).

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
