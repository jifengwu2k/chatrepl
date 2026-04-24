# chatrepl

A minimal agent for OpenAI-compatible Chat Completions APIs with four built-in tools:

- `read`
- `write`
- `edit`
- `shell`

## Features

- Interactive Python-powered REPL
- Python 2.7+ and Python 3 compatible
- Works with OpenAI-compatible `/chat/completions` endpoints
- Discovery of `AGENTS.md` and `CLAUDE.md` under the current working directory
- Streaming assistant responses
- Editable multiline input through your editor
- Non-interactive CLI mode for piped input or one-shot prompts

## Installation

```bash
pip install chatrepl
```

## Usage

### CLI options

```text
-k, --api-key           API key for the OpenAI-compatible endpoint
-u, --base-url          Base URL, e.g. http://localhost:11434/v1
-m, --model             Model ID
--no-stream             Disable streaming
--no-context-files      Disable AGENTS.md and CLAUDE.md discovery
```

### Interactive REPL

```bash
chatrepl \
  --api-key "your-api-key" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-5.4"
```

You enter a Python interactive console with helper functions preloaded.

Available commands:

| Function | Description |
|---|---|
| `send(text='', image_path=None, stream=True)` | Send a message and let the agent complete tool calls, optionally with a local image path used as-is and streaming control |
| `append(text)` | Append a user message without sending |
| `multiline()` | Append multiline input from your editor |
| `txt(path)` | Append a UTF-8 text file as a user message using the provided path as-is |
| `img(path)` | Append a local image as a user message; files are embedded as data URLs using the provided path as-is |
| `reset()` | Reset to only the system prompt |

Exit with `exit()` or EOF.

### One-shot prompt

```bash
chatrepl \
  --api-key "your-api-key" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o" \
  "Inspect this repository and summarize the build system"
```

### Piped input

```bash
cat prompt.txt | chatrepl \
  --api-key "your-api-key" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o"
```

## Tool model

The agent is intentionally small and constrained.

### `read`
Reads a text file with optional `offset` and `limit` arguments.

- Uses the provided path as-is

### `write`
Writes full file contents.

- Creates parent directories automatically
- Rewrites the destination file completely

### `edit`
Applies exact text replacements to an existing file.

Rules:
- each `oldText` must match exactly once
- edits must not overlap
- all edits are matched against the original file

Returns a unified diff after a successful edit.

### `shell`
Runs a shell command in the current working directory.

- live output is streamed to the terminal
- full output is returned to the model

## Example session

### Turn 1: appended user message, then send with no new text

This is useful because it exercises the path where `send()` is called with no pending user message.

```python
>>> append("Reply with exactly APPENDED-OK.")
>>> send(stream=False)
```

Assistant response:

```text
APPENDED-OK
```

### Turn 2: streamed text + streamed tool-call deltas + follow-up tool result

```python
>>> send("Read hello.txt and quote its contents back to me.", stream=True)
```

Mock streamed assistant output:

```text
I’ll inspect the file.
[assistant is preparing tool call(s)]
[tool_call 0]
id: call_read_1
type: function
name += re
name += ad
arguments += {"path":"he
arguments += llo.txt"}
```

Then tool execution:

```text
[assistant is using 1 tool(s)]

[tool read]
Hello!
```

Then assistant follow-up response:

```text
The file contains `Hello!`.
```

### Turn 3: multiple tool calls in one assistant message

This exercises:
- multiple tool calls
- `write`
- `edit`
- `read`
- diff generation
- tool index tracking in streaming

```python
>>> send("Create tmp/demo.txt with alpha and beta on separate lines, change beta to gamma, then read it back.", stream=True)
```

Mock streamed assistant output:

```text
I’ll create the file, patch it, and verify the result.
[assistant is preparing tool call(s)]
[tool_call 0]
id: call_write_1
type: function
name += wr
name += ite
arguments += {"path":"tmp/demo.txt","content":"alpha\nbeta\n"}
[tool_call 1]
id: call_edit_1
type: function
name += ed
name += it
arguments += {"path":"tmp/demo.txt","edits":[{"oldText":"beta","newText":"gamma"}]}
[tool_call 2]
id: call_read_2
type: function
name += read
arguments += {"path":"tmp/demo.txt"}
```

Then tool execution:

```text
[assistant is using 3 tool(s)]

[tool write]
Successfully wrote 11 characters to <repo>/tmp/demo.txt

[tool edit]
Applied 1 edit(s) to <repo>/tmp/demo.txt

--- <repo>/tmp/demo.txt
+++ <repo>/tmp/demo.txt
@@
-alpha
-beta
+alpha
+gamma

[tool read]
alpha
gamma
```

Then assistant follow-up:

```text
Done. `tmp/demo.txt` now contains:

- alpha
- gamma
```

### Turn 4: shell tool, stdout + stderr, non-streaming

```python
>>> send("Run a shell command that prints one line to stdout and one to stderr.", stream=False)
```

Assistant response with tool call:

```text
I’ll run a small shell command.
```

Then tool execution:

```text
[assistant is using 1 tool(s)]

[tool shell]
$ printf 'stdout-line\n'; printf 'stderr-line\n' >&2
Exit code: 0

stdout-line
stderr-line
```

Then assistant follow-up:

```text
The shell command succeeded and produced both stdout and stderr output.
```

## License

This project is licensed under the [MIT License](LICENSE).
