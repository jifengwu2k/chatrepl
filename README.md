A command-line interface for interacting with LLMs with an OpenAI-compatible API. This script allows you to have conversational chats directly from your terminal.
## Features

*   **Interactive Chat:** Engage in conversational chats with LLMs.
*   **Conversation History:** Maintains a history of your conversation within a session.
*   **Save/Load Conversations:** Save your conversations to a file and load them later for resuming or reviewing.  The history file is stored at `~/.chat_history`.
*   **Multiline Input:** Supports entering multiline input for complex prompts.
*   **File Input:** Load a text file as a prompt.
*   **Unbuffered Output:** Provides immediate feedback in the terminal.
*   **Streaming Responses:** Displays responses as they arrive from OpenAI-compatible APIs.

## Prerequisites

*   Python 3.7 or newer
*   `openai` Python package.  Install with:

    ```bash
    pip install openai
    ```

## Usage

Run the script with the following arguments:

```bash
python terminal_chat.py --api-key <your_api_key> --base-url <your_base_url> --model <model_name>
```

*   `--api-key`: Your API key.
*   `--base-url`: Your base URL. (e.g., `https://api.openai.com/v1`)
*   `--model`: The name of the model to use (e.g., `gpt-4o`).

**Commands within the chat:**

*   `:save <filename>`: Save the current conversation to a file.
*   `:load <filename>`: Load a conversation from a file.
*   `:multiline`: Enter multiline input.
*   `:file <filename>`: Load the content of a file as input.
*   `:quit`: Exit the chat.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).