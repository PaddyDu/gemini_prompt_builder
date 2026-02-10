# Gemini Prompt Builder for ComfyUI

A ComfyUI custom node that integrates Google's Gemini AI models with Google Search grounding to generate enhanced prompts.

## Features

- Support for multiple Gemini models (Flash, Pro, Flash-Lite)
- Google Search grounding integration
- Customizable system instructions
- API key management through environment variables or config file

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone <repository-url> gemini_prompt_builder
```

2. Configure your API key:
   - Copy `config.json.example` to `config.json`
   - Add your Gemini API key to `config.json`:
   ```json
   {
       "GEMINI_API_KEY": "your-api-key-here"
   }
   ```
   
   Alternatively, set the environment variable:
   ```bash
   export GEMINI_API_KEY=your-api-key-here
   ```

3. Restart ComfyUI

## Usage

After installation, you'll find the Gemini Prompt Builder node in ComfyUI's node menu. Configure the model, system instructions, and input prompt to generate enhanced prompts using Gemini AI.

## Requirements

- ComfyUI
- Python 3.7+
- Valid Google Gemini API key

## License

See LICENSE file for details.
