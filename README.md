# Home Assistant - OpenAI API Conversation
A Home Assistant integration to enable use of any OpenAI compatible API endpoint.

## Features
- Use with any OpenAI-compatible API endpoint such as Openrouter.ai or self-hosted solutions like Open WebUI
- Configure custom base URLs and organization IDs
- Supports all OpenAI models and custom endpoints that follow the OpenAI API specification

## Installation

### Manual Installation
1. Copy the `openai_api_conversation` folder from this repository to your Home Assistant's custom_components folder:
   ```
   custom_components/openai_api_conversation/
   ```

2. Restart Home Assistant
3. Go to Settings -> Devices & Services -> Add Integration
4. Search for "OpenAI API Conversation" and set it up with your API key and custom endpoint information

### HACS Installation
1. Add this repository as a custom repository in HACS
2. Install the "OpenAI API Conversation" integration
3. Restart Home Assistant
4. Go to Settings -> Devices & Services -> Add Integration
5. Search for "OpenAI API Conversation" and set it up

## Configuration
When setting up the integration, you can:

1. Enable "Custom Endpoint" to use alternative OpenAI-compatible API endpoints
2. Set a Base URL (e.g., "https://openrouter.ai/api/v1" for OpenRouter or "http://your-server:5000/v1" for Open WebUI)
3. Optionally provide an Organization ID if your service requires it

## Supported Services
- OpenRouter.ai
- Self-hosted Open WebUI
- Any service that implements the OpenAI API specification
