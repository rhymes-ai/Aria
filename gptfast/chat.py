from typing import List, Optional

import requests
from generate import GenerationConfig, Generator, ModelConfig
from PIL import Image


class ChatMessage:
    def __init__(self, role: str, content: str, image_path: Optional[str] = None):
        self.role = role
        self.content = content
        self.image_path = image_path


class AriaChat:
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.generator = Generator(model_config, generation_config)
        self.history: List[ChatMessage] = []

    def add_message(self, role: str, content: str, image_path: Optional[str] = None):
        """Add a message to the chat history."""
        self.history.append(ChatMessage(role, content, image_path))

    def format_prompt(self) -> tuple[str, Optional[str]]:
        """Format the chat history into a prompt for the model."""
        messages = []
        images = []
        for msg in self.history:
            content = []
            if msg.image_path:
                content.append({"text": None, "type": "image"})
                images.append(msg.image_path)
            content.append({"text": msg.content, "type": "text"})
            messages.append({"role": msg.role, "content": content})

        processed_images = []
        for image in images:
            if isinstance(image, str):
                if image.startswith("http://") or image.startswith("https://"):
                    image = Image.open(requests.get(image, stream=True).raw)
                else:
                    image = Image.open(image)
                image = image.convert("RGB")
            processed_images.append(image)
        return messages, processed_images

    def chat(self, message: str, image_path: Optional[str] = None) -> str:
        """Send a message and get a response."""
        self.add_message("user", message, image_path)
        messages, image = self.format_prompt()

        response = self.generator.generate(messages, image)

        # Extract the assistant's response from the full generated text
        assistant_message = response.split("<|assistant|>")[-1].strip()
        # Remove the end token if present
        for stop_string in self.generator.generation_config.stop_strings:
            assistant_message = assistant_message.replace(stop_string, "").strip()

        self.add_message("assistant", assistant_message)
        return assistant_message

    def reset(self):
        """Clear the chat history."""
        self.history = []


if __name__ == "__main__":
    from pathlib import Path

    from gptfast.chat import AriaChat
    from gptfast.generate import GenerationConfig, ModelConfig

    model_config = ModelConfig(
        checkpoint_path=Path("checkpoints/rhymes-ai/Aria/model.pth"),
        compile=True,
    )
    generation_config = GenerationConfig(
        max_new_tokens=4096,
        top_k=40,
        temperature=0.8,
        cache_size=8192,
        stop_strings=["<|im_end|>"],
    )

    chat = AriaChat(model_config, generation_config)

    # Add welcome message and command instructions
    print("\n=== AriaChat Terminal Interface ===")
    print("\nAvailable commands:")
    print("  'help'    - Show this help message")
    print("  'quit'    - Exit the chat")
    print("  'reset'   - Clear chat history")
    print("  'image'   - Chat with an image")
    print("\nType your message or command to begin...")

    while True:
        user_input = input("\n> You: ").strip()

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "help":
            print("\nAvailable commands:")
            print("  'help'    - Show this help message")
            print("  'quit'    - Exit the chat")
            print("  'reset'   - Clear chat history")
            print("  'image'   - Chat with an image")
            continue
        elif user_input.lower() == "reset":
            chat.reset()
            print("Chat history cleared.")
            continue
        elif user_input.lower() == "image":
            image_path = input("Enter image path or URL: ").strip()
            message = input("Enter your message about the image: ").strip()
            response = chat.chat(message, image_path)
        else:
            response = chat.chat(user_input)

        print(f"\n> Aria: {response}")

    print("\nGoodbye!")
