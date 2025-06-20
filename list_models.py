# list_models.py
import os
from anthropic import Client

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ANTHROPIC_API_KEY")

    client = Client(api_key=api_key)
    page = client.models.list()    # yields ModelInfo objects

    print("Available models:")
    for model in page:
        # each `model` is a ModelInfo with an `id` field
        print(" ", model.id)

if __name__ == "__main__":
    main()
