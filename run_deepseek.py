import requests

def ask_chat(prompt: str) -> str:
    payload = {
        "model": "deepseek-r1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ]
    }
    resp = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json=payload
    )
    resp.raise_for_status()
    data = resp.json()  # single JSON object
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(ask_chat("What’s the average-case runtime of quicksort?"))
