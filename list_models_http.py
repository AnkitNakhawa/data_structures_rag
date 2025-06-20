# list_models_http.py
import os
import requests

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in your env")

resp = requests.get(
    "https://api.anthropic.com/v1/models",
    headers={"x-api-key": API_KEY}
)
resp.raise_for_status()
models = resp.json()["models"]
for m in models:
    print(m["id"])
