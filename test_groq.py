import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "user", "content": "Reply with: API works"}
    ],
    "max_tokens": 20,
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    print("HTTP status:", response.status_code)

    if response.ok:
        print("AI response:", response.json()["choices"][0]["message"]["content"])
    else:
        # Avoid printing headers or your API key.
        print("API error:", response.text[:1000])

except requests.RequestException as exc:
    print("Connection error:", str(exc))