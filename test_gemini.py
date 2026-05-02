import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
# Try v1 instead of v1beta and gemini-pro instead of 1.5-flash
url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={key}"

body = json.dumps({
    "contents": [{"parts": [{"text": "Say 'Gemini Pro Online' if you can read this."}]}]
}).encode()

try:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
        print(data["candidates"][0]["content"]["parts"][0]["text"])
except Exception as e:
    print(f"FAILED: {e}")
