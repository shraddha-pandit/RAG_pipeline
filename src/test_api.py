import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    raise ValueError("GOOGLE_API_KEY not set in .env file")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.8-flash")
response = model.generate_content("Say 'API key is working!' and nothing else.")

print(response.text.strip())
