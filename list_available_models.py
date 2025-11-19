"""List available Gemini models"""
import google.generativeai as genai
from scripts.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

print("Available models:")
print("=" * 70)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print("-" * 70)
