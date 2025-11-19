"""Test Gemini API response structure"""
import google.generativeai as genai
from scripts.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

# Tắt safety filters
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

model = genai.GenerativeModel('gemini-flash-latest', safety_settings=safety_settings)

print(f"Testing with model: gemini-flash-latest")
print(f"API Key: {GEMINI_API_KEY[:20]}...")

try:
    response = model.generate_content(
        "What is diversification in investing?",
        generation_config={
            'temperature': 0.7,
            'max_output_tokens': 500,
        }
    )
    
    print("=" * 50)
    print("RESPONSE OBJECT:")
    print(f"Type: {type(response)}")
    print(f"Has 'parts': {hasattr(response, 'parts')}")
    
    # Kiểm tra candidates
    if hasattr(response, 'candidates'):
        print(f"\nCandidates: {len(response.candidates) if response.candidates else 0}")
        for i, candidate in enumerate(response.candidates):
            print(f"\nCandidate {i}:")
            print(f"  finish_reason: {candidate.finish_reason}")
            print(f"  safety_ratings: {candidate.safety_ratings}")
            if hasattr(candidate, 'content') and candidate.content:
                print(f"  content.parts: {candidate.content.parts}")
    
    if hasattr(response, 'parts'):
        print(f"\nParts: {response.parts}")
    
    if hasattr(response, 'prompt_feedback'):
        print(f"\nPrompt feedback: {response.prompt_feedback}")
    
    print("\n" + "=" * 50)
    print("TRYING TO GET TEXT:")
    try:
        text = response.text
        print(text)
    except Exception as e:
        print(f"Error getting text: {e}")
    
    print("=" * 50)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
