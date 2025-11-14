from google import genai
import json, re
from api_calls.private_keys import secret_key 

client = genai.Client(api_key=secret_key)

def extract_json(text: str) -> dict:
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
        return {"label": "unknown", "reason": f"No JSON found in response: {text}"}
    except json.JSONDecodeError:
        return {"label": "unknown", "reason": f"JSON decode error for text: {text}"}
    except Exception as e:
        return {"label": "unknown", "reason": f"Extraction error: {e} with text: {text}"}


def verify_with_vertexai(statement: str) -> dict:
    prompt = f"""You are an expert AI fact-checking assistant specialized in verifying news.
        
        ### Task:
        Your goal is to determine if the following statement is **True** or **Fake**. The statement is based on **current, real-world events**.
        
        ### Statement:
        {statement}
        
        ### Instructions (Strict):
        1. **MUST USE GOOGLE SEARCH** to find recent, credible, and official sources (news, government, etc.) to verify the claims.
        2. Base your conclusion **STRICTLY on the evidence found**.
        3. If the claims are substantively correct and widely reported in major news outlets, return **"True"**.
        4. Output **ONLY** the required strict JSON object. Do not provide reasoning.
        
        ### Output Format:
        {{
        "label": "True" or "Fake"
        }}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "system_instruction": "You are a specialized fact-checking assistant. Your only output is the required strict JSON object: {'label': 'True' or 'Fake'}.",
                "tools": [{"google_search": {}}]
            }
        )
        
        raw_text = response.text.strip()
        return extract_json(raw_text)

    except Exception as e:
        if "unexpected keyword argument 'system_instruction'" in str(e):
             return {"label": "error", "reason": "API call failed: Check your SDK version. The 'system_instruction' argument may need to be moved to 'config' if you are on an older version of the SDK, but this code is using the current correct format. Ensure your 'google-genai' library is up to date."}
        return {"label": "error", "reason": f"API call failed: {e}"}
