import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai

# --- CONFIGURATION ---
app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Configure Google Gemini
# We will set the API key in the environment variables later during deployment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- DATA MODELS ---
class TopicRequest(BaseModel):
    topic: str

class ExplanationResponse(BaseModel):
    what_it_is: str
    human_connection: str
    social_influence: str
    relevant_studies: str
    confidence_level: str
    confidence_reason: str

# --- CORE AI LOGIC ---
SYSTEM_PROMPT = """
You are a concise Human Relevance Explanation Engine.
The user is a general public member.

STRICT SAFETY GUARDRAILS:
1. If the user asks for instructions on how to harm themselves or others, build weapons, or commit crimes, REFUSE with: "I cannot provide information on this topic due to safety guidelines."
2. If the user describes specific symptoms and asks for a diagnosis, REFUSE with: "I cannot provide medical diagnoses. Please consult a professional."

If the topic is safe, provide a structured answer:

1. What it is: A brief factual explanation (1-2 sentences).
2. Human connection: How it relates to biology, emotions, or psychology.
3. Social/behavioral influence: Impact on behavior or society.
4. Relevant studies: Mention "General Scientific Consensus" if no specific famous paper exists. If citing, use (Author, Year).
5. Confidence level: rate as "High", "Moderate", or "Preliminary".
6. Confidence reason: A very short justification (e.g. "Decades of study" or "New emerging field").

Your output must be raw JSON with keys: what_it_is, human_connection, social_influence, relevant_studies, confidence_level, confidence_reason.
Do not use Markdown formatting in the JSON.
"""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/explain")
async def explain_topic(request: TopicRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured.")

    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Combine system prompt and user input
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER TOPIC: {request.topic}\n\nRespond in JSON."
        
        response = model.generate_content(full_prompt)
        
        # Simple cleanup to ensure we get pure text/json
        text_response = response.text.replace('```json', '').replace('```', '').strip()
        
        # In a production app, we would parse JSON strictly here.
        # For this MVP, we will return the text and let the frontend handle display,
        # or we can parse it if Gemini follows instructions well.
        import json
        try:
            data = json.loads(text_response)
            return data
        except json.JSONDecodeError:
            # Fallback if AI doesn't return perfect JSON
            return {
                "what_it_is": "Error parsing AI response.",
                "human_connection": text_response,
                "social_influence": "",
                "relevant_studies": "",
                "confidence_level": "Low",
                "confidence_reason": "Format error"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
