import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

# ADK Core Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# API Format is http://localhost:8000/search?q=tcs+events 

# 1. Agent Configuration
# Using gemini-1.5-flash for the most stable tool-calling logic
search_agent = LlmAgent(
    name="UniversalSearchBot",
    model="gemini-2.5-flash", 
    tools=[google_search],
    instruction=(
    "You are a professional researcher. "
    "CRITICAL: Always use 'google_search' to find current data for March 2026. "
    "Search for and collect the following values: Event Name, Event Type (Summit/Webinar/Conference), "
    "Sponsorship Details, Location, and Link. "
    "After the search, you MUST provide a 3-4 sentence summary of the findings. "
    "Do not stop until you have written the summary text."
    )
)

# 2. Runner & Session Setup
runner = Runner(
    agent=search_agent,
    app_name="SearchAPI",
    session_service=InMemorySessionService(),
    auto_create_session=True
)

app = FastAPI()

@app.get("/search")
async def execute_search(q: str):
    user_id = "user_001"
    session_id = f"sess_{hash(q)}" # Fresh session per query to prevent context overlap

    try:
        # Initial query to the agent
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=q)])
        )

        final_answer = ""
        
        # We loop through events to collect all generated text
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_answer += part.text

        # --- THE FIX FOR "ALL CASES" ---
        # If final_answer is empty, it means the agent called a tool 
        # but the runner stopped. We send a "hidden" prompt to force the summary.
        if not final_answer.strip():
            retry_events = runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text="Now, summarize those results for me.")])
            )
            for rev in retry_events:
                if rev.content and rev.content.parts:
                    for rp in rev.content.parts:
                        if rp.text:
                            final_answer += rp.text

        if not final_answer.strip():
            return {"error": "The agent was unable to synthesize a response. Check your API limits."}

        return {
            "query": q,
            "answer": final_answer.strip()
        }

    except Exception as e:
        print(f"Debug Log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
