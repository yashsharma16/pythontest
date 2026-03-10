import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

# ADK Core Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# 1. Agent Configuration
search_agent = LlmAgent(
    name="UniversalSearchBot",
    model="gemini-2.5-flash", 
    tools=[google_search],
    instruction=(
        "You are a professional researcher. "
        "CRITICAL: Always use 'google_search' to find current data for March 2026. "
        "Summarize findings in 3-4 sentences."
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
    session_id = f"sess_{hash(q)}" 

    try:
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=q)])
        )

        final_answer = "".join([part.text for event in events if event.content and event.content.parts for part in event.content.parts if part.text])

        # Logic to force summary if tool-call didn't return text
        if not final_answer.strip():
            retry_events = runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text="Please provide the summary of those search results.")])
            )
            final_answer = "".join([part.text for rev in retry_events if rev.content and rev.content.parts for part in rev.content.parts if part.text])

        if not final_answer.strip():
            return {"error": "Synthesis failed."}

        return {"query": q, "answer": final_answer.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Note: No uvicorn.run here for Gunicorn usage.
