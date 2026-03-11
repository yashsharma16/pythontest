import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ADK Core Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

load_dotenv()

# 1. Initialize FastAPI once
app = FastAPI()

# 2. CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://vtest.mygreenhorn.com",
    "https://vtest.mygreenhorn.com/",
    "*"  # In production, it's safer to remove "*" and keep only your domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow everything
    allow_credentials=False, # Must be False if origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Agent Configuration
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

# 4. Runner & Session Setup
runner = Runner(
    agent=search_agent,
    app_name="SearchAPI",
    session_service=InMemorySessionService(),
    auto_create_session=True
)

# 5. API Routes
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

        # Collect text parts
        final_answer = "".join([
            part.text for event in events 
            if event.content and event.content.parts 
            for part in event.content.parts if part.text
        ])

        # Logic to force summary if tool-call didn't return text immediately
        if not final_answer.strip():
            retry_events = runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text="Please provide the summary of those search results.")])
            )
            final_answer = "".join([
                part.text for rev in retry_events 
                if rev.content and rev.content.parts 
                for part in rev.content.parts if part.text
            ])

        if not final_answer.strip():
            return {"error": "Synthesis failed."}

        return {"query": q, "answer": final_answer.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
