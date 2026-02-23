from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Updated imports to match the simplified api.py
from backend.api import (
    router_ingest,
    router_summary,
    router_qa,
    router_evaluate
)

app = FastAPI(title="PolicyExplainer API")

# Enable CORS so Streamlit can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(router_ingest, tags=["Ingest"])
app.include_router(router_summary, prefix="/summary", tags=["Summary"])
app.include_router(router_qa, prefix="/qa", tags=["Q&A"])
app.include_router(router_evaluate, prefix="/evaluate", tags=["Evaluate"])

@app.get("/")
async def root():
    return {"message": "PolicyExplainer API is running"}
