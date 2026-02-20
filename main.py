"""FastAPI app: mounts ingest, summary, qa, evaluate, chunks."""

from fastapi import FastAPI

from backend.api import (
    router_chunks,
    router_evaluate,
    router_ingest,
    router_qa,
    router_summary,
)

app = FastAPI(
    title="PolicyExplainer",
    description="AI-powered insurance policy document explanation API.",
    version="0.1.0",
)

app.include_router(router_ingest, tags=["ingest"])
app.include_router(router_summary, prefix="/summary", tags=["summary"])
app.include_router(router_qa, prefix="/qa", tags=["qa"])
app.include_router(router_evaluate, prefix="/evaluate", tags=["evaluate"])
app.include_router(router_chunks, prefix="/chunks", tags=["chunks"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
