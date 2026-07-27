"""
FastAPI inference service for credit card fraud detection.

TODO: implemented in Phase 9.
"""

from fastapi import FastAPI

app = FastAPI(title="Credit Card Fraud Detection API")


@app.get("/health")
def health():
    return {"status": "ok"}