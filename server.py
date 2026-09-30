#!/usr/bin/env python3
"""
server.py — Launch script for the Fuzzy-Evolutionary ICU Arrhythmia Pipeline web server.
Starts FastAPI backend server on http://localhost:8000.
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = 8000
    host = "0.0.0.0"
    print("\n=======================================================")
    print(">> Starting Fuzzy-Evolutionary ICU Arrhythmia Pipeline")
    print(f">> Website & API: http://localhost:{port}")
    print(f">> Interactive Docs: http://localhost:{port}/docs")
    print("=======================================================\n")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
