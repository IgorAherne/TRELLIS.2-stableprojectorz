# api_spz/main_api.py
import logging
import os
import sys
import platform
import torch
from contextlib import asynccontextmanager

# Add the parent directory to sys.path to allow imports from api_spz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the StableProjectorz API routes
from api_spz.routes.generation import router as generation_router
from api_spz.core.state_manage import state

# Set up logging
os.makedirs("temp", exist_ok=True)
os.makedirs("temp/current_generation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("temp", "api.log"))
    ]
)
logger = logging.getLogger("trellis2_api")

# Print system information
print(
    f"\n[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}\n"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run Trellis2-StableProjectorz API server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=7960, help="Port to bind the server to")
parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unrecognized arguments: {unknown}")

# Print startup information
print("\n" + "="*50)
print("Trellis2-StableProjectorz API Server is starting up:")
print("If it's the first time, neural nets will download. Next runs will be faster.")
print("Touching this window will pause it. If it happens, click inside it and press 'Enter' to unpause")
print("="*50 + "\n")


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and resources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state.initialize_pipeline(device=device)
    logger.info(f"Initialized Trellis 2 pipeline on {device}")

    print("\n" + "="*50)
    print(f"Trellis2-StableProjectorz API Server v1.0.0")
    print(f"Server is active and listening on {args.host}:{args.port}")
    logger.info(f"Now in StableProjectorz, enter the 3D mode, click on the connection button and enter {args.host}:{args.port}")
    print("="*50 + "\n")

    yield  # This is where the FastAPI application runs

    # Shutdown: Clean up resources
    state.cleanup()
    logger.info("Server shutting down, resources cleaned up")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Trellis2-StableProjectorz API",
    description="API for Trellis 2 3D generation compatible with StableProjectorz",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the generation routes
app.include_router(generation_router)

@app.get("/")
async def root():
    return {
        "message": "Trellis2-StableProjectorz API is running",
        "status": "ready",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")