from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from facescanner import run_scanner
from rebuild_floorplan import rebuild_visual

app = FastAPI()

# Allow React (localhost:3000 or 5173) to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Phygital Backend is Running"}

@app.post("/scan")
def start_scan():
    # 1. Run the Video Scanner -> Generates Analytics -> Triggers AI
    result = run_scanner()
    
    # 2. Automatically generate the new "Proposed Layout" image
    visual_result = rebuild_visual()
    
    return {
        "scan_result": result,
        "visual_rebuild": visual_result
    }