from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import parking_detection
from routes import license_plate

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parking_detection.router)
app.include_router(license_plate.router)

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
