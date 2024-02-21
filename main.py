# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from pydantic import BaseModel
from typing import List

from mainSS import recognizeFaces

app = FastAPI()

# Add CORS middleware to allow requests from your Vue.js application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with the actual origin of your Vue.js app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    data: str

@app.post("/process_data")
async def process_data(item: Item):
    processed_data = item.data

    celeb_ids = recognizeFaces(processed_data)
    print(processed_data)
    return {"celebIds": celeb_ids}
