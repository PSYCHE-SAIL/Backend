from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from pydantic import BaseModel, Field
from typing import Optional
from typing import List
from inferencing import load_and_predict
from VectorSearch import getChatID

app = FastAPI()

# Add CORS middleware to allow requests from your Vue.js application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with the actual origin of your Vue.js app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MyInput(BaseModel):
    inputs: List[str]


@app.post("/getChatRoomID") 
async def getChatroomId(data: MyInput):
    inputs = data.inputs
    chatID = getChatID(inputs)
    return chatID

@app.post("/process_data")
async def process_data(data: MyInput):
    try: 
        inputs = data.inputs

        print(f"Received Inputs: {inputs}")

        model_path = 'stress_classifier_model.h5'

        predicted_stress_levels = load_and_predict(model_path, inputs)

        for message, stress_level in zip(inputs, predicted_stress_levels):
            print(f"Message: {message} --> Predicted Stress Level: {stress_level}")
        predicted_stress_levels = load_and_predict(model_path, inputs)

        total_stress = sum(predicted_stress_levels)
        average_stress = total_stress / len(predicted_stress_levels)

        final_stress = int(average_stress)

        print(f"Average Stress Level: {final_stress}")

        response = {"Final Stress Level": str(final_stress)}

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

