from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from pydantic import BaseModel, Field
from typing import Optional
from typing import List
from inferencing import load_and_predict

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


@app.get("/") 
async def getData():
    print("Disha")
    return {"message":"success"}

@app.post("/process_data")
async def process_data(data: MyInput):
    try:
        # Access the data sent from the frontend
        inputs = data.inputs

        # Process the data here (e.g., save to database, perform computations, etc.)
        # Replace the following print statements with your actual processing logic
        print(f"Received Inputs: {inputs}")

# Path to the saved model
        model_path = 'stress_classifier_model.h5'

    # New messages to predict stress levels for
      
        # Perform prediction using the trained model
        predicted_stress_levels = load_and_predict(model_path, inputs)

        # Display the predicted stress levels for each message
        for message, stress_level in zip(inputs, predicted_stress_levels):
            print(f"Message: {message} --> Predicted Stress Level: {stress_level}")
        predicted_stress_levels = load_and_predict(model_path, inputs)

# Calculate the average stress level
        total_stress = sum(predicted_stress_levels)
        average_stress = total_stress / len(predicted_stress_levels)

        # Store the average stress level in a variable
        final_stress = int(average_stress)

        # Display the average stress level
        print(f"Average Stress Level: {final_stress}")
        # Return a response if needed
        response = {"Final Stress Level": str(final_stress)}

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

