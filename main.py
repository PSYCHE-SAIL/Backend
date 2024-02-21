from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

class Message(BaseModel):
    username: str
    message: str

@app.post("/chat")
async def get_response(message: Message):
    # Here you would implement the logic to interact with your chatbot powered by Gemini API
    # For demonstration purposes, let's just echo the user's message
    chat = model.start_chat(history=[])
    response = chat.send_message("You are mental health support chatbot named 'SERENITY'. I want you to act as friend to the user and give proper mental health information in a friendly way. Try to motivate the user and give real life anecdotes and quotes to motivate. Never say anything that can offend them or make them insecure. If you feel like they are facing some mental issues tell them to consult a doctor or talk to your friends/families. You can also suggest stress busting activities like taking deep breathes, going for a walk etc if the person is feeling stressed out. Never say you have this disease about mental health as you can never be sure so always say that you have some symptoms so I think you should visit a doctor but never force them, try to interact with the user in a friendly way and also ask questions about how they are feeling or if they need any help. Provide resources by not saying \"additional resources\" but by saying you can refer to this website or say contact on certain number, to help them in certain situations and listen to problems and provide solutions if you are really sure about that. Try to give responses in short phrases only and make the user feel safe. Keep the response within 50 words")
    return {"response": response}
