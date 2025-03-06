from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory
from dotenv import load_dotenv
from src.chain import Chaining
from google.cloud import firestore
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
#Load the environment variable for the api keys
load_dotenv()

app = FastAPI()

#structured class for easy parameters
class ChatRequest(BaseModel):
    query: str
    session_id: str
    model: str
    temperature: float
    top_p: float

#Use the post method of the fastapi
@app.post("/chat")
async def run(request: ChatRequest):

    #Initialize the firebase database
    PROJECT_ID = os.getenv("PROJECT_ID")
    SESSION_ID = request.session_id
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    client = firestore.Client(PROJECT_ID)
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #Model Selection for the user
    if "gemini" in request.model.lower():
        model = ChatGoogleGenerativeAI(
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p
        )
    else:
        model = ChatOpenAI(
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p
            )
    
    #Initialize the firestore history
    chat_history = FirestoreChatMessageHistory(
        session_id=SESSION_ID,
        collection=COLLECTION_NAME,
        client=client
    )
    #Add the user query to the firestore
    chat_history.add_user_message(request.query)

    #Call the chaining class
    ch = Chaining(model, chat_history)
    
    #Get the final result
    result = ch.combine_branch(request.query, chat_history)

    #Add the AI message to the firestore
    chat_history.add_ai_message(result['response'])
    return {
        "response": result['response'], 
        "model": request.model, 
        "session_id":SESSION_ID
        }

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Fetch chat history for a given session ID."""
    print("[][][][][][]", session_id)
    PROJECT_ID = os.getenv("PROJECT_ID")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    client = firestore.Client(PROJECT_ID)

    chat_history = FirestoreChatMessageHistory(
        session_id=session_id,
        collection=COLLECTION_NAME,
        client=client
    )
    return {"session_id": session_id, "history": chat_history.messages}
 
        
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,host="localhost",port=8000)