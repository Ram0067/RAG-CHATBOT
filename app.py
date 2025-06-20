from fastapi import FastAPI, Request
from pydantic import BaseModel
from generate_response import generate_response

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(data: Query):
    user_query = data.question
    answer = generate_response(user_query)
    return {"answer": answer}
