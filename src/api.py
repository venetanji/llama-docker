from fastapi import FastAPI, Request
import uvicorn
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from config import Redis
import asyncio
from fastapi import WebSocket, WebSocketDisconnect




redis = Redis()


load_dotenv()



api = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

class ChatCompletion(BaseModel):
    prompt: str
   
manager = ConnectionManager()
 

@api.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    redis_connection = await redis.create_connection()
    try:
        while True:
            data = await websocket.receive_text()
            await redis_connection.xadd('chat', {'role':'user', 'message': data}) 
            
            completion = await redis_connection.xread({'chat': '$'}, None, 0)
            print(completion)
            role = completion[0][1][0][1][b'role'].decode('utf-8')
            if  role == 'assistant':
                msg = completion[0][1][0][1][b'message'].decode('utf-8')
                await manager.send_personal_message(msg, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
 


if __name__ == "__main__":
    if os.environ.get('APP_ENV') == "development":
        uvicorn.run("api:api", host="0.0.0.0", port=3500,
                    workers=4, reload=True)
    else:
      pass
