# main.py
import os
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
import asyncio
import base64
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from openai import OpenAI
import uvicorn
from pyngrok import ngrok
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Voice AI API",
    description="An API for handling voice-based interactions with AI",
    version="1.0.0"
)

# Qwen Client Setup
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Store per-connection chat history (in-memory)
conversation_histories = {}

class Message(BaseModel):
    role: str
    content: List[Dict]

class ConversationHistory(BaseModel):
    messages: List[Message]

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint for health check
    """
    return {"status": "healthy", "message": "Voice AI API is running"}

@app.get("/history/{connection_id}", response_model=ConversationHistory, tags=["Conversation"])
async def get_conversation_history(connection_id: str):
    """
    Get the conversation history for a specific connection
    """
    if connection_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Connection history not found")
    return {"messages": conversation_histories[connection_id]}

def get_audio_response(audio_path, history):
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
        b64_audio = base64.b64encode(audio_data).decode("utf-8")

    history.append({
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/wav;base64,{b64_audio}",
                    "format": "wav",
                },
            },
            {"type": "text", "text": "Please respond to this audio."},
        ],
    })

    completion = client.chat.completions.create(
        model="qwen2.5-omni-7b",
        messages=history,
        modalities=["audio"],
        audio={"voice": "Chelsie", "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
    )

    audio_content = b""
    last_assistant_text = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.type == "audio":
            audio_content += base64.b64decode(chunk.choices[0].delta.audio)
        elif chunk.choices and chunk.choices[0].delta.type == "text":
            last_assistant_text += chunk.choices[0].delta.text

    # Append assistant message to history
    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": last_assistant_text}]
    })

    # Convert to 8000 Hz mono PCM for Twilio compatibility
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        AudioSegment(
            audio_content, sample_width=2, frame_rate=24000, channels=1
        ).export(temp_audio.name, format="wav", parameters=["-ar", "8000"])
        converted_audio = open(temp_audio.name, "rb").read()

    return converted_audio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio communication
    """
    await websocket.accept()
    audio_bytes = b""
    # Generate a unique ID for this connection
    connection_id = str(id(websocket))
    conversation_histories[connection_id] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an interviewer. You will always say hello to the caller and ask their name before continuing with some random interview question."
                }
            ]
        }
    ]

    try:
        while True:
            data = await websocket.receive_text()
            message = eval(data)
            if message['event'] == 'media':
                chunk = base64.b64decode(message['media']['payload'])
                audio_bytes += chunk

            if message['event'] == 'stop':
                # Save received audio to file
                with NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    AudioSegment(
                        audio_bytes, sample_width=2, frame_rate=8000, channels=1
                    ).export(f.name, format="wav")
                    audio_response = get_audio_response(f.name, conversation_histories[connection_id])

                b64_audio = base64.b64encode(audio_response).decode("utf-8")

                await websocket.send_text(f"{{\"event\": \"media\", \"media\": {{\"payload\": \"{b64_audio}\"}}}}")
                await websocket.send_text("{\"event\": \"stop\"}")
                audio_bytes = b""  # Reset for next round (continue conversation)

    except Exception as e:
        conversation_histories.pop(connection_id, None)
        await websocket.close(code=1000)

@app.on_event("startup")
async def startup():
    """
    Initialize ngrok tunnel on startup if NGROK_AUTH_TOKEN is provided
    """
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        tunnel = ngrok.connect(8000)
        print(f"Ngrok tunnel established at: {tunnel.public_url}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
