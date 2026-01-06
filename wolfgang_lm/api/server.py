import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from wolfgang_lm.inference.generation import WolfgangGenerator

app = FastAPI(title="Wolfgang-LM API")

# Best Practice: Allow Frontend to communicate with Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
generator = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 200


@app.on_event("startup")
async def startup_event():
    global generator
    print("Loading Model...")

    # Paths (Assume running from root)
    ckpt_path = "out-pretrain/ckpt_final.pt"
    tokenizer_path = "data_clean/tokenizer.json"

    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Interface will fail.")
        return

    generator = WolfgangGenerator(ckpt_path, tokenizer_path)
    print("Model Loaded Successfully.")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not generator:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Format Prompt
    # Simple formatting: User: ... \n Assistant: ...
    # Ideally use the same template as training
    prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            prompt += f"<|system|>\n{msg.content}\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content}\n"
    prompt += "<|assistant|>\n"

    # 2. Generate
    output_text = generator.generate(
        prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        include_prompt=False,
    )

    return {
        "id": "chatcmpl-wolfgang",
        "object": "chat.completion",
        "created": 0,
        "model": "wolfgang-lm-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "length",
            }
        ],
    }
