import os
import time
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.2
    seed: int = None
    stream: bool = False


@app.on_event("startup")
async def startup_event():
    global generator
    print("Loading Model...")

    # Paths (Assume running from root)
    ckpt_path = "out-finetune/finetune_ckpt.pt"
    tokenizer_path = "data_clean/tokenizer.json"

    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Interface will fail.")
        return

    generator = WolfgangGenerator(ckpt_path, tokenizer_path)
    print("Model Loaded Successfully.")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": generator is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    print("\n" + "=" * 60)
    print(f"📥 REQUEST RECEIVED at {time.strftime('%H:%M:%S')}")

    if not generator:
        print("❌ Error: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Format Prompt
    user_token_id = generator.tokenizer.token_to_id("<|user|>")
    end_token_id = generator.tokenizer.token_to_id("<|endoftext|>")
    stop_tokens = [user_token_id, end_token_id]

    # <|role|>\nContent\n<|endoftext|>
    prompt = ""
    for msg in req.messages:
        content = msg.content.strip()
        if msg.role == "system":
            prompt += f"<|system|>\n{content}\n<|endoftext|>"
        elif msg.role == "user":
            prompt += f"<|user|>\n{content}\n<|endoftext|>"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{content}\n<|endoftext|>"

    # Prompt for the new generation
    prompt += "<|assistant|>\n"

    print(f"\n📝 PROMPT ({len(prompt)} chars):")
    print("-" * 20)
    print(prompt.strip())
    print("-" * 20)
    print("🤖 Generating...", end="", flush=True)

    start_time = time.perf_counter()

    # 2. Generator Logic
    if req.stream:
        return StreamingResponse(
            stream_generator(prompt, req, stop_tokens), media_type="text/event-stream"
        )
    else:
        # Non-streaming generation
        output_text = generator.generate(
            prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            include_prompt=False,
            stop_tokens=stop_tokens,
        )

        duration = time.perf_counter() - start_time
        print(f" Done! ({duration:.2f}s)")
        print(f"\n📤 OUTPUT ({len(output_text)} chars):")
        print("-" * 20)
        print(output_text.strip())
        print("-" * 20)
        print("=" * 60 + "\n")

        return {
            "id": "chatcmpl-wolfgang",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "wolfgang-lm-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "length",
                }
            ],
        }


def stream_generator(prompt, req, stop_tokens):

    # Send initial chunk with role
    chunk_data = {
        "id": "chatcmpl-wolfgang",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "wolfgang-lm-v1",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(chunk_data)}\n\n"

    full_response = ""

    # Stream content
    for text_chunk in generator.generate(
        prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        seed=req.seed,
        include_prompt=False,
        stop_tokens=stop_tokens,
        stream=True,
    ):
        full_response += text_chunk
        chunk_data = {
            "id": "chatcmpl-wolfgang",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "wolfgang-lm-v1",
            "choices": [
                {"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

    # Log the full output
    print(f"\n📤 OUTPUT ({len(full_response)} chars):")
    print("-" * 20)
    print(full_response.strip())
    print("-" * 20)
    print("=" * 60 + "\n")

    # Final 'done' chunk to signify stop
    chunk_data = {
        "id": "chatcmpl-wolfgang",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "wolfgang-lm-v1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk_data)}\n\n"
    yield "data: [DONE]\n\n"
