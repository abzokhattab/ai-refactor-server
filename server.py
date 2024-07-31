from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import logging
from pydantic import BaseModel
from typing import List
import os
import torch
import asyncio


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for user history for future work.
user_history = {}

# Hugging Face model setup
login(token="ENTER_YOUR_TOKEN")
model_name = "google/codegemma-1.1-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
torch.cuda.empty_cache()


try:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quantization_config
    )
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

logger.info(f"Model device: {device}, CUDA available: {torch.cuda.is_available()}")

class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str

class ChatRequest(BaseModel):
    snippet_id: str
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("Received chat request for snippet_id: %s", request.snippet_id)
    try:
        if request.snippet_id in user_history:
            history = user_history[request.snippet_id]['history']
            attempts = user_history[request.snippet_id]['attempts'] + 1
            if attempts > 30:
                raise HTTPException(status_code=400, detail="Maximum attempts reached.")
        else:
            history = []
            attempts = 1
        
        # Append new messages to the history
        history.extend([message.dict() for message in request.messages])

        # Apply chat template to construct the prompt
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                inputs,
                max_new_tokens=1024,  
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.0,
                temperature=0.7, 
            )

        torch.cuda.empty_cache()
        
        # Slice the generated tokens to remove the prompt part
        generated_text = tokenizer.batch_decode(gen_tokens[:, inputs.shape[-1]:], skip_special_tokens=True)[0]
        history.append({"role": "model", "content": generated_text})

        # Save updated history to the in-memory storage
        user_history[request.snippet_id] = {
            'history': history,
            'attempts': attempts
        }

        
        
        return ChatResponse(response=generated_text)
    except Exception as e:
        logger.error("Error during chat: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    import sys
    import nest_asyncio

    def run_server():
        logger.info("Starting server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)

    if asyncio.get_event_loop().is_running():
        nest_asyncio.apply()
        run_server()
    else:
        run_server()
