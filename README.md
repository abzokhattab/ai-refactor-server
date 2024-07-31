

# AI-Refactor Server 

This server hosts a large language model (LLM) using FastAPI. The model is run within a Docker container and can be accessed via HTTP requests.

## Setup and Installation 

### Prerequisites 

- Docker

- Python

- CUDA-enabled GPU (optional, but recommended for performance)

### Local Deployment 
To build and run the server locally within a Docker container, use the provided `run.sh` script.

```bash
./run.sh
```

### Server Configuration 
The server is configured to load the `google/codegemma-1.1-7b-it` model from Hugging Face. Ensure that you have access to the model by setting up a Hugging Face account and obtaining an API token. For ease of use, an access token is provided in the script.
### Model Details 
 
- **Model Name:**  `google/codegemma-1.1-7b-it`
 
- **Framework:**  PyTorch

## API Endpoints 
POST `/chat`
Initiate a chat session with the model.
**Request:**  
- `snippet_id` (string): A unique identifier for the chat session.
 
- `messages` (list of objects): 
  - `role` (string): Either "user" or "model".
 
  - `content` (string): The text content of the message.
**Response:**  
- `response` (string): The model's generated response.

### Example Request 


```json
{
  "snippet_id": "refactor123",
  "messages": [
    {"role": "user", "content": "Please refactor this code:\n\nfor i in range(len(my_list)):\n    print(my_list[i])"}
  ]
}
```

### Example Response 


```json
{
  "response": "Refactored code:\n\nfor item in my_list:\n    print(item)"
}
```

## Notes 

- The server uses an in-memory cache for storing user chat history. This cache is cleared periodically every 10 minutes to optimize memory usage.
 
- The server is set to run on `0.0.0.0` at port `8000`.

- Model evaluation is performed on the client side, not on the server.

## Running the Server 

To start the server manually, run:


```bash
python main.py
```
Alternatively, the server can be launched via `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
