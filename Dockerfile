
FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install fastapi uvicorn transformers nest_asyncio asyncio torch pydantic bitsandbytes accelerate

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
