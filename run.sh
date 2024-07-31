#!/bin/bash

docker stop solo-ai-docker  || true  &&  docker remove solo-ai-docker  || true && docker build -t solo-server . && docker run -d --gpus "device=all" -p 80:8000 --name solo-ai-docker solo-server