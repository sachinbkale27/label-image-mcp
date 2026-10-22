#!/bin/bash
# AWS Lambda entry point (backend/template.yaml). The Lambda Web Adapter layer
# (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap) starts this server and forwards each request to port 8080.
export PYTHONPATH="/var/task/src:/var/task:${PYTHONPATH}"
exec python3 -m label_image_mcp.server --http --hosted --host 127.0.0.1 --port 8080
