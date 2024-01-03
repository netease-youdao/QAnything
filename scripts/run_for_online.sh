#!/bin/bash
nohup nginx -g "daemon off;" 1>nginx.log 2>&1 &
# start llm server
cd /workspace/qanything_local/front_end
npm install
npm install -g http-server
npm run build
cd dist/
nohup http-server ./  -p 5002 --cors --name qAnything > http_server.log 2>&1 &
echo "The front-end service is ready!"

cd /workspace/qanything_local
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready!"

nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready!"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py --mode online > api.log 2>&1 &
echo "The qanything service is ready!"

# 保持容器运行
while true; do
  sleep 2
done
