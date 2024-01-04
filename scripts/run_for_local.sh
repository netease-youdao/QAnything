#!/bin/bash
nohup nginx -g "daemon off;" 1>nginx.log 2>&1 &
# start llm server
nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble --http-port=10000 --grpc-port=10001 --metrics-port=10002 > /model_repos/QAEnsemble/QAEnsemble.log 2>&1 &

cd /workspace/qanything_local/front_end
npm install
npm run build
echo "The front-end service is ready!"

echo "The triton service is starting up, he can be long... you have time to make a coffee :)"
while true; do
  response=$(curl -s -w "%{http_code}" http://localhost:10000/v2/health/ready -o /dev/null)
  if [ $response -eq 200 ]; then
    echo "The triton service is ready!, now start the qanything service"
    break
  fi
  sleep 1
done

cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm service is ready!"

cd /workspace/qanything_local
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready!"

nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready!"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything service is ready!"

# 保持容器运行
while true; do
  sleep 2
done
