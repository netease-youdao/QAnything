#!/bin/bash

start_time=$(date +%s)  # 璁板綍寮€濮嬫椂闂?# 妫€鏌ユā鍨嬫枃浠跺す鏄惁瀛樺湪
check_folder_existence() {
  if [ ! -d "/workspace/qanything_local/models/$1" ]; then
    echo "The $1 folder does not exist under /workspace/qanything_local/models/. Please check your setup."
    echo "鍦?workspace/qanything_local/models/涓嬩笉瀛樺湪$1鏂囦欢澶广€傝妫€鏌ユ偍鐨勮缃€?
    exit 1
  fi
}

echo "Checking model directories..."
echo "妫€鏌ユā鍨嬬洰褰?.."
check_folder_existence "base"
check_folder_existence "embed"
check_folder_existence "rerank"
echo "Model directories check passed. (0/7)"
echo "妯″瀷璺緞妫€鏌ラ€氳繃. (0/7)"

# start llm server
nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble --http-port=10000 --grpc-port=10001 --metrics-port=10002 > /model_repos/QAEnsemble/QAEnsemble.log 2>&1 &

cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm transfer service is ready! (1/7)"
echo "澶фā鍨嬩腑杞湇鍔″凡灏辩华! (1/7)"

cd /workspace/qanything_local
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready! (2/7)"
echo "rerank鏈嶅姟宸插氨缁? (2/7)"

nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready! (3/7)"
echo "OCR鏈嶅姟宸插氨缁? (3/7)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything backend service is ready! (4/7)"
echo "qanything鍚庣鏈嶅姟宸插氨缁? (4/7)"

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 璁＄畻缁忚繃鐨勬椂闂达紙绉掞級
echo "Time elapsed: ${elapsed} seconds."
echo "宸茶€楁椂: ${elapsed} 绉?"

cd /workspace/qanything_local/front_end
# 瀹夎渚濊禆
nohup npm i -g yarn > npm_install_yarn.log 2>&1
nohup yarn > npm_install.log 2>&1
if [ $? -eq 0 ]; then
    echo "npm install completed, starting front_end development service... (5/7)"
    echo "npm install 瀹屾垚锛屾鍦ㄥ惎鍔ㄥ墠绔湇鍔?.. (5/7)"
    # 瀹夎瀹屾垚鍚庯紝鍚姩鍓嶇鏈嶅姟
    nohup yarn dev > npm_run_dev.log 2>&1 &
    DEV_SERVER_PID=$!
    # echo "鍓嶇鏈嶅姟杩涚▼ID: $DEV_SERVER_PID"
    while ! grep -q "ready" npm_run_dev.log; do
        echo "Waiting for the front-end service to start..."
        echo "绛夊緟鍓嶇鏈嶅姟鍚姩..."
        sleep 5
    done

    echo "The front-end service is ready!...(6/7)"
    echo "鍓嶇鏈嶅姟宸插氨缁?...(6/7)"
else
    echo "npm install failed, please check the npm_install.log log file."
    echo "npm install 澶辫触锛岃妫€鏌?npm_install.log 鏃ュ織鏂囦欢銆?
    exit 1
fi

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 璁＄畻缁忚繃鐨勬椂闂达紙绉掞級
echo "Time elapsed: ${elapsed} seconds."
echo "宸茶€楁椂: ${elapsed} 绉?"

while true; do
  response=$(curl -s -w "%{http_code}" http://localhost:10000/v2/health/ready -o /dev/null)
  if [ $response -eq 200 ]; then
    echo "The triton service is ready!, now you can use the qanything service. (7/7)"
    echo "Triton鏈嶅姟宸插噯澶囧氨缁紒鐜板湪鎮ㄥ彲浠ヤ娇鐢╭anything鏈嶅姟銆傦紙7/7锛?
    break
  else
    echo "The triton service is starting up, it can be long... you have time to make a coffee :)"
    echo "Triton鏈嶅姟姝ｅ湪鍚姩锛屽彲鑳介渶瑕佷竴娈垫椂闂?..浣犳湁鏃堕棿鍘诲啿鏉挅鍟?:)"
    sleep 5
  fi
done

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 璁＄畻缁忚繃鐨勬椂闂达紙绉掞級
echo "Time elapsed: ${elapsed} seconds."
echo "宸茶€楁椂: ${elapsed} 绉?"

# 淇濇寔瀹瑰櫒杩愯
while true; do
  sleep 2
done

