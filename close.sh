#!/bin/bash

# 检测支持的 Docker Compose 命令
if docker compose version &>/dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
elif docker-compose version &>/dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  echo "无法找到 'docker compose' 或 'docker-compose' 命令。"
  exit 1
fi

if [ -e /proc/version ]; then
  if grep -qi microsoft /proc/version || grep -qi MINGW /proc/version; then
    if grep -qi microsoft /proc/version; then
        echo "Running under WSL"
    else
        echo "Running under git bash"
    fi
    $DOCKER_COMPOSE_CMD -p user -f docker-compose-windows.yaml down
  else
    echo "Running under native Linux"
    $DOCKER_COMPOSE_CMD -p user -f docker-compose-linux.yaml down
  fi
else
  echo "/proc/version 文件不存在。请确认自己位于Linux或Windows的WSL环境下"
fi
