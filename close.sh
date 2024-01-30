#!/bin/bash

if [ -e /proc/version ]; then
  if grep -qi microsoft /proc/version || grep -qi MINGW /proc/version; then
    if grep -qi microsoft /proc/version; then
        echo "Running under WSL"
    else
        echo "Running under git bash"
    fi
    docker-compose -p user -f docker-compose-windows.yaml down
  else
    echo "Running under native Linux"
    docker-compose -p user -f docker-compose-linux.yaml down
  fi
else
  echo "/proc/version 文件不存在。请确认自己位于Linux或Windows的WSL环境下"
fi
