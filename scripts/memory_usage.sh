#!/bin/bash

processes=("rerank_server.py" "embedding_server.py" "ocr_server.py" "insert_files_server.py" "sanic_api.py" "pdf_parser_server.py")
# processes=("embedding_server.py" "insert_files_server.py" "sanic_api.py" "pdf_parser_server.py")

get_memory_usage() {
    total_mem=0
    output=""
    for proc in "${processes[@]}"
    do
        pid=$(pgrep -f "$proc")
        if [ ! -z "$pid" ]
        then
            mem=$(ps --ppid $pid -p $pid --no-headers -o rss | awk '{sum+=$1} END {printf "%.2f", sum/1024/1024}')
            total_mem=$(awk "BEGIN {print $total_mem + $mem}")
            output+="$proc:$mem "
        fi
    done
    echo "$output$total_mem"
}

truncate_name() {
    name=$1
    if [ ${#name} -gt 15 ]; then
        echo "${name:0:12}..."
    else
        printf "%-15s" "$name"
    fi
}

print_header() {
    printf "%-20s" "时间"
    for proc in "${processes[@]}"; do
        printf "%-20s" "$(truncate_name "$proc")"
    done
    printf "%-25s\n" "总内存(GB)"
    echo "--------------------------------------------------------------------------------------------"
}

if [ "$1" == "long" ]; then
    log_file="memory_usage_log_$(date +%Y%m%d%H%M%S).txt"
    echo "长期记录模式启动，日志文件: $log_file"

    print_header | tee "$log_file"

    initial_mem=()
    first_total=0
    count=0

    while true; do
        timestamp=$(date "+%Y%m%d%H%M%S")
        mem_data=$(get_memory_usage)
        total_mem=$(echo $mem_data | awk '{print $NF}')

        printf "%-20s" "$timestamp" | tee -a "$log_file"

        i=0
        for proc in "${processes[@]}"; do
            current_mem=$(echo $mem_data | awk -v proc="$proc:" '{for(i=1;i<=NF;i++) if($i~proc) print $i}' | cut -d':' -f2)
            if [ $count -eq 0 ]; then
                initial_mem[$i]=$current_mem
                printf "%-20s" "${current_mem}GB" | tee -a "$log_file"
            else
                growth=$(awk "BEGIN {printf \"%.2f\", ($current_mem - ${initial_mem[$i]}) * 100 / ${initial_mem[$i]}}")
                printf "%-20s" "${current_mem}GB (${growth}%)" | tee -a "$log_file"
            fi
            i=$((i+1))
        done

        if [ $count -eq 0 ]; then
            first_total=$total_mem
            printf "%-25s\n" "${total_mem}GB" | tee -a "$log_file"
        else
            total_growth=$(awk "BEGIN {printf \"%.2f\", ($total_mem - $first_total) * 100 / $first_total}")
            printf "%-25s\n" "${total_mem}GB (${total_growth}%)" | tee -a "$log_file"
        fi


        count=$((count+1))
        sleep 10
    done
else
    echo "一次性模式..."
    print_header
    mem_data=$(get_memory_usage)
    total_mem=$(echo $mem_data | awk '{print $NF}')
    timestamp=$(date "+%Y%m%d%H%M%S")
    printf "%-20s" "$timestamp"
    for proc in "${processes[@]}"; do
        current_mem=$(echo $mem_data | awk -v proc="$proc:" '{for(i=1;i<=NF;i++) if($i~proc) print $i}' | cut -d':' -f2)
        printf "%-20s" "${current_mem}GB"
    done
    printf "%-25s\n" "${total_mem}GB"
fi
