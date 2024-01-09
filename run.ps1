# @echo off
# 转换文件格式
(Get-Content -Raw -Path .\scripts\run_for_local.sh) -replace "`r`n", "`n" | Set-Content -Path .\scripts\run_for_local_unix.sh -Encoding utf8

# 判断Docker容器是否启动
$dockerStatus = Get-Service | Where-Object {$_.DisplayName -like '*Docker*'} | Select-Object -ExpandProperty Status

if ($dockerStatus -eq 'Running') {
    Write-Host "Docker 客户端已启动."
} else {
    Write-Host "Docker 客户端未启动."
    return
}

# 启动 Docker 容器
docker-compose -f docker-compose-windows.yaml up qanything_local
