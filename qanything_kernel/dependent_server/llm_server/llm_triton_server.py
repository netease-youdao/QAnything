import argparse
import subprocess
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--tritonserver',
                        type=str,
                        default='/opt/tritonserver/bin/tritonserver')
    path = str(Path(__file__).parent.absolute()) + '/../all_models/gpt'
    parser.add_argument('--model_repo', type=str, default=path)
    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo):
    cmd = 'mpirun --allow-run-as-root '
    for i in range(world_size):
        cmd += ' -n 1 {} --model-repository={} --log-file=qwen.log --grpc-port=26001 --http-port=26000 --metrics-port=26002 --log-info=1 --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{}_ : '.format(
            tritonserver, model_repo, i)
    cmd += '&'
    return cmd


if __name__ == '__main__':
    args = parse_arguments()
    cmd = get_cmd(int(args.world_size), args.tritonserver, args.model_repo)
    subprocess.call(cmd, shell=True)