from setuptools import setup, find_packages
import platform
import os
import requests
from setuptools.command.install import install
import pkg_resources

root_path = os.path.dirname(os.path.abspath(__file__))
qanything_path = os.path.join(root_path, 'qanything_kernel')

def check_onnx_version(version):
    try:
        onnx_version = pkg_resources.get_distribution("onnxruntime-gpu").version
        if onnx_version == version:
            print(f"onnxruntime-gpu {version} 已经安装。")
            return True
        else:
            print(f"onnxruntime-gpu 版本过低，当前版本为 {onnx_version}，需要安装 {version} 版本。")
            return False
    except pkg_resources.DistributionNotFound:
        print(f"onnxruntime-gpu {version} 未安装。")
    return False


class CustomInstallCommand(install):
    """自定义的安装命令"""

    def run(self):
        # 先执行原来的install命令
        install.run(self)

        from tqdm import tqdm
        import torch
        # 执行你的Python代码

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            with open(filename, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")

        cuda_version = torch.version.cuda
        if cuda_version is None:
            raise ValueError("CUDA is not installed.")
        elif float(cuda_version) < 12:
            raise ValueError("CUDA version must be 12.0 or higher.")
        python_version = platform.python_version()
        python3_version = python_version.split('.')[1]
        os_system = platform.system()
        system_name = None
        if os_system == "Windows":
            system_name = 'win_amd64'
        elif os_system == "Linux":
            system_name = 'manylinux_2_28_x86_64'
        if system_name is not None:
            if not check_onnx_version("1.17.1"):
                download_url = f"https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.17.1/onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl/content"
                whl_name = f'onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl'
                download_file(download_url, os.path.join(qanything_path, whl_name))
                os.system(f"pip install {whl_name}")
        else:
            raise ValueError(f"Unsupported system: {os_system}")

# 读取requirements.txt文件内容
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qanything',
    version='0.1',
    packages=find_packages(),
    install_requires=required,
    python_requires='>=3.10, <=3.12',  # 这里限定了Python的版本
    include_package_data=True,  # 包含所有的非.py文件
    entry_points={
        'console_scripts': [
            'qanything-server=qanything_kernel.qanything_server.sanic_api:main'
        ],
    },
    cmdclass={
        'install': CustomInstallCommand,
    },
    description='A QA server',
    author='Junxiong Liu',
    author_email='xixihahaliu01@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='qanything server',
)

