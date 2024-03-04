from setuptools import setup, find_packages

# 读取requirements.txt文件内容
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qanything',
    version='0.1',
    # packages=find_packages(include=['qanything_kernel*', 'third_party/FastChat/fastchat*']),
    packages=find_packages(),
    install_requires=required,
    package_data={
        # 如果有需要包含在内的数据文件，可以在这里指定
    },
    include_package_data=True,  # 包含所有的非.py文件
    entry_points={
        # 'console_scripts': [
        #     'fastchat-controller=fastchat.serve.controller:main',
        #     'fastchat-api-server=fastchat.serve.openai_api_server:main',
        #     'fastchat-vllm-worker=fastchat.serve.vllm_worker:main',
        # ],
    },
    description='A QA server',
    author='Your Name',
    author_email='your.email@example.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='qa server',
)
