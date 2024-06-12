from setuptools import setup


# 读取requirements.txt文件内容
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qanything',
    version='0.2',
    packages=['qanything'],
    package_dir={'qanything': 'qanything_kernel'},  # 指定源代码目录
    install_requires=required,
    python_requires='>=3.10, <=3.12',  # 这里限定了Python的版本
    include_package_data=True,  # 包含所有的非.py文件
    package_data={
        "qanything_kernel": ["nltk_data/taggers/averaged_perceptron_tagger/*", "nltk_data/tokenizers/punkt/*", "nltk_data/tokenizers/punkt/PY3/*"],
    },
    # entry_points={
    #     'console_scripts': [
    #         'qanything-server=qanything_kernel.qanything_server.sanic_api:main'
    #     ],
    # },
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

