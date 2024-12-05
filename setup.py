from setuptools import setup, find_packages

setup(
    name="ola_vlm",
    version="1.0.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.2.0", "torchvision==0.17.0",
        "tokenizers==0.19.1", "sentencepiece==0.1.99", "shortuuid",
        "peft", "bitsandbytes", "open_clip_torch", "diffdist",
        "pydantic", "markdown2[all]", "numpy==1.26.2",
        "gradio==4.16.0", "gradio_client==0.8.1", "huggingface_hub",
        "requests", "httpx==0.24.0", "uvicorn", "fastapi",
        "einops==0.6.1", "einops-exts==0.0.4", "timm==1.0.8",
        "diffusers===0.27.2", "protobuf", "accelerate==0.27.2"
    ],
    extras_require={
        "train": ["deepspeed==0.12.6", "ninja", "wandb"],
        "eval": ["seaborn", "sty", "tabulate", "spacy", "word2number", "inflect"],
        "demo": ["pydantic==2.8.2", "pydantic-core==2.20.1", "fastapi==0.111.0"],
        "build": ["build", "twine"]
    },
    url="https://praeclarumjj3.github.io/ola_vlm",
    project_urls={
        "Bug Tracker": "https://github.com/SHI-Labs/OLA-VLM/issues"
    },
    packages=find_packages(exclude=["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]),
    include_package_data=True,
)
