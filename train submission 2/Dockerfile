FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace

RUN wget https://github.com/tradertanmay/llama-recipes/raw/main/train%20submission%202/custom_dataset.py


ENV HUGGINGFACE_TOKEN="YOUR_TOKEN"
ENV HUGGINGFACE_REPO="YOUR_USERNAME/YOUR_REPO"

COPY train.py ./

CMD [ "python", "train.py"]
