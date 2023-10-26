FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/tradertanmay/llama-recipes.git@main


WORKDIR /workspace

RUN wget https://raw.githubusercontent.com/tradertanmay/llama-recipes/main/train%20submission%201/custom_dataset.py


ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="TanmaySah/lmbase"

COPY train.py ./

CMD [ "python", "train.py"]
