FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install --no-cache-dir git+https://github.com/tradertanmay/llama-recipes.git@a7504a2e99ec27c429223b17ed851ad56c24c909

WORKDIR /workspace

RUN wget https://github.com/tradertanmay/llama-recipes/raw/main/g/custom_dataset.py



ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="TanmaySah/mv2mm"

COPY train.py ./

CMD [ "python", "train.py"]
