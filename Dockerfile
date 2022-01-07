FROM pytorch/pytorch

RUN apt update && apt upgrade -y

RUN pip install tensorboard

COPY $PWD /content/nlp

WORKDIR /content

ENTRYPOINT ["python", "-m", "nlp"]
