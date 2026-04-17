FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /Helmet

COPY Requirements.txt .

RUN pip install -r Requirements.txt
RUN pip install torchmetrics
COPY src ./src

CMD ["python","-m","src.train"]