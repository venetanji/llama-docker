FROM pytorch/pytorch:latest
COPY requirements.txt .
RUN pip install -r requirements.txt