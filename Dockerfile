FROM python:3.10.3-slim
ENV PYTHONUNBUFFERED 1
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
RUN python main.py
CMD ["flask", "run", "--host=0.0.0.0"]