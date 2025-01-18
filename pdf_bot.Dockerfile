FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt
#RUN chmod 755 CET_4_6_edited.txt

COPY pdf_bot.py .
COPY utils.py .
COPY chains.py .
COPY sqlite.py .
COPY CET_4_6_edited.txt /app
COPY COCA_20000.txt /app
COPY nltk_data /root/nltk_data

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "./pdf_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
