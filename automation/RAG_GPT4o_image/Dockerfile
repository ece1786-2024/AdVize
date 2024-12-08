# base image
FROM python:3.9-slim

# working directory
WORKDIR /app

# copy requirements 
# and dataset :(
COPY requirements.txt .
COPY train_250k_query_emb.csv .

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy app source
COPY app.py .

# expose port
EXPOSE 5001

# run!
CMD ["python", "app.py"]
