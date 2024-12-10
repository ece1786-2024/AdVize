FROM python:3.9-slim

# Set working directory:
WORKDIR /app

# Copy file(s) to the working directory:
COPY default_algo.py /app/

# Run `python algorithm.py` by default:
CMD ["python", "default_algo.py"]
# This default behaviour can be overriden by:
# `docker run algorithm-image "search_query_string" "/path/to/ad_dataset.csv"`