FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /operator
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
 && python3 -m pip install --no-cache-dir --no-deps git+https://github.com/tercen/tercen_python_client@0.12.14
COPY main.py ./
COPY oneday_stats ./oneday_stats

# Tercen runs operator containers as UID 1000.
RUN useradd -u 1000 -m tercen && chown -R 1000:1000 /operator
USER 1000

ENV TERCEN_SERVICE_URI=https://tercen.com
ENTRYPOINT ["python3", "main.py"]
CMD ["--taskId", "someid", "--serviceUri", "https://tercen.com", "--token", "sometoken"]
