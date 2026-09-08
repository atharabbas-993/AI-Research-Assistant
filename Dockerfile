# Dockerfile

# Start from an official, lightweight Python base image.
# "slim" variants have fewer pre-installed OS packages — smaller image,
# faster builds, smaller attack surface. Pin the exact version (not
# just "python:3.11") so builds are reproducible over time.
FROM python:3.11-slim

# Set the working directory INSIDE the container — all subsequent
# commands (COPY, RUN) happen relative to this path.
WORKDIR /app

# Copy ONLY requirements.txt first, before the rest of the code.
# Why separately? Docker caches each step. If only your source code
# changes (not dependencies), Docker reuses the cached "pip install"
# layer instead of re-running it — much faster rebuilds during development.
COPY requirements.txt .

# Install dependencies. --no-cache-dir keeps the image smaller by not
# storing pip's download cache inside the container.
RUN pip install --no-cache-dir -r requirements.txt

# NOW copy the rest of the application code.
COPY . .

# Create the folders our app expects at runtime (in case they're
# not already present when building from a fresh clone)
RUN mkdir -p data/raw_pdfs data/chroma_db logs

# Document which port the app listens on (informational — doesn't
# actually publish the port; that happens at `docker run` time)
EXPOSE 8000

# The command that runs when the container starts.
# --host 0.0.0.0 is REQUIRED — 127.0.0.1 would only accept connections
# from inside the container itself, making it unreachable from outside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]