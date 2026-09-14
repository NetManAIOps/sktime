# TSBox Sandbox Playground image.
#
# Builds the full Playground environment (sktime editable install + the frozen
# soft-dependency set from playground/requirements.txt) and serves the web UI.
#
# Build:   sudo docker build -t tsbox-playground:latest .
# Web UI:  sudo docker run --rm -p 8765:8765 tsbox-playground:latest
# LabTS CLI (override the command):
#   sudo docker run --rm tsbox-playground:latest \
#     python playground/labts.py catalog --compact
#   sudo docker run --rm tsbox-playground:latest \
#     python playground/labts.py run --spec '{"task":"forecasting"}' --compact

# docker.io is unreachable from this host; the daocloud mirror is used instead.
FROM docker.m.daocloud.io/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=http://pypi.ksyun.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.ksyun.cn

WORKDIR /app

# playground/requirements.txt starts with `-e .`, so the install needs the
# package metadata and sources present; copy them before the install layer to
# keep pip's layer cache useful.
COPY pyproject.toml README.md playground/requirements.txt ./
COPY sktime ./sktime
COPY playground ./playground

# pycatch22 ships no wheel for this platform and compiles a C extension;
# install a toolchain for the build and remove it afterwards.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && pip install -r playground/requirements.txt \
    && apt-get purge -y gcc libc6-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8765

# The server defaults to 127.0.0.1; inside a container it must bind 0.0.0.0.
CMD ["python", "playground/server.py", "--host", "0.0.0.0", "--port", "8765"]
