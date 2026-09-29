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
#
# Remote build (Agent Office / agi build service):
#   agi build submit . --dest <registry>/labts:<tag> -o json
#
# Mirrors default to pypi.org: measured from the agi build service it is
# ~700x faster than mirrors.aliyun.com (340 MB/s vs 0.5 MB/s). For CN-only
# hosts, override with e.g.:
#   --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
#   --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
# (pypi.msh.team/simple is NOT a PyPI proxy — internal packages only.)

ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
FROM ${BASE_IMAGE}

# Torch wheel flavor:
#   cuda (default) — plain PyPI torch wheel with the bundled CUDA runtime;
#                    works on GPU sandboxes AND on CPU-only hosts.
#   cpu            — keep the frozen +cpu wheel from requirements.txt (smaller).
ARG TORCH_FLAVOR=cuda
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_EXTRA_INDEX_URL=

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

# playground/requirements.txt starts with `-e .`, so the install needs the
# package metadata and sources present; copy them before the install layer to
# keep pip's layer cache useful.
COPY pyproject.toml README.md playground/requirements.txt ./
COPY sktime ./sktime
COPY playground ./playground

# pycatch22 ships no wheel for this platform and compiles a C extension;
# install a toolchain for the build and remove it afterwards.
# For TORCH_FLAVOR=cuda, swap the pinned +cpu wheel for the plain PyPI wheel
# (=== excludes the +cpu local version) and drop the pytorch CPU index.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && if [ "$TORCH_FLAVOR" = "cuda" ]; then \
         sed -i -e '\#--extra-index-url https://download.pytorch.org/whl/cpu#d' \
                -e 's|^torch==2.14.0+cpu$|torch===2.14.0|' playground/requirements.txt; \
       fi \
    && pip install -r playground/requirements.txt \
    && apt-get purge -y gcc libc6-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8765

# The server defaults to 127.0.0.1; inside a container it must bind 0.0.0.0.
CMD ["python", "playground/server.py", "--host", "0.0.0.0", "--port", "8765"]
