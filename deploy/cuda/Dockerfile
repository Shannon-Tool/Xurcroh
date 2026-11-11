FROM python:3.12-bookworm AS build

ARG RUST_VERSION=1.99.1
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:$PATH
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && pip install --no-cache-dir maturin==1.16.0
WORKDIR /src
COPY pyproject.toml README.md ./
COPY crates ./crates
COPY python ./python
RUN mkdir /wheels \
    && pip wheel --no-deps --no-build-isolation . --wheel-dir /wheels \
    && maturin build --release --manifest-path crates/transport/Cargo.toml \
       --features python --out /wheels

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && groupadd --gid 10001 xurcroh \
    && useradd --uid 10001 --gid xurcroh --create-home xurcroh \
    && mkdir -p /models /artifacts /runtime-data \
    && chown -R xurcroh:xurcroh /models /artifacts /runtime-data
COPY requirements-cuda.lock /tmp/requirements-cuda.lock
RUN pip install --no-cache-dir -r /tmp/requirements-cuda.lock
COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-deps /tmp/wheels/xurcroh-*.whl /tmp/wheels/xurcroh_quic-*.whl \
    && rm -rf /tmp/wheels /tmp/requirements-cuda.lock
USER 10001:10001
WORKDIR /runtime-data
EXPOSE 9444 29615
ENTRYPOINT ["python", "-m", "xurcroh.stage_worker"]
CMD ["--help"]
