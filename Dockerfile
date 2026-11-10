FROM python:3.12-bookworm AS build

ARG RUST_VERSION=1.99.1
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/opt/venv/bin:$PATH
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && python -m venv /opt/venv
WORKDIR /src
COPY requirements.lock pyproject.toml README.md ./
COPY crates ./crates
COPY python ./python
RUN pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
       -r requirements.lock \
    && maturin build --release --manifest-path crates/transport/Cargo.toml --features python --out /wheels \
    && pip install --no-cache-dir --no-deps /wheels/xurcroh_quic-*.whl \
    && pip install --no-cache-dir --no-deps .

FROM python:3.12-slim-bookworm AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10001 xurcroh \
    && useradd --uid 10001 --gid xurcroh --create-home xurcroh \
    && mkdir -p /models /runtime-data \
    && chown -R xurcroh:xurcroh /models /runtime-data
COPY --from=build /opt/venv /opt/venv
USER 10001:10001
WORKDIR /runtime-data
EXPOSE 9443
ENTRYPOINT ["python", "-m", "xurcroh.worker"]
CMD ["--help"]
