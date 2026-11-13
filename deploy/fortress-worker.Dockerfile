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

FROM gcr.io/distroless/python3-debian12:nonroot
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /opt/venv /opt/venv
WORKDIR /runtime-data
EXPOSE 9443
ENTRYPOINT ["/opt/venv/bin/python", "-m", "xurcroh.worker"]
CMD ["--help"]
