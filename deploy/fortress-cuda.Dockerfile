FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ARG RUST_VERSION=1.99.1
ARG CUDA_ARCH=sm_89
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/opt/venv/bin:$PATH
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv curl build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 --retry 3 --retry-all-errors -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && python3.12 -m venv /opt/venv
WORKDIR /src
COPY requirements-fortress-cuda.lock pyproject.toml README.md ./
COPY crates ./crates
COPY python ./python
COPY native ./native
COPY scripts/build-native.sh ./scripts/build-native.sh
RUN pip install --no-cache-dir -r requirements-fortress-cuda.lock maturin==1.16.0 \
    && pip wheel --no-deps --no-build-isolation . --wheel-dir /wheels \
    && maturin build --release --manifest-path crates/transport/Cargo.toml --features python --out /wheels \
    && CUDA_ARCH="${CUDA_ARCH}" ./scripts/build-native.sh

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS runtime

ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    XURCROH_NATIVE_LIBRARY=/opt/xurcroh/native/build/xurcroh_burn_alloc.so
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv --system-site-packages /opt/venv \
    && groupadd --gid 10001 xurcroh \
    && useradd --uid 10001 --gid xurcroh --create-home xurcroh \
    && mkdir -p /runtime-data /opt/xurcroh/native/build \
    && chown -R xurcroh:xurcroh /runtime-data /opt/xurcroh
COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/native/build/xurcroh_burn_alloc.so /opt/xurcroh/native/build/xurcroh_burn_alloc.so
USER 10001:10001
WORKDIR /runtime-data
EXPOSE 9444
ENTRYPOINT ["/opt/venv/bin/python", "-m", "xurcroh.fortress.worker_cli"]
CMD ["--help"]
