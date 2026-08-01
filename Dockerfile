# syntax=docker/dockerfile:1

# IteraScope ships as a static WebGPU/WASM bundle. The runtime only serves the
# files; all fractal computation happens on the visitor's GPU.

########################  build  ########################
FROM rust:1.96-bookworm AS build

ARG TRUNK_VERSION=0.21.14
ARG BINARYEN_VERSION=131
ARG TARGETARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends brotli \
 && rm -rf /var/lib/apt/lists/*

# Use official prebuilt tools. Compiling Trunk and Binaryen from source would
# dominate an otherwise small deployment build.
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) arch=x86_64 ;; \
      arm64) arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/trunk-rs/trunk/releases/download/v${TRUNK_VERSION}/trunk-${arch}-unknown-linux-gnu.tar.gz" \
      | tar -xz -C /usr/local/bin trunk; \
    curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/version_${BINARYEN_VERSION}/binaryen-version_${BINARYEN_VERSION}-${arch}-linux.tar.gz" \
      | tar -xz -C /opt; \
    ln -s "/opt/binaryen-version_${BINARYEN_VERSION}/bin/wasm-opt" /usr/local/bin/wasm-opt

RUN rustup target add wasm32-unknown-unknown

WORKDIR /app

# Compile dependencies in a cacheable layer. The real source replaces this
# minimal crate below, so application edits do not rebuild the full wgpu tree.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
 && printf 'fn main() {}\n' > src/main.rs \
 && printf '\n' > src/lib.rs \
 && cargo build --release --target wasm32-unknown-unknown --bin iterascope \
 && rm -rf src

COPY . .
RUN touch src/main.rs src/lib.rs

# Trunk's own wasm-opt integration is disabled in index.html. Optimize here,
# after Trunk emits the bundle; --no-sri is required because the wasm bytes are
# rewritten after their filename is selected.
RUN trunk build --release --no-sri
RUN set -eux; \
    wasm="$(find dist -maxdepth 1 -name '*_bg.wasm' | head -1)"; \
    test -n "$wasm"; \
    wasm-opt -Oz --output "${wasm}.opt" "$wasm"; \
    mv "${wasm}.opt" "$wasm"

# Prepare both formats once during the build. Caddy selects the best supported
# representation without spending runtime CPU on compression.
RUN find dist -type f \( -name '*.wasm' -o -name '*.js' -o -name '*.html' -o -name '*.css' \) \
      -exec gzip -9 -k {} \; \
      -exec brotli -9 -k {} \;

########################  runtime  ########################
FROM caddy:2-alpine

COPY --from=build /app/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 8080
