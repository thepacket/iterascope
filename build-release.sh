#!/usr/bin/env bash
# Build the production static bundle into dist/.
set -euo pipefail

cd "$(dirname "$0")"

trunk build --release --no-sri

wasm=$(find dist -maxdepth 1 -name '*_bg.wasm' | head -1)
if [[ -z "$wasm" ]]; then
    echo "no wasm found in dist/ — did trunk succeed?" >&2
    exit 1
fi

if command -v wasm-opt >/dev/null 2>&1; then
    before=$(wc -c <"$wasm")
    wasm-opt -Oz --output "$wasm.opt" "$wasm"
    mv "$wasm.opt" "$wasm"
    after=$(wc -c <"$wasm")
    printf 'wasm-opt: %d KB -> %d KB\n' "$((before / 1024))" "$((after / 1024))"
else
    echo "wasm-opt not found; keeping Trunk's unoptimized wasm" >&2
fi

echo "dist/ is ready for static hosting."
