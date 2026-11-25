#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export ALPHAMASTER_DEMO=1
export ALPHAMASTER_CPU_THREADS=2
export MPLCONFIGDIR="$PWD/demo/mpl-cache"
exec .venv/bin/python demo_server.py "$@"
