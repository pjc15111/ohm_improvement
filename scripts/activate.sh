# Source this script; do not execute it.
oi_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

source "$oi_root/venv/bin/activate"

# UConn's aliases can otherwise override the activated environment.
unalias python python3 pip pip3 2>/dev/null || true

export PYTHONPATH="$oi_root${PYTHONPATH:+:$PYTHONPATH}"
unset oi_root
