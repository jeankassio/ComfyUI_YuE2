#!/bin/sh
set -eu
YUE2_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
. "$YUE2_ROOT/scripts/shell-common.sh"
YUE2_PYTHON=$(yue2_bootstrap_python)
exec "$YUE2_PYTHON" "$YUE2_ROOT/scripts/setup_runtime.py" "$@"
