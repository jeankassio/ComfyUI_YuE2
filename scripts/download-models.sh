#!/bin/sh
set -eu
YUE2_ROOT=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
. "$YUE2_ROOT/scripts/shell-common.sh"
YUE2_PYTHON=$(yue2_runtime_python)
YUE2_ACTION=download
for argument in "$@"; do
    case "$argument" in
        --check|--download) YUE2_ACTION=explicit ;;
    esac
done
if [ "$YUE2_ACTION" = download ]; then
    set -- --download "$@"
fi
exec "$YUE2_PYTHON" "$YUE2_ROOT/scripts/download_models.py" "$@"
