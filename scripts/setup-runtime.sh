#!/bin/sh
set -eu
YUE2_ROOT=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
exec sh "$YUE2_ROOT/setup-runtime.sh" "$@"
