# Shared interpreter selection for the Linux and macOS launchers.
yue2_bootstrap_python() {
    for candidate in \
        "$YUE2_ROOT/.venv/bin/python" \
        "$YUE2_ROOT/../../.venv/bin/python" \
        "$YUE2_ROOT/../../venv/bin/python"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    printf '%s\n' 'Python 3 was not found. Install Python 3 or use the ComfyUI virtual environment.' >&2
    return 1
}

yue2_runtime_python() {
    if [ -x "$YUE2_ROOT/.venv/bin/python" ]; then
        printf '%s\n' "$YUE2_ROOT/.venv/bin/python"
        return 0
    fi
    printf '%s\n' 'The isolated YuE2 runtime was not found. Run sh setup-runtime.sh first.' >&2
    return 1
}
