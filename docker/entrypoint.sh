#!/bin/sh
set -e

# Django migrations
python3 manage.py migrate --noinput

# Optionally collectstatic if later needed
# python3 manage.py collectstatic --noinput

# Print ONNX providers at startup (if possible)
python3 - <<'PY'
import os
try:
    import onnxruntime as ort
    providers = getattr(ort, 'get_available_providers', lambda: [])()
    print('ONNXRuntime available providers:', providers)
    if os.getenv('REQUIRE_CUDA', '0') == '1' and 'CUDAExecutionProvider' not in providers:
        print('ERROR: CUDAExecutionProvider not available while REQUIRE_CUDA=1. Exiting...')
        raise SystemExit(7)
except SystemExit as e:
    raise
except Exception as e:
    print('ONNXRuntime not available:', e)
    if os.getenv('REQUIRE_CUDA', '0') == '1':
        raise SystemExit(8)
PY

# Start server
exec python3 manage.py runserver 0.0.0.0:${PORT:-8000}
