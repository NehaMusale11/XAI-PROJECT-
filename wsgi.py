"""
Production WSGI Entrypoint.
Automatically selects Waitress on Windows and Gunicorn on Linux/macOS.
"""
import os
import sys

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    if sys.platform == 'win32':
        try:
            from waitress import serve
            print(f"[WSGI] Starting production Waitress server on http://{host}:{port} ...", flush=True)
            serve(app, host=host, port=port)
        except ImportError:
            print(f"[WSGI] Falling back to standard server on http://{host}:{port} ...", flush=True)
            app.run(host=host, port=port)
    else:
        os.system(f"gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind {host}:{port}")
