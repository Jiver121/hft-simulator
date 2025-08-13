"""
WSGI/ASGI entrypoint for production servers.

Exposes `app` and `socketio` so you can run with Gunicorn:
  gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 src.wsgi:app
or SocketIO-aware entrypoint:
  gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 'src.wsgi:socketio'
"""

from src.visualization.realtime_dashboard import create_demo_dashboard

_dashboard = create_demo_dashboard()
app = _dashboard.app
socketio = _dashboard.socketio


def main() -> None:  # Optional: local run helper
    _dashboard.run()


if __name__ == "__main__":
    main()


