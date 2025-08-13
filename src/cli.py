"""
Command Line Interface for HFT Simulator

Provides convenient commands to launch the dashboard and validate the install.

Usage:
  - hft-simulator dashboard --host 127.0.0.1 --port 8080
  - hft-simulator validate
"""

import sys
import argparse


def _cmd_validate() -> int:
    try:
        # Import a few core components to verify installation
        from src.utils.constants import OrderSide, OrderType
        from src.engine.order_types import Order
        from src.utils.logger import get_logger

        logger = get_logger("hft_cli_validate")
        logger.info("Validation: imports OK")

        import pandas as pd

        _ = Order(
            order_id="cli_test",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=1,
            price=None,
            timestamp=pd.Timestamp.now(),
        )
        print("[OK] Basic object construction works")
        print("HFT Simulator installation looks good.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Validation failed: {exc}")
        return 1


def _cmd_dashboard(host: str, port: int, debug: bool) -> int:
    try:
        from src.visualization.realtime_dashboard import create_demo_dashboard

        dashboard = create_demo_dashboard()
        dashboard.run(host=host, port=port, debug=debug)
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to start dashboard: {exc}")
        return 1


def _cmd_serve(host: str, port: int) -> int:
    """Production-ish server: prefer eventlet if available (Socket.IO compatible)."""
    try:
        try:
            import eventlet  # type: ignore
            eventlet.monkey_patch()
        except Exception:
            eventlet = None  # type: ignore

        from src.wsgi import socketio, app  # exposes dashboard app/socketio

        # debug=False for production
        socketio.run(app, host=host, port=port, debug=False)
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to serve dashboard: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hft-simulator", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate installation and imports")

    p_dash = sub.add_parser("dashboard", help="Launch the real-time dashboard")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8080)
    p_dash.add_argument("--debug", action="store_true")

    p_serve = sub.add_parser("serve", help="Run production server (eventlet if available)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return _cmd_validate()
    if args.cmd == "dashboard":
        return _cmd_dashboard(args.host, args.port, args.debug)
    if args.cmd == "serve":
        return _cmd_serve(args.host, args.port)

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


