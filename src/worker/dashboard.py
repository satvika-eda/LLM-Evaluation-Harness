"""
RQ Dashboard — job monitoring web interface.

Mounts rq-dashboard as a Blueprint on a standalone Flask app so it can be
run independently of the FastAPI eval API (different port, different process).

Usage
-----
    # From the project root:
    python -m src.worker.dashboard

    # Or via gunicorn for a more robust deployment:
    gunicorn "src.worker.dashboard:create_app()" --bind 0.0.0.0:9181

    Then open http://localhost:9181/

Configuration (environment variables)
--------------------------------------
REDIS_URL         : Redis connection string (default: redis://localhost:6379/0)
RQ_DASHBOARD_PORT : Port to listen on when run as __main__ (default: 9181)
RQ_POLL_INTERVAL  : Dashboard poll interval in milliseconds (default: 2500)
SECRET_KEY        : Flask secret key (default: dev-only fallback — change
                    this in production)

Security note
-------------
rq-dashboard has no built-in authentication.  In production, put it behind
a reverse proxy (nginx / Caddy) with basic-auth or VPN access restriction.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask

import rq_dashboard

load_dotenv()

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """
    Create and return the Flask application with rq-dashboard mounted.

    The app factory pattern lets gunicorn / pytest import the app without
    immediately starting a server.
    """
    app = Flask(__name__)

    # ── Flask config ──────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "rq-dashboard-dev-secret-change-in-prod"
    )

    # ── rq-dashboard config ───────────────────────────────────────────────────
    # Apply rq-dashboard's own defaults first, then override selectively.
    app.config.from_object(rq_dashboard.default_settings)

    app.config["RQ_DASHBOARD_REDIS_URL"] = os.environ.get(
        "REDIS_URL", "redis://localhost:6379/0"
    )
    app.config["RQ_DASHBOARD_POLL_INTERVAL"] = int(
        os.environ.get("RQ_POLL_INTERVAL", "2500")
    )
    # Show jobs from all queues by default, not just "default".
    app.config["RQ_DASHBOARD_QUEUES_BY_DEFAULT"] = ["eval_jobs", "failed"]

    # ── Mount the dashboard Blueprint ─────────────────────────────────────────
    # url_prefix="" mounts it at root so http://host:9181/ shows the dashboard.
    rq_dashboard.web.setup(app, url_prefix="")

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/health")
    def health():  # type: ignore[return]
        """Lightweight liveness probe for load balancers / Docker HEALTHCHECK."""
        from flask import jsonify
        return jsonify({"status": "ok", "service": "rq-dashboard"})

    return app


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("RQ_DASHBOARD_PORT", "9181"))
    app = create_app()
    print(f"rq-dashboard running at http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
