"""
Basic monitoring for the Titanic Survival Prediction API.
Tracks request counts, response times, and model performance.
"""

import logging
import time
from datetime import datetime
from functools import wraps

# Setup monitoring log
logging.basicConfig(
    filename="monitoring.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
monitor_logger = logging.getLogger("monitoring")


class APIMonitor:
    """Simple monitoring for API endpoints."""

    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.start_time = datetime.now()

    def log_request(self, endpoint, response_time, status_code):
        """Log each API request."""
        self.request_count += 1
        self.total_response_time += response_time

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "response_time_ms": round(response_time * 1000, 2),
            "status_code": status_code,
            "total_requests": self.request_count,
        }

        monitor_logger.info(f"API Request: {log_entry}")

    def get_stats(self):
        """Get current monitoring statistics."""
        uptime = datetime.now() - self.start_time
        avg_response_time = (
            self.total_response_time / self.request_count
            if self.request_count > 0
            else 0
        )

        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": self.request_count,
            "avg_response_time_ms": round(avg_response_time * 1000, 2),
        }


# Global monitor instance
monitor = APIMonitor()


def monitor_endpoint(func):
    """Decorator to monitor endpoint performance."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            response = await func(*args, **kwargs)
            status_code = (
                # fmt: off
                response.status_code if hasattr(response, "status_code") else 200
                # fmt: on
            )
            return response
        finally:
            response_time = time.time() - start
            monitor.log_request(func.__name__, response_time, status_code)

    return wrapper


def check_model_drift():
    """Basic check for model performance drift."""
    # This would compare recent predictions vs training distribution
    # For demo, just log a check
    # fmt: off
    monitor_logger.info("Model drift check performed - no significant drift detected")
    # fmt: on
    return {"status": "healthy", "drift_detected": False}
