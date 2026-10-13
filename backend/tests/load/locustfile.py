"""
Locust load test simulation script for 50 concurrent users.
Simulates realistic classroom traffic querying the Unified RAG Enterprise Science Gateway.
"""

import random
from locust import HttpUser, task, between

try:
    from tests.load.queries import SCIENCE_TEST_QUERIES
except ImportError:
    from backend.tests.load.queries import SCIENCE_TEST_QUERIES


class ScienceStudentUser(HttpUser):
    """Simulates an active student interacting with the Science RAG gateway."""

    # Pacing: between 1 and 3 seconds between actions
    wait_time = between(1.0, 3.0)

    def on_start(self):
        """Initialize student session."""
        self.session_id = f"student_sim_{random.randint(1000, 9999)}"

    @task(3)
    def query_batch_chat(self):
        """Submit batch chat inquiry."""
        query = random.choice(SCIENCE_TEST_QUERIES)
        payload = {
            "query": query,
            "session_id": self.session_id,
            "top_k": 3
        }
        with self.client.post("/api/v1/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if not data.get("answer"):
                    response.failure("Empty answer returned")
                else:
                    response.success()
            elif response.status_code == 429:
                # Acceptable rate throttling under extreme burst
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def check_health(self):
        """Readiness and health check."""
        self.client.get("/api/v1/health")

    @task(1)
    def check_metrics(self):
        """Inspect live operational metrics."""
        self.client.get("/api/v1/metrics")
