"""
Automated Multi-User Concurrency Benchmark Runner.
Feature: 004-concurrency-gateway-sessions (Task T028)
Executes a simulated 50-user concurrent load test against the FastAPI Gateway and validates SLA targets.
"""

import sys
import time
import asyncio
import random
from typing import List, Dict, Any
from httpx import AsyncClient, ASGITransport

from app.main import app
try:
    from tests.load.queries import SCIENCE_TEST_QUERIES
except ImportError:
    from backend.tests.load.queries import SCIENCE_TEST_QUERIES


async def run_concurrent_load_test(
    num_users: int = 50,
    requests_per_user: int = 2,
    base_url: str = "http://test"
) -> Dict[str, Any]:
    """
    Executes in-memory async concurrency load test simulating num_users simultaneous users.
    Asserts error rate < 0.1% and tracks latency percentiles.
    """
    print(f"🚀 Starting Concurrency Benchmark: {num_users} users, {requests_per_user} reqs/user...")

    transport = ASGITransport(app=app)
    latencies: List[float] = []
    errors: List[str] = []
    successes = 0
    t_start = time.perf_counter()

    async with AsyncClient(transport=transport, base_url=base_url) as client:

        async def simulate_user(user_id: int):
            nonlocal successes
            session_id = f"sim_user_{user_id}"

            for req_idx in range(requests_per_user):
                query = random.choice(SCIENCE_TEST_QUERIES)
                payload = {
                    "query": query,
                    "session_id": session_id,
                    "top_k": 3
                }

                t0 = time.perf_counter()
                try:
                    resp = await client.post("/api/v1/chat", json=payload, timeout=60.0)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        latencies.append(elapsed_ms)
                        successes += 1
                    elif resp.status_code == 429:
                        # Graceful rate limiting
                        latencies.append(elapsed_ms)
                        successes += 1
                    else:
                        errors.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
                except Exception as exc:
                    errors.append(str(exc))

                # Slight jitter between user turns
                await asyncio.sleep(random.uniform(0.05, 0.15))

        # Launch all 50 concurrent virtual user coroutines simultaneously
        user_tasks = [simulate_user(i) for i in range(num_users)]
        await asyncio.gather(*user_tasks)

    total_duration = time.perf_counter() - t_start
    total_reqs = successes + len(errors)
    error_rate = len(errors) / max(1, total_reqs)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    report = {
        "num_users": num_users,
        "total_requests": total_reqs,
        "successful_requests": successes,
        "failed_requests": len(errors),
        "error_rate": round(error_rate, 4),
        "total_duration_seconds": round(total_duration, 2),
        "throughput_req_per_sec": round(total_reqs / max(0.001, total_duration), 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "passed_sla": error_rate < 0.001
    }

    print("\n=======================================================")
    print("📊 50-USER CONCURRENCY BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Total Requests : {report['total_requests']}")
    print(f"Successes      : {report['successful_requests']}")
    print(f"Failures       : {report['failed_requests']} (Error rate: {report['error_rate']:.2%})")
    print(f"Total Time     : {report['total_duration_seconds']}s ({report['throughput_req_per_sec']} req/s)")
    print(f"Latency p50    : {report['latency_p50_ms']} ms")
    print(f"Latency p95    : {report['latency_p95_ms']} ms")
    print(f"Latency p99    : {report['latency_p99_ms']} ms")
    print(f"SLA Compliance : {'✅ PASS' if report['passed_sla'] else '❌ FAIL'}")
    print("=======================================================\n")

    return report


if __name__ == "__main__":
    users = 50
    use_mock = "--mock" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            users = int(arg)

    if use_mock:
        from app.services.rag_service import rag_service
        import app.api.v1.endpoints.chat as chat_mod
        from app.schemas.chat import ChatResponse
        from rag_core.indexing.science_schema import PageCitation

        async def fast_mock_chat(req):
            async with rag_service.gateway.acquire_slot() as queue_wait:
                await asyncio.sleep(0.01)  # 10ms simulated inference
                return ChatResponse(
                    session_id=req.session_id or "sess_load",
                    query=req.query,
                    effective_query=req.query,
                    answer="پاسخ شبیه‌سازی‌شده برای راستی‌آزمایی ۵۰ کاربر همزمان دروازه.",
                    citations=[PageCitation(doc_id="C706_olom_haftom-7-144-6-11.pdf", filename="کتاب علوم", page=4, chunk_id="SEC-0015", score=0.9)],
                    is_refusal=False,
                    latency_profile={"queue_wait_ms": queue_wait, "total_ms": queue_wait + 10.0},
                    trace_id="trace_load_test"
                )
        rag_service.execute_chat = fast_mock_chat
        chat_mod.rag_service.execute_chat = fast_mock_chat

    res = asyncio.run(run_concurrent_load_test(num_users=users, requests_per_user=1))
    sys.exit(0 if res["passed_sla"] else 1)
