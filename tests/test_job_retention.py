from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comfyui2api.job_retention import prune_jobs
from comfyui2api.jobs import Job


class DummyManager:
    def __init__(self, runs_dir: Path) -> None:
        self.cfg = SimpleNamespace(runs_dir=runs_dir)
        self._lock = asyncio.Lock()
        self._sub_lock = asyncio.Lock()
        self._jobs = {}
        self._subscribers = {}


def _job(*, job_id: str, created_at: int, status: str) -> Job:
    return Job(
        job_id=job_id,
        created_at_utc="2026-03-19T00:00:00Z",
        created_at=created_at,
        status=status,
        kind="txt2img",
        workflow="test.json",
    )


class JobRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_prune_jobs_removes_expired_terminal_jobs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            manager = DummyManager(runs_dir)
            manager._jobs = {
                "expired-completed": _job(job_id="expired-completed", created_at=10, status="completed"),
                "expired-failed": _job(job_id="expired-failed", created_at=20, status="failed"),
                "still-running": _job(job_id="still-running", created_at=5, status="running"),
                "fresh-completed": _job(job_id="fresh-completed", created_at=95, status="completed"),
            }
            manager._subscribers = {
                "expired-completed": {object()},
                "expired-failed": {object()},
                "still-running": {object()},
            }
            for job_id in manager._jobs:
                (runs_dir / job_id).mkdir(parents=True, exist_ok=True)

            with patch("comfyui2api.job_retention.utc_now_unix", return_value=100):
                removed = await prune_jobs(manager, ttl_seconds=50, max_jobs=0)

            self.assertEqual(removed, ["expired-completed", "expired-failed"])
            self.assertNotIn("expired-completed", manager._jobs)
            self.assertNotIn("expired-failed", manager._jobs)
            self.assertIn("still-running", manager._jobs)
            self.assertIn("fresh-completed", manager._jobs)
            self.assertNotIn("expired-completed", manager._subscribers)
            self.assertNotIn("expired-failed", manager._subscribers)
            self.assertTrue((runs_dir / "still-running").exists())
            self.assertFalse((runs_dir / "expired-completed").exists())

    async def test_prune_jobs_enforces_max_jobs_without_dropping_active_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            manager = DummyManager(runs_dir)
            manager._jobs = {
                "running-new": _job(job_id="running-new", created_at=40, status="running"),
                "done-oldest": _job(job_id="done-oldest", created_at=10, status="completed"),
                "done-middle": _job(job_id="done-middle", created_at=20, status="failed"),
                "done-newest": _job(job_id="done-newest", created_at=30, status="completed"),
            }
            for job_id in manager._jobs:
                (runs_dir / job_id).mkdir(parents=True, exist_ok=True)

            with patch("comfyui2api.job_retention.utc_now_unix", return_value=100):
                removed = await prune_jobs(manager, ttl_seconds=0, max_jobs=2)

            self.assertEqual(removed, ["done-oldest", "done-middle"])
            self.assertIn("running-new", manager._jobs)
            self.assertIn("done-newest", manager._jobs)
            self.assertEqual(set(manager._jobs), {"running-new", "done-newest"})
            self.assertFalse((runs_dir / "done-oldest").exists())
            self.assertFalse((runs_dir / "done-middle").exists())
            self.assertTrue((runs_dir / "done-newest").exists())


if __name__ == "__main__":
    unittest.main()
