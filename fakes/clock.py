from __future__ import annotations

import asyncio


class FakeClock:
    """Virtual time: sleep() advances the clock instantly. Single-run cases are sequential,
    so virtual latency = the sum of the latencies the fakes declare -> TTFT is reproducible."""

    def __init__(self, start_ms: int = 0):
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    async def sleep(self, ms: int) -> None:
        self._now += max(0, int(ms))
        await asyncio.sleep(0)
