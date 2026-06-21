"""動態批次（dynamic batching）：把「同時進來的請求」合成一個 batch 一起算。

B=1 時 GPU 大半閒著；把 B 個請求疊成 (B, T) 一次 forward，幾乎同樣時間做完 B 個 →
吞吐大增。這是 vLLM / TGI 在做的事的簡化版（它們再加「連續批次 + 不同長度」）。

設計：一個 async 佇列收 job；批次器醒來抓第一個、再用一個極短視窗(max_wait_ms)多收幾個、
湊到 max_batch；把「相容（同長度+同生成參數+同模型）」的湊成一批，丟到 thread 跑 torch
（不擋 event loop），再把結果發回各自的 future。
"""

import asyncio

import torch


class DynamicBatcher:
    def __init__(self, max_batch: int = 8, max_wait_ms: float = 8.0, on_batch=None):
        self.q: asyncio.Queue = asyncio.Queue()
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.on_batch = on_batch      # callback(batch_size)：給 Prometheus 觀測批量分布
        self.batches_run = 0          # 觀測：跑了幾批
        self.reqs_served = 0          # 觀測：服務幾個請求 → 平均批量 = reqs/batches

    async def submit(self, job: dict):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((job, fut))
        return await fut

    async def run(self):
        while True:
            job, fut = await self.q.get()
            pairs = [(job, fut)]
            while len(pairs) < self.max_batch:        # 短視窗內盡量多收
                try:
                    pairs.append(await asyncio.wait_for(self.q.get(), timeout=self.max_wait))
                except asyncio.TimeoutError:
                    break
            groups: dict = {}                         # 只有「相容」的能疊一起
            for j, f in pairs:
                groups.setdefault(j["key"], []).append((j, f))
            for grp in groups.values():
                outs = await asyncio.to_thread(self._forward, [j for j, _ in grp])
                for (_, f), o in zip(grp, outs):
                    if not f.done():
                        f.set_result(o)
                self.batches_run += 1
                self.reqs_served += len(grp)
                if self.on_batch:
                    self.on_batch(len(grp))      # 回報這批多大 → Prometheus

    def _forward(self, jobs: list[dict]) -> list[list[int]]:
        """同 key 的 job：ids 等長、參數相同 → 直接疊成 (B, T) 批次生成。"""
        p, device, model = jobs[0]["params"], jobs[0]["device"], jobs[0]["model"]
        idx = torch.tensor([j["ids"] for j in jobs], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                idx, p["max_new_tokens"], temperature=p["temperature"],
                top_k=p["top_k"], top_p=p["top_p"], min_p=p["min_p"],
                use_kv_cache=p["use_kv_cache"])
        if device == "cuda":
            torch.cuda.synchronize()
        return [out[i].tolist() for i in range(len(jobs))]
