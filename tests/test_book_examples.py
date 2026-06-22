"""書本 examples 的核心不變量測試——讓機器顧著「別讓書裡的範例悄悄壞掉」。

只測「不需語料、不需訓練」就能驗的核心宣稱，跑得快、適合進 make test：
  - tiny_kvcache：KV-cache 與 naive 必須**逐 token 完全相同**（它是省、不是近似）。
  - tiny_dedup：偵測器抓到所有真含髒語法的；LSH 抓到的是 naive 的子集（近似不會無中生有）；
    候選對遠少於 C(N,2)（這就是 O(n^2)→近 O(n)）。

需要語料 + 訓練的端到端煙霧測試在 `make book-smoke`（BOOK_SMOKE=1 縮到極小設定）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "book", "examples")))


class TestKVCacheCorrectness(unittest.TestCase):
    def test_cached_equals_naive(self):
        import torch
        import tiny_kvcache as kv
        torch.manual_seed(0)
        model = kv.GPT().eval()
        prompt = torch.randint(0, kv.vocab_size, (1, 5))
        a = model.generate_naive(prompt, 16)
        b = model.generate_cached(prompt, 16)
        self.assertTrue(torch.equal(a, b),
                        "KV-cache 必須與 naive 逐 token 相同（是省、不是近似）")


class TestDedupInvariants(unittest.TestCase):
    def setUp(self):
        import tiny_dedup as dd
        self.dd = dd
        # 合成語料：60 段大致互異的「文件」（避免全都互相相似）
        text = "\n\n".join(
            " ".join(f"w{(i * 7 + j) % 200}" for j in range(40)) for i in range(60))
        rand = dd.make_rand()
        self.docs, self.dup_truth = dd.build_corpus(text, rand, n_docs=50, n_dupes=10)
        self.result = dd.dedup(self.docs, rand)

    def test_detector_catches_all_dirty(self):
        actual = {i for i, d in enumerate(self.docs) if self.dd.DIRTY in d}
        self.assertEqual(self.dd.detector_hits(self.docs), actual)

    def test_lsh_is_subset_of_naive(self):
        # 近似法可能漏，但不該無中生有
        self.assertTrue(self.result["lsh_dupes"] <= self.result["naive_dupes"])

    def test_candidates_far_fewer_than_all_pairs(self):
        self.assertLess(len(self.result["candidates"]), self.result["naive_pairs"])

    def test_naive_finds_injected_dupes(self):
        found = len(self.result["naive_dupes"] & self.dup_truth)
        self.assertGreaterEqual(found, len(self.dup_truth) - 2)


class TestDriftPSI(unittest.TestCase):
    def setUp(self):
        import tiny_drift as dr
        self.dr = dr
        self.edges = list(range(0, 61, 5)) + [1e9]
        rand = dr.make_rng()
        self.train = dr.sample_lengths(rand, 3000, 25, 6)
        self.same = dr.sample_lengths(rand, 1500, 25, 6)
        self.shifted = dr.sample_lengths(rand, 1500, 35, 8)

    def test_same_distribution_psi_near_zero(self):
        v = self.dr.psi(self.train, self.same, self.edges)
        self.assertLess(v, 0.1, "同分布 PSI 應落在『穩定』(<0.1)")

    def test_shift_raises_psi_above_same(self):
        same_v = self.dr.psi(self.train, self.same, self.edges)
        shift_v = self.dr.psi(self.train, self.shifted, self.edges)
        self.assertGreater(shift_v, same_v)
        self.assertGreater(shift_v, 0.25, "明顯漂移 PSI 應越過重訓門檻(0.25)")


if __name__ == "__main__":
    unittest.main()
