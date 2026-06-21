"""DPO 的數學不變量測試——別信「我覺得公式對」，讓機器驗。

核心不變量：
  - policy == reference 時，margin=0、loss = -log σ(0) = log 2 ≈ 0.6931（DPO 的「無偏好」基準）
  - 拉高 chosen 的 logπ（margin 變正）→ loss 一定下降
  - 回答段遮罩真的有把「指令段」排除在 logπ 之外
"""

import importlib
import math
import unittest

import torch

dpo = importlib.import_module("pipeline.06_dpo")


class TestDPOLoss(unittest.TestCase):
    def test_zero_margin_is_log2(self):
        # policy 與 reference 完全相同 → margin 0 → loss = log 2
        lp = torch.tensor([0.0, 0.0, 0.0, 0.0])      # 交錯 (chosen, rejected) x2
        loss, margin, acc = dpo.dpo_loss(lp, lp.clone(), beta=0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)
        self.assertAlmostEqual(margin, 0.0, places=6)

    def test_positive_margin_lowers_loss(self):
        ref = torch.tensor([-1.0, -1.0])             # 1 組偏好對
        better = torch.tensor([-0.2, -2.0])          # policy 偏好 chosen（margin>0）
        worse = torch.tensor([-2.0, -0.2])           # policy 偏好 rejected（margin<0）
        l_better, m_better, acc_better = dpo.dpo_loss(better, ref, beta=0.1)
        l_worse, m_worse, _ = dpo.dpo_loss(worse, ref, beta=0.1)
        self.assertGreater(m_better, 0)
        self.assertLess(m_worse, 0)
        self.assertLess(l_better.item(), math.log(2))   # 偏好對 → loss < 基準
        self.assertGreater(l_worse.item(), math.log(2))
        self.assertEqual(acc_better, 1.0)

    def test_acc_counts_preference_direction(self):
        # 兩組：第一組偏好 chosen、第二組偏好 rejected → acc=0.5
        pol = torch.tensor([0.0, -1.0, -1.0, 0.0])
        ref = torch.zeros(4)
        _, _, acc = dpo.dpo_loss(pol, ref, beta=0.1)
        self.assertAlmostEqual(acc, 0.5, places=6)


class TestIPOLoss(unittest.TestCase):
    def test_min_at_target_margin(self):
        # IPO 損失在 margin = 1/(2β) 時為 0（最小）。β=0.1 → target=5
        beta = 0.1
        target = 1.0 / (2 * beta)
        # 造一筆讓 margin 剛好 = target：policy chosen-rejected 差 target、ref 為 0
        pol = torch.tensor([target, 0.0])     # (chosen, rejected) logπ
        ref = torch.tensor([0.0, 0.0])
        loss, margin, _ = dpo.ipo_loss(pol, ref, beta)
        self.assertAlmostEqual(margin, target, places=5)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss_grows_away_from_target(self):
        beta = 0.1
        ref = torch.tensor([0.0, 0.0])
        near = dpo.ipo_loss(torch.tensor([4.0, 0.0]), ref, beta)[0]
        far = dpo.ipo_loss(torch.tensor([50.0, 0.0]), ref, beta)[0]   # margin 推爆 → 被罰
        self.assertLess(near.item(), far.item())   # 離目標越遠損失越大（DPO 不會罰這個）


class TestRespMask(unittest.TestCase):
    class _Tok:                                       # 假 tokenizer：一字元一 id
        def encode(self, s):
            return [ord(c) % 50 for c in s]

    def test_mask_excludes_prompt(self):
        tok = self._Tok()
        ids, mask = dpo.build_example(tok, "甲", "乙丙", block_size=128)
        # prompt = "問：甲\n答：" 共 5 字 → 前 5 個 mask 應為 0
        p_len = len(tok.encode(dpo.PROMPT_TMPL.format(q="甲")))
        self.assertEqual(sum(mask[:p_len]), 0)
        self.assertTrue(all(m == 1 for m in mask[p_len:]))
        self.assertEqual(len(ids), len(mask))

    def test_truncates_to_block_size(self):
        tok = self._Tok()
        ids, mask = dpo.build_example(tok, "甲" * 100, "乙" * 100, block_size=32)
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(mask), 32)


if __name__ == "__main__":
    unittest.main()
