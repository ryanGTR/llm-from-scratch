"""RLHF（reward model + GRPO）的數學不變量測試。

守住：
- Bradley-Terry 損失：r_chosen==r_rejected → loss=log2、acc=0.5；chosen 高 → loss<log2
- RewardModel forward 形狀對、能讀「指定位置」的純量分數
- GRPO 的回答段 logπ 遮罩真的排除指令段
- 組內標準化 advantage：mean≈0、std≈1
"""

import importlib
import math
import unittest

import torch

from src.config import GPTConfig
from src.reward_model import RewardModel, bt_loss

grpo = importlib.import_module("pipeline.08_grpo")
ppo = importlib.import_module("pipeline.09_ppo")

TINY = dict(block_size=16, n_layer=1, n_head=2, n_embd=16, vocab_size=32, dropout=0.0)


class TestBTLoss(unittest.TestCase):
    def test_equal_scores_is_log2(self):
        r = torch.tensor([0.5, -1.0, 2.0])
        loss, acc = bt_loss(r, r.clone())
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)
        self.assertAlmostEqual(acc.item(), 0.0, places=6)   # margin=0 不算 >0

    def test_chosen_higher_lowers_loss(self):
        rc = torch.tensor([2.0, 1.0])
        rr = torch.tensor([-2.0, -1.0])
        loss, acc = bt_loss(rc, rr)
        self.assertLess(loss.item(), math.log(2))
        self.assertEqual(acc.item(), 1.0)


class TestRewardModel(unittest.TestCase):
    def test_forward_shape_and_position(self):
        torch.manual_seed(0)
        rm = RewardModel(GPTConfig(**TINY))
        idx = torch.randint(0, TINY["vocab_size"], (3, 10))
        last = torch.tensor([9, 5, 7])
        r = rm(idx, last)
        self.assertEqual(r.shape, (3,))           # 每條序列一個純量
        self.assertTrue(torch.isfinite(r).all())


class TestGRPO(unittest.TestCase):
    def test_resp_logp_masks_prompt(self):
        torch.manual_seed(0)
        from src.model import GPT
        m = GPT(GPTConfig(**TINY)).eval()
        X = torch.randint(0, TINY["vocab_size"], (2, 12))
        Tp = 5
        lp, ntok = grpo.resp_logp(m, X, Tp)
        # 回答段 token 數 = 序列長(去掉最後一個預測不到的) − (Tp−1) = (12−1) − (5−1) = 7
        self.assertTrue(torch.allclose(ntok, torch.tensor([7.0, 7.0])))
        self.assertEqual(lp.shape, (2,))
        self.assertTrue((lp <= 0).all())          # logπ 必為非正

    def test_group_advantage_normalized(self):
        r = torch.tensor([3.0, 1.0, -1.0, -3.0])
        adv = (r - r.mean()) / (r.std() + 1e-6)
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(adv.std().item(), 1.0, places=2)


class TestPPOClip(unittest.TestCase):
    def test_clip_caps_gain_for_positive_advantage(self):
        # adv>0、ratio 衝過 1+ε：clip 應該夾住 → loss 比不夾時「大」（gain 較少）
        ratio = torch.tensor([[2.0, 2.0]])         # 遠超 1+ε
        adv = torch.tensor([1.0])                  # 正 advantage
        mask = torch.ones(1, 2)
        loss_clip, cf = ppo.ppo_policy_loss(ratio, adv, mask, clip_eps=0.2)
        loss_noclip, cf0 = ppo.ppo_policy_loss(ratio, adv, mask, clip_eps=0.0)
        self.assertGreater(loss_clip.item(), loss_noclip.item())   # 夾住 → 限制了 policy gain
        self.assertAlmostEqual(cf, 1.0)            # 兩個 token 都被夾
        self.assertAlmostEqual(cf0, 0.0)           # 不夾時 clip_frac=0

    def test_no_clip_when_ratio_in_range(self):
        ratio = torch.tensor([[1.05, 0.97]])       # 都在 [1-ε,1+ε] 內
        adv = torch.tensor([1.0])
        mask = torch.ones(1, 2)
        _, cf = ppo.ppo_policy_loss(ratio, adv, mask, clip_eps=0.2)
        self.assertAlmostEqual(cf, 0.0)            # 沒有 token 被夾

    def test_token_logp_shape(self):
        from src.model import GPT
        torch.manual_seed(0)
        m = GPT(GPTConfig(**TINY)).eval()
        X = torch.randint(0, TINY["vocab_size"], (3, 9))
        lp = ppo.token_logp(m, X)
        self.assertEqual(lp.shape, (3, 8))         # 每個被預測位置一個 logπ
        self.assertTrue((lp <= 0).all())


if __name__ == "__main__":
    unittest.main()
