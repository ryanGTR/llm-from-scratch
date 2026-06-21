"""Reward Model（RM）——RLHF 的第一塊：把「一個回答有多好」壓成一個純量分數。

對照 DPO：DPO 把 reward **隱含**在 policy 裡（r = β·log π/π_ref，不用另外訓模型）；
RLHF 把 reward 做成一顆**獨立模型**——好處是之後能用 RL 拿它當分數去優化 policy，
壞處是它只是「學來的代理指標」，優化過頭會被 **reward hacking**（policy 找到讓 RM 給高分、
但人類其實覺得爛的捷徑＝Goodhart's law）。這正是 RLHF 最有名的坑。

結構：GPT 骨幹（從 SFT 接權重）＋ 一個純量 value head 接在「最後一個回答 token」的隱藏狀態上。
用 Bradley-Terry 損失訓練：loss = −log σ( r(chosen) − r(rejected) )，跟 DPO 的偏好損失同源，
只是這裡 r 是獨立模型的輸出、不是 policy 的對數機率比。

Java 類比：RM 就像一個學來的「自動評分器/驗收 gate」——但它本身可能有 bug，被優化器鑽。
"""

import torch
import torch.nn as nn

from .config import GPTConfig
from .model import GPT


class RewardModel(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        cfg = gpt_config if isinstance(gpt_config, GPTConfig) else GPTConfig(**gpt_config)
        self.gpt = GPT(cfg)
        self.v_head = nn.Linear(cfg.n_embd, 1, bias=False)
        self._hidden = None
        # 用 forward hook 抓 ln_f 的輸出（最後一層隱藏狀態），不必改 model.py
        self.gpt.ln_f.register_forward_hook(lambda m, i, o: setattr(self, "_hidden", o))

    def forward(self, idx, last_idx):
        """idx: (B,T)；last_idx: (B,) 每條序列「最後一個回答 token」的位置。

        回傳 (B,) 純量 reward——讀「最後一個回答 token」的隱藏狀態，因為到那裡模型才看完整個回答。
        """
        self.gpt(idx)                                  # 觸發 hook 填 self._hidden: (B,T,n_embd)
        h = self._hidden[torch.arange(idx.size(0), device=idx.device), last_idx]
        return self.v_head(h).squeeze(-1)

    @classmethod
    def from_ckpt(cls, path, device):
        ck = torch.load(path, map_location=device)
        m = cls(ck["gpt_config"]).to(device)
        m.load_state_dict(ck["model"])
        return m


def bt_loss(r_chosen, r_rejected):
    """Bradley-Terry 偏好損失：要 chosen 的分數高過 rejected。

    跟 DPO 同源（−log σ(好−壞)），差別只在這裡的「好/壞」是 RM 的純量輸出，
    不是 policy 的對數機率比。回傳 (loss, 偏好準確率)。
    """
    margin = r_chosen - r_rejected
    loss = -torch.nn.functional.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()
    return loss, acc
