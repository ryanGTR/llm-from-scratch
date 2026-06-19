"""把模型「真實的 attention 權重」抓出來，給視覺化用。

刻意不改 model.py（不打亂教學引用的行號）——改用 PyTorch 的 forward hook
從外面攔截每層 c_attn 的輸出，再照 model.py 同樣的算法還原 attention 矩陣。

Java 類比：forward hook 就像 AOP 的 around advice——不動原方法，從旁邊插一個
攔截器把中間結果側錄下來。
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def get_attention(model, idx):
    """跑一次 forward，回傳每一層的 attention 權重。

    回傳：list（長度 = n_layer），每個元素是 tensor (n_head, T, T)，
    att[h, i, j] = 第 h 個頭裡，第 i 個 token「注意」第 j 個 token 的權重。
    每一列（固定 i）加總為 1（softmax）；因為因果遮罩，j > i 的位置為 0。
    """
    model.eval()
    caps: list[torch.Tensor] = []
    handles = []

    for block in model.blocks:
        attn = block.attn

        def hook(mod, inp, out, _attn=attn):
            # out = c_attn(x)：(B, T, 3*n_embd)，拆成 q/k/v
            B, T, _ = out.shape
            C = mod.out_features // 3
            q, k, v = out.split(C, dim=2)
            nh = _attn.n_head
            hd = C // nh
            q = q.view(B, T, nh, hd).transpose(1, 2)
            k = k.view(B, T, nh, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)   # 同 model.py line 47
            mask = torch.tril(torch.ones(T, T, device=out.device))
            att = att.masked_fill(mask == 0, float("-inf"))   # 同 line 48
            att = F.softmax(att, dim=-1)                       # 同 line 49
            caps.append(att[0].detach().cpu())                # 取 batch 第 0 筆

        handles.append(attn.c_attn.register_forward_hook(hook))

    model(idx)
    for h in handles:
        h.remove()
    return caps
