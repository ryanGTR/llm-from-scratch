"""Shared helpers for the training/eval stages."""

import numpy as np
import torch


def get_batch(data, block_size, batch_size, device):
    """隨機抽 batch_size 段序列。x 是輸入，y 是「往右錯一位」的目標。

    語言模型的本質：給 t0..t_{n-1}，預測 t1..t_n（每個位置都預測下一字）。
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    if device == "cuda":
        # pin_memory + non_blocking 在 GPU 上搬資料更快
        return x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
