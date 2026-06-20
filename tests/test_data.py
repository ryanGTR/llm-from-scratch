"""資料 pipeline 的自動驗證——別信我嘴巴說「好了」，跑這個。

用 stdlib unittest，零依賴（你 3.14 直接 `make test` 就能跑）。
每個 test 就是一條「我宣稱的事實」被機器檢查一次：
  - 清洗真的去掉 HTML / 控制字元
  - 品質過濾真的丟掉短/符號/重複文件
  - exact / near dedup 真的去重
  - 打包的 .bin 真的能 decode 回原文

Java 類比：這就是你的 JUnit。紅燈=我吹牛，綠燈=有憑有據。
"""

import unittest
from array import array

from src.data import clean as C
from src.data import dedup as D
from src.data import stats as S
from src.tokenizer import CharTokenizer


class TestClean(unittest.TestCase):
    def test_strip_html(self):
        out = C.normalize_text("<p>hello <b>world</b></p>")
        self.assertNotIn("<", out)
        self.assertIn("hello", out)

    def test_strip_control_chars(self):
        out = C.normalize_text("a\x00b\x07c")
        self.assertNotIn("\x00", out)
        self.assertNotIn("\x07", out)
        self.assertIn("abc", out.replace(" ", ""))

    def test_collapse_whitespace(self):
        self.assertEqual(C.normalize_text("a    b"), "a b")


class TestQuality(unittest.TestCase):
    def setUp(self):
        self.cfg = C.QualityConfig()

    def test_too_short_rejected(self):
        ok, why = C.quality_check("hi", self.cfg)
        self.assertFalse(ok)
        self.assertEqual(why, "too_short")

    def test_too_repetitive_rejected(self):
        ok, why = C.quality_check("a" * 200, self.cfg)
        self.assertFalse(ok)
        self.assertEqual(why, "too_repetitive")

    def test_too_many_symbols_rejected(self):
        ok, why = C.quality_check("###@@@$$$%%%^^^&&&***!!!" * 3, self.cfg)
        self.assertFalse(ok)
        self.assertEqual(why, "too_many_symbols")

    def test_normal_text_kept(self):
        ok, why = C.quality_check(
            "A normal English sentence with enough length to pass.", self.cfg)
        self.assertTrue(ok)
        self.assertEqual(why, "ok")


class TestDedup(unittest.TestCase):
    def test_exact_dedup_drops_identical(self):
        texts = ["hello world", "hello   world", "different text here"]
        keep, dropped = D.exact_dedup(texts)
        self.assertEqual(dropped, 1)          # 只差空白也算同一篇
        self.assertEqual(keep, [0, 2])

    def test_near_dedup_catches_minor_edit(self):
        # near-dup 是機率性的：文件越長、改動越少，估出的相似度越穩。
        # 用一段夠長的文 + 只改一個字，相似度會明確高於門檻（不是壓線）。
        base = (
            "self attention lets every token in a sequence look at every other "
            "token and decide how much to focus on each one this is computed "
            "with queries keys and values the dot product of a query and a key "
            "gives an attention score the scores are normalized with a softmax "
            "and the output is a weighted sum of the values across all positions"
        )
        near = base.replace("focus on each one", "weight each one")  # 改一處
        far = ("quantum chromodynamics describes the strong interaction between "
               "quarks and gluons inside hadrons such as protons and neutrons "
               "and explains why colored particles are never seen in isolation")
        keep, dropped = D.near_dedup([base, near, far], D.NearDupConfig())
        self.assertEqual(dropped, 1)          # near 被當成 base 的重複丟掉
        self.assertIn(0, keep)
        self.assertIn(2, keep)                # 完全不相關的留著


class TestPackRoundTrip(unittest.TestCase):
    def test_uint32_holds_large_token_ids(self):
        # ⑧：uint16 上限 65535，大 vocab（如真實 BPE 10-20 萬）要用 uint32 才不溢位
        big = [70000, 150000, 200000]
        a = array("I", big)              # I = uint32（x86-64 上 4 bytes）
        self.assertEqual(a.itemsize, 4)
        b = array("I")
        b.frombytes(a.tobytes())
        self.assertEqual(list(b), big)   # 大於 65535 的 id 無損存取

    def test_tokenize_pack_decode(self):
        text = "hello GPT, this is a round-trip test.\n"
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        # 模擬 pack：寫成 uint16 再讀回
        buf = array("H", ids).tobytes()
        back = array("H")
        back.frombytes(buf)
        self.assertEqual(tok.decode(list(back)), text)


class TestBPE(unittest.TestCase):
    def test_merge_replaces_pair(self):
        from src.bpe import merge
        # [1,2,1,2,3] 把 (1,2) 換成 99 -> [99,99,3]
        self.assertEqual(merge([1, 2, 1, 2, 3], (1, 2), 99), [99, 99, 3])

    def test_train_learns_frequent_pair(self):
        from src.bpe import train_bpe
        # "ababab" 裡 (a,b) 最常見，第一步就該合併它，序列變短
        res = train_bpe("abababab", num_merges=1)
        self.assertEqual(len(res["log"]), 1)
        self.assertEqual(res["log"][0]["pair"], ["a", "b"])
        self.assertEqual(res["log"][0]["merged"], "ab")
        self.assertLess(res["log"][0]["seq_len"], 8)   # 8 -> 4

    def test_tokenizer_roundtrip_and_saveload(self):
        import os
        import tempfile
        from src.bpe import BPETokenizer
        text = "the cat sat on the mat. the cat ran. the mat is flat."
        tok = BPETokenizer.from_text(text, num_merges=20)
        self.assertGreater(tok.vocab_size, len(set(text)))   # 有長出新 token
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)              # 無損還原
        self.assertLess(len(ids), len(text))                 # 比 char 短
        p = tempfile.mktemp(suffix=".json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
        os.remove(p)
        self.assertEqual(tok2.decode(tok2.encode(text)), text)
        self.assertEqual(tok2.vocab_size, tok.vocab_size)


class TestLossLog(unittest.TestCase):
    def test_load_run_parses_csv(self):
        import tempfile, os
        from pipeline.plot_loss import load_run
        p = tempfile.mktemp(suffix=".csv")
        with open(p, "w") as f:
            f.write("step,train_loss,val_loss\n0,4.2,4.2\n200,2.7,2.8\n")
        steps, train, val = load_run(p)
        os.remove(p)
        self.assertEqual(steps, [0, 200])
        self.assertEqual(val, [4.2, 2.8])


class TestModernComponents(unittest.TestCase):
    def test_rmsnorm_normalizes_to_unit_rms(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.model import RMSNorm
        x = torch.randn(4, 16) * 5 + 3
        out = RMSNorm(16)(x)                       # weight 初始為 1
        rms = out.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones(4), atol=1e-3))

    def test_both_norms_build_and_run(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT
        for flag in (False, True):
            cfg = GPTConfig(vocab_size=20, n_layer=2, n_head=2, n_embd=16,
                            block_size=8, use_rmsnorm=flag)
            idx = torch.randint(0, 20, (1, 8))
            _, loss = GPT(cfg)(idx, idx)
            self.assertTrue(loss.item() > 0)

    def test_swiglu_param_parity_and_runs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT

        def mk(flag):
            return GPT(GPTConfig(vocab_size=20, n_layer=4, n_head=2, n_embd=128,
                                 block_size=8, use_swiglu=flag))
        mlp, swiglu = mk(False), mk(True)
        # SwiGLU 用 8/3·n_embd 寬度 → 參數量跟 GELU MLP 相差 < 2%（公平對比）
        ratio = swiglu.num_params() / mlp.num_params()
        self.assertLess(abs(ratio - 1.0), 0.02)
        idx = torch.randint(0, 20, (1, 8))
        self.assertTrue(swiglu(idx, idx)[1].item() > 0)

    def test_rope_is_norm_preserving_and_drops_posemb(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT, build_rope_cache, apply_rope
        cos, sin = build_rope_cache(16, 8)
        x = torch.randn(1, 2, 8, 16)
        rx = apply_rope(x, cos, sin)
        # 旋轉保長度；位置 0 不動（角度=0）
        self.assertTrue(torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-4))
        self.assertTrue(torch.allclose(rx[:, :, 0], x[:, :, 0], atol=1e-5))
        # use_rope 時模型不再有學習式 pos_emb
        m = GPT(GPTConfig(vocab_size=20, n_layer=2, n_head=2, n_embd=16,
                          block_size=8, use_rope=True))
        self.assertIsNone(m.pos_emb)
        self.assertTrue(m(torch.randint(0, 20, (1, 8)), torch.randint(0, 20, (1, 8)))[1].item() > 0)

    def test_gqa_fewer_params_and_runs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT

        def mk(nkv):
            return GPT(GPTConfig(vocab_size=20, n_layer=2, n_head=4, n_embd=64,
                                 block_size=8, n_kv_head=nkv))
        mha, gqa, mqa = mk(0), mk(2), mk(1)   # 0 = 標準 MHA
        # kv 頭越少 → 參數越少（k/v 投影變小）
        self.assertLess(gqa.num_params(), mha.num_params())
        self.assertLess(mqa.num_params(), gqa.num_params())
        idx = torch.randint(0, 20, (1, 8))
        for m in (gqa, mqa):                    # 都能正常 forward
            self.assertTrue(m(idx, idx)[1].item() > 0)

    def test_flash_matches_naive(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT
        kw = dict(vocab_size=30, n_layer=3, n_head=4, n_embd=64, block_size=32)
        naive = GPT(GPTConfig(use_flash=False, **kw)).eval()
        flash = GPT(GPTConfig(use_flash=True, **kw)).eval()
        flash.load_state_dict(naive.state_dict())   # 同權重
        idx = torch.randint(0, 30, (2, 16))
        with torch.no_grad():
            # FlashAttention 與樸素版數學等價 → 同權重下輸出幾乎相同
            self.assertTrue(torch.allclose(naive(idx)[0], flash(idx)[0], atol=1e-4))


class TestAttention(unittest.TestCase):
    def test_attention_shapes_and_properties(self):
        # 只在有裝 torch 時測（資料 pipeline 本身不需要 torch）
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT
        from src.viz import get_attention

        cfg = GPTConfig(vocab_size=20, n_layer=2, n_head=2, n_embd=16, block_size=8)
        model = GPT(cfg)
        idx = torch.randint(0, 20, (1, 5))      # T=5
        atts = get_attention(model, idx)
        self.assertEqual(len(atts), cfg.n_layer)          # 每層一個
        self.assertEqual(tuple(atts[0].shape), (cfg.n_head, 5, 5))
        # 每一列（query）softmax 後加總為 1
        self.assertTrue(torch.allclose(atts[0].sum(-1), torch.ones(cfg.n_head, 5), atol=1e-5))
        # 因果性：第 0 個字不能看未來（j>0 全為 0）
        self.assertTrue(torch.allclose(atts[0][:, 0, 1:], torch.zeros(cfg.n_head, 4)))


class TestSampling(unittest.TestCase):
    def test_topk_topp_minp_all_produce_valid_ids(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT
        m = GPT(GPTConfig(vocab_size=30, n_layer=2, n_head=2, n_embd=32,
                          block_size=16)).eval()
        start = torch.zeros((1, 1), dtype=torch.long)
        for kw in (dict(top_k=5), dict(top_p=0.9), dict(min_p=0.05)):
            out = m.generate(start, 12, **kw)
            self.assertEqual(tuple(out.shape), (1, 13))
            self.assertTrue(0 <= out.min() and out.max() < 30)

    def test_kv_cache_matches_no_cache(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安裝")
        from src.config import GPTConfig
        from src.model import GPT
        # 含 RoPE/GQA/Flash 的現代設定也要對（offset/mask 都要正確）
        m = GPT(GPTConfig(vocab_size=30, n_layer=3, n_head=4, n_embd=64,
                          block_size=64, use_rope=True, n_kv_head=2, use_flash=True)).eval()
        start = torch.randint(0, 30, (1, 5))
        a = m.generate(start, 30, top_k=1, use_kv_cache=False)   # greedy=確定性
        b = m.generate(start, 30, top_k=1, use_kv_cache=True)
        self.assertTrue(torch.equal(a, b))   # 快取版必須逐 token 完全相同


class TestStats(unittest.TestCase):
    def test_entropy_single_char_is_zero(self):
        self.assertEqual(S.char_entropy("aaaa"), 0.0)

    def test_entropy_two_equal_chars_is_one_bit(self):
        self.assertAlmostEqual(S.char_entropy("abab"), 1.0)

    def test_repetitive_compresses_more_than_diverse(self):
        repetitive = S.compression_ratio("ab" * 500)
        diverse = S.compression_ratio("".join(chr(33 + i % 90) for i in range(1000)))
        self.assertLess(repetitive, diverse)   # 越重複壓縮比越小


if __name__ == "__main__":
    unittest.main(verbosity=2)
