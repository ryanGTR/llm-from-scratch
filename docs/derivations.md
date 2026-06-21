---
title: 關鍵演算法與數學推導 — 從「會用」到「知道為什麼」
type: derivations
created: 2026-06-21
updated: 2026-06-21
tags: [llm, math, proofs, rlhf, dpo, attention, derivations]
sources:
  - src/model.py
  - src/reward_model.py
  - pipeline/06_dpo.py
  - pipeline/08_grpo.py
  - src/data/dedup.py
  - "Rafailov et al. 2023 — Direct Preference Optimization"
  - "Su et al. 2021 — RoFormer (RoPE)"
---

# 關鍵演算法與數學推導

本頁把專案裡「推導本身有 aha、又扣得上實作」的數學寫清楚。每條的格式：
**命題 → 推導 → 這在說什麼（含 Java 直覺）→ 對應程式碼**。和 [[theory-map]] 互補：
theory-map 告訴你「哪塊出自哪篇論文」，本頁告訴你「為什麼成立」。

數學以 GitHub 的 `$…$` / `$$…$$` 顯示。符號：$\pi_\theta$＝policy、$\pi_\text{ref}$＝凍結的
參考模型（SFT）、$x$＝prompt、$y$＝回答、$r(x,y)$＝reward、$\beta$＝KL 約束強度。

---

## 一、後訓練的數學核心（皇冠）

這三條是一條鏈：**先有 RLHF 的約束最優解 → DPO 從中掉出封閉式 → 同一套解釋實測的 $\beta$ 行為**。

### 1. KL 約束下的最優 policy（整條鏈的根）

**命題.** RLHF 要解的最佳化是「最大化獎勵、但別離參考模型太遠」：

$$\max_{\pi}\ \mathbb{E}_{y\sim\pi(\cdot|x)}\big[r(x,y)\big]\ -\ \beta\, D_{\mathrm{KL}}\big(\pi(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x)\big)$$

其閉式最優解是

$$\boxed{\ \pi^*(y|x)=\frac{1}{Z(x)}\,\pi_\text{ref}(y|x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)\ },\qquad Z(x)=\sum_{y}\pi_\text{ref}(y|x)\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big).$$

**推導.** 固定 $x$、把 $\pi(y)\equiv\pi(y|x)$ 當變數，加上歸一化約束 $\sum_y\pi(y)=1$，寫 Lagrangian：

$$\mathcal{L}=\sum_y \pi(y) r(y)-\beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_\text{ref}(y)}+\lambda\Big(\sum_y\pi(y)-1\Big).$$

對 $\pi(y)$ 偏微分並令為 0（用 $\frac{\partial}{\partial \pi}\,\pi\log\frac{\pi}{\pi_\text{ref}}=\log\frac{\pi}{\pi_\text{ref}}+1$）：

$$r(y)-\beta\Big(\log\tfrac{\pi(y)}{\pi_\text{ref}(y)}+1\Big)+\lambda=0
\ \Longrightarrow\ \log\tfrac{\pi(y)}{\pi_\text{ref}(y)}=\tfrac{r(y)}{\beta}+\tfrac{\lambda}{\beta}-1.$$

取指數得 $\pi(y)\propto \pi_\text{ref}(y)\exp(r(y)/\beta)$，再用 $\sum_y\pi(y)=1$ 把常數吸進 $Z(x)$。$\square$

**這在說什麼.** 最優 policy＝「拿參考模型當底，再按獎勵做指數加權」。$\beta$ 大 → 指數被壓平
→ 貼著 $\pi_\text{ref}$；$\beta$ 小 → 獎勵主導 → 敢大改。**這就是 KL 錨的數學本體**：不是外加的
trick，是「獎勵 − β·KL」這個目標的解長這樣。$Z(x)$（配分函數）要對所有可能回答求和，**算不出來**
——記住這個痛點，DPO 的魔術就是把它消掉。

> Java 直覺：像在「已核准的 baseline 設定檔」上，按分數做加權覆寫；$\beta$ 是「允許偏離 baseline 多少」的旋鈕。

---

### 2. DPO 封閉式：把 RL 和 reward model 一起消掉（皇冠）

**命題.** 給偏好資料 $(x,y_w\succ y_l)$，下式不需 reward model、不需 RL，等價於「先學 reward 再用上式做 RLHF」：

$$\boxed{\ \mathcal{L}_\text{DPO}=-\,\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big]\ }$$

**推導.** 從第 1 條的最優解反解 $r$：對 $\pi^*(y|x)=\frac1{Z}\pi_\text{ref}\,e^{r/\beta}$ 取對數、整理，

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x).$$

DPO 的關鍵一步：**把待訓練的 $\pi_\theta$ 當成「它自己隱含獎勵的最優 policy」**，於是上式給出
「policy ↔ reward」的對應（這就是論文標題*Your LM is Secretly a Reward Model*）。把它代進偏好模型
（見下方 Bradley-Terry），$\beta\log Z(x)$ **同 $x$、對 $y_w,y_l$ 相同 → 相減時抵消**：

$$r(x,y_w)-r(x,y_l)=\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}.$$

代入 $P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$、取負對數概似即得 $\mathcal{L}_\text{DPO}$。$\square$

**這在說什麼.** 那個算不出來的 $Z(x)$ **被相減消掉了**——這就是 DPO 不用跑 RL、不用 reward model
的根本原因。代價：它吃的是**靜態**偏好對（不像 RL 會線上取樣），所以學不到資料沒涵蓋的行為。

> 對應程式碼：`pipeline/06_dpo.py` 的 `dpo_loss()`。`margin = (pc-rc)-(pr-rr)` 就是上式中括號裡那團
> log-ratio 差，`loss = -logsigmoid(beta*margin)` 一字不差。`seq_logp()` 算的就是 $\log\pi(y|x)$（只在
> 回答段加總，prompt 遮掉）。

> **Bradley-Terry 設定（小引理）.** 偏好機率 $P(y_w\succ y_l)=\dfrac{e^{r_w}}{e^{r_w}+e^{r_l}}=\sigma(r_w-r_l)$。
> reward model（`src/reward_model.py` 的 `bt_loss`）直接用它；DPO 把 $r$ 換成上面的 log-ratio。
> 兩者同源——**唯一差別是「$r$ 是獨立模型的輸出，還是 policy 的對數機率比」**。

---

### 3. 為什麼 margin 的尺度 ≈ $1/\beta$（接你實測到的反直覺）

**命題.** DPO 對單筆偏好對的損失 $\ell=-\log\sigma(\beta m)$，其中
$m=\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}$ 是 log-ratio margin。
梯度在 $m\gtrsim 1/\beta$ 之後才開始消失，所以**$1/\beta$ 是「DPO 還在用力推」的 margin 尺度**。

**推導.** 對 $m$ 微分：

$$\frac{\partial \ell}{\partial m}=-\beta\,\sigma(-\beta m)=-\beta\big(1-\sigma(\beta m)\big).$$

恆為負（一直把 $m$ 往上推），大小 $=\beta\,\sigma(-\beta m)$：

- $\beta m\ll 1$（$m$ 還小）：$\sigma(-\beta m)\approx\tfrac12$，梯度 $\approx\beta/2$——**還在用力**。
- $\beta m\gg 1$（$m> 1/\beta$）：$\sigma(-\beta m)\to 0$，梯度 $\to 0$——**飽和、不再推**。

所以 DPO 會把 $m$ 一直推到 $\beta m=O(1)$、即 $m\sim 1/\beta$ 這個尺度才鬆手。$\square$

**這在說什麼（扣實測）.** $\beta$ 越**小** → 飽和尺度 $1/\beta$ 越**大** → DPO 把 log-ratio 推得越遠 →
policy 離 $\pi_\text{ref}$ 越遠（**漂移越大**）。這正是 `make dpo-beta` 量到的反直覺結果：

| $\beta$ | 飽和尺度 $1/\beta$ | 實測漂移 \|Δlogπ\| |
|---|---|---|
| 0.02 | 50 | 10.08 |
| 0.1 | 10 | 2.88 |
| 0.5 | 2 | 2.07 |

**誠實標注**：$1/\beta$ 是「梯度開始消失的*尺度*」，不是模型一定停在那個精確值——實際停在哪還受
學習率、步數、AdamW、margin 是對多個 token 加總等影響（所以上表不是嚴格等於 $1/\beta$，但**單調趨勢
完全吻合**）。重點是方向：小 $\beta$＝更激進，不是更保守。

> 對應程式碼：`scripts/dpo_beta_sweep.py`。

---

## 二、地基

### 4. Policy Gradient 定理 + 為什麼 GRPO 的「組內平均」當 baseline 不偏

**命題（PG 定理）.** $\nabla_\theta\,\mathbb{E}_{y\sim\pi_\theta}[R(y)]=\mathbb{E}_{y\sim\pi_\theta}\big[R(y)\,\nabla_\theta\log\pi_\theta(y)\big].$

**推導.** 用 log-derivative 技巧 $\nabla\pi=\pi\nabla\log\pi$：

$$\nabla_\theta\sum_y \pi_\theta(y)R(y)=\sum_y R(y)\nabla_\theta\pi_\theta(y)=\sum_y R(y)\,\pi_\theta(y)\,\nabla_\theta\log\pi_\theta(y)=\mathbb{E}\big[R\,\nabla\log\pi_\theta\big].\ \square$$

**命題（baseline 不偏）.** 對任何與動作無關的 $b$，$\mathbb{E}\big[(R-b)\nabla\log\pi_\theta\big]=\mathbb{E}\big[R\,\nabla\log\pi_\theta\big]$。

**推導.** 只需證 $\mathbb{E}[\nabla\log\pi_\theta]=0$：

$$\mathbb{E}_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)]=\sum_y\pi_\theta\,\frac{\nabla\pi_\theta}{\pi_\theta}=\sum_y\nabla\pi_\theta=\nabla\sum_y\pi_\theta=\nabla 1=0.\ \square$$

**這在說什麼.** 減掉一個 baseline 不改變梯度的期望（不偏），但能大幅**降低變異數**→訓練更穩。
經典做法（PPO/A2C）要另外**學一個 value 網路 $V(s)$** 當 baseline；**GRPO 的省事之處＝直接拿「同一個
prompt 取樣那組 $K$ 個回答的平均獎勵」當 baseline**，所以不需要 critic。優勢

$$A_i=\frac{r_i-\operatorname{mean}(r_{1..K})}{\operatorname{std}(r_{1..K})}$$

裡，**減 mean** 是上面證明的不偏 baseline；**除 std** 只是把尺度正規化（等效於自適應步長）。

> 對應程式碼：`pipeline/08_grpo.py`，`adv = (r - r.mean())/(r.std()+eps)`；`resp_logp()` 提供 $\log\pi_\theta(y)$，
> 損失 $-A\cdot\log\pi_\theta$ 就是 PG。`tests/test_rlhf.py` 驗 advantage 的 mean≈0、std≈1。

---

### 5. softmax + cross-entropy 的梯度 = 「預測機率 − one-hot」

**命題.** 對 logits $z$、真實類別 $t$，$L=-\log\operatorname{softmax}(z)_t$，則
$\dfrac{\partial L}{\partial z_i}=p_i-\mathbf{1}[i=t]$，其中 $p=\operatorname{softmax}(z)$。

**推導.** $L=-z_t+\log\sum_j e^{z_j}$，故

$$\frac{\partial L}{\partial z_i}=-\mathbf{1}[i=t]+\frac{e^{z_i}}{\sum_j e^{z_j}}=p_i-\mathbf{1}[i=t].\ \square$$

**這在說什麼.** 整個語言模型訓練的梯度訊號就這麼乾淨：**把真實 token 的 logit 往上推、其餘往下壓，
力道正比於「現在錯多少」**（$p_i$ 離目標多遠）。模型對了（$p\to$ one-hot）梯度自然趨 0。

> 對應程式碼：`src/model.py` 的 `forward()`（`F.cross_entropy`）；這是 `pipeline/02_train.py` 整支訓練的核心訊號。

> **附帶引理 $D_{\mathrm{KL}}\ge 0$（KL 錨之所以是「拉力」）.** 由 Jensen 不等式（$-\log$ 為凸）：
> $D_{\mathrm{KL}}(P\|Q)=\mathbb{E}_P[-\log\frac{Q}{P}]\ge -\log\mathbb{E}_P[\frac{Q}{P}]=-\log\sum_x Q(x)=0$，
> 等號 $\iff P=Q$。所以 $\beta\,D_{\mathrm{KL}}$ 是個 $\ge 0$、只在 $\pi_\theta=\pi_\text{ref}$ 時為 0 的懲罰——它真的把 policy 往 $\pi_\text{ref}$ 拉。

---

## 三、架構彩蛋

### 6. RoPE：為什麼「旋轉」就給出「相對位置」

**命題.** 把位置 $m$ 的 query 旋轉 $R_m$、位置 $n$ 的 key 旋轉 $R_n$（$R_m$＝各 2 維子空間轉 $m\theta_k$ 的
block-diagonal 旋轉矩陣），則內積只跟**相對位置 $n-m$** 有關：

$$\langle R_m q,\,R_n k\rangle = q^\top R_m^\top R_n\, k = q^\top R_{\,n-m}\, k.$$

**推導.** 旋轉矩陣正交且可加：$R_a^\top=R_{-a}$、$R_aR_b=R_{a+b}$，故 $R_m^\top R_n=R_{-m}R_n=R_{n-m}$。
代入即得；對每個 2 維子空間 $R_{n-m}$ 只含 $\cos((n-m)\theta_k),\sin((n-m)\theta_k)$。$\square$

**這在說什麼.** attention 分數 $q\!\cdot\!k$ 自動變成「**相對距離**的函數」，不靠任何學習式位置向量。
這就是 RoPE 外推較好（看的是相對距離）、且不需 position embedding 參數的原因——也對得上你實測的
「train@64 → eval 256，RoPE 幾乎不掉、學習式位置一過 64 就爆」。

> 對應程式碼：`src/model.py` 的 `build_rope_cache()`／`apply_rope()`（偶奇維配對旋轉），`use_rope`。

---

### 7. attention 為什麼要除以 $\sqrt{d}$

**命題.** 若 $q,k\in\mathbb{R}^d$ 各維獨立、零均值單位變異，則 $q\!\cdot\!k$ 的變異數為 $d$；除以 $\sqrt{d}$ 把它正規化回 $1$。

**推導.** $q\!\cdot\!k=\sum_{i=1}^d q_ik_i$，各項獨立、$\mathbb{E}[q_ik_i]=0$、$\operatorname{Var}(q_ik_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1$。
故 $\operatorname{Var}(q\!\cdot\!k)=d$、標準差 $\sqrt{d}$。除以 $\sqrt{d}$ 後變異數回到 $1$。$\square$

**這在說什麼.** 不除的話，$d$ 一大、logits 的尺度就 $\sim\!\sqrt d$ 變很大 → softmax 飽和成幾乎 one-hot →
梯度趨 0、難訓練。除 $\sqrt{d}$ 讓 softmax 不管維度多大都待在「有梯度」的區間。這就是論文裡那個不起眼
的 *scaled* dot-product 的理由。

> 對應程式碼：`src/model.py` 的 attention，`(q @ k.transpose(-2,-1)) / sqrt(head_dim)`。

---

### 8. MinHash 碰撞機率 = Jaccard 相似度；LSH 的 S 曲線

**命題（MinHash）.** 對隨機排列 $\pi$、集合 $A,B$，
$\Pr\big[\min\pi(A)=\min\pi(B)\big]=\dfrac{|A\cap B|}{|A\cup B|}=\operatorname{Jaccard}(A,B).$

**推導.** 在 $A\cup B$ 上，$\pi$ 把哪個元素排到最小是均勻的。兩集合的 minhash 相等 $\iff$ 那個「全域最小」
元素同時落在 $A$ 和 $B$（即在 $A\cap B$）。該事件機率 $=|A\cap B|/|A\cup B|$。$\square$

**命題（LSH banding S 曲線）.** 把 $n$ 個 minhash 切成 $b$ 個 band、每 band $r$ 行（$n=br$）。兩文件至少
共用一個 band（→成為候選對）的機率為

$$P(s)=1-(1-s^{r})^{b},\qquad s=\operatorname{Jaccard}.$$

**推導.** 單一 band 的 $r$ 個 hash 全相等：機率 $s^{r}$；該 band 不全相等：$1-s^{r}$；$b$ 個 band 都不中：
$(1-s^{r})^{b}$；至少一個中：取補。$\square$

**這在說什麼.** $P(s)$ 是一條對 $s$ 的 **S 形曲線**，拐點（門檻）約在 $s\approx(1/b)^{1/r}$。調 $b,r$ 就能把
「相似度高於門檻的對」幾乎全抓出來、低於的幾乎全濾掉——於是近似去重從 $O(n^2)$ 兩兩比，變成「只比
同 band 的候選對」、接近 $O(n)$。這就是你那 11k 篇中文維基能 21 秒去完的數學原因。

> 對應程式碼：`src/data/dedup.py`（字元 n-gram → MinHash 簽章 → LSH banding 找候選 → 再算真實 Jaccard 確認）。

---

## See Also

- [[exercise-dpo-derivation]] — **練習版**：把皇冠那條 DPO 推導改成「你來推、答案摺疊」的自推練習
- [[theory-map]] — 哪塊程式碼出自哪篇論文（本頁是「為什麼成立」，那頁是「出處」）
- `pipeline/06_dpo.py` / `pipeline/08_grpo.py` / `src/reward_model.py` — 後訓練實作
- `src/model.py` — attention / RoPE / softmax-CE 的所在
- `README.md` §9–§10 — 後訓練實驗的實測結果（這些推導的「跑出來長怎樣」）
- Rafailov et al. 2023（DPO）；Su et al. 2021（RoPE）；Vaswani 2017（scaled attention）；Broder 1997（MinHash）
