---
title: 練習 — 親手推導 DPO 損失（你來推，我在旁糾錯）
type: exercise
created: 2026-06-21
updated: 2026-06-21
tags: [exercise, dpo, rlhf, math, self-derive]
sources:
  - docs/derivations.md
  - pipeline/06_dpo.py
---

# 練習：親手把 DPO 損失推出來

> 用法：**先別看答案**。每一步我給「目標 + 提示」，你拿紙筆（或在這頁下面打字）推推看，
> 卡住再展開 `▶ 看答案`。全部推完，你就握住了整個後訓練最漂亮的一條數學。
> 想要的話，跟我說一聲，我可以在 Telegram **一步一步當關主**帶你推、即時糾錯。
>
> 完整版證明在 [[derivations]] §1–2；這頁是「換你主導」的版本。

**你最後要推出來的目標**（先看一眼終點長怎樣，但別急著背）：

$$\mathcal{L}_\text{DPO}=-\,\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big]$$

符號：$\pi_\theta$＝要訓練的 policy、$\pi_\text{ref}$＝凍結的 SFT、$r(x,y)$＝reward、$\beta$＝KL 強度、
$(y_w\succ y_l)$＝人比較喜歡 $y_w$ 勝過 $y_l$。

---

## Step 0 — 起點（這步給你）

RLHF 想解的最佳化問題是「最大化獎勵，但別離參考模型太遠」：

$$\max_{\pi}\ \mathbb{E}_{y\sim\pi(\cdot|x)}\big[r(x,y)\big]\ -\ \beta\, D_{\mathrm{KL}}\big(\pi(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x)\big).$$

把 $D_{\mathrm{KL}}=\sum_y \pi(y)\log\frac{\pi(y)}{\pi_\text{ref}(y)}$ 展開後，整個目標只跟 $\{\pi(y)\}$ 有關。

---

## Step 1 — 解出最優 policy $\pi^*$

🎯 **目標**：在約束 $\sum_y\pi(y)=1$ 下，對上面那個目標求極大，解出 $\pi^*(y|x)$。

💡 **提示**：寫 Lagrangian（加一項 $\lambda(\sum_y\pi(y)-1)$），對單一個 $\pi(y)$ 偏微分、令為 0。
會用到 $\dfrac{\partial}{\partial\pi}\big[\pi\log\tfrac{\pi}{\pi_\text{ref}}\big]=\log\tfrac{\pi}{\pi_\text{ref}}+1$。

✍️ 你的推導：

<details><summary>▶ 看答案</summary>

$$\frac{\partial}{\partial\pi(y)}\Big[\pi(y)r(y)-\beta\,\pi(y)\log\tfrac{\pi(y)}{\pi_\text{ref}(y)}+\lambda\pi(y)\Big]=r(y)-\beta\big(\log\tfrac{\pi(y)}{\pi_\text{ref}(y)}+1\big)+\lambda=0.$$

整理 → $\log\frac{\pi(y)}{\pi_\text{ref}(y)}=\frac{r(y)+\lambda}{\beta}-1$，取指數、用歸一化把常數吸成 $Z(x)$：

$$\boxed{\ \pi^*(y|x)=\tfrac{1}{Z(x)}\,\pi_\text{ref}(y|x)\,\exp\!\big(\tfrac1\beta r(x,y)\big)\ },\quad Z(x)=\sum_y \pi_\text{ref}(y|x)\exp\!\big(\tfrac1\beta r(x,y)\big).$$

**自問**：$Z(x)$ 要對「所有可能的回答 $y$」求和——這算得出來嗎？（記住這個痛點。）
</details>

---

## Step 2 — 反解：把 $r$ 用 $\pi^*$ 表示

🎯 **目標**：把 Step 1 的式子反過來，解出 $r(x,y)=\,?$

💡 **提示**：兩邊取對數就好。

<details><summary>▶ 看答案</summary>

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x).$$

reward 被「最優 policy 對參考模型的對數機率比」表示出來了——這就是論文標題
*Your Language Model is Secretly a Reward Model* 的意思。
</details>

---

## Step 3 — 關鍵一躍（觀念題，沒有算式）

🎯 **問題**：Step 2 的 $r$ 是用「最優 policy $\pi^*$」寫的，但我們手上只有「正在訓練的 $\pi_\theta$」。
DPO 做了什麼假設，讓我們可以把 $\pi^*$ 換成 $\pi_\theta$？這個假設的代價是什麼？

<details><summary>▶ 看答案</summary>

**假設**：把 $\pi_\theta$ 當成「它自己隱含的那個 reward 的最優 policy」——即直接令 $\pi^*\!\leftarrow\!\pi_\theta$，
於是 $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)$ 變成「policy ↔ reward」的對應。
我們不再訓練一個獨立的 reward model，而是讓 policy **隱含**地當 reward。

**代價**：DPO 吃的是**靜態**偏好資料（不像 RLHF 會用當前 policy 線上取樣再打分），
所以它只能在「資料涵蓋的範圍」內學偏好，學不到資料沒出現過的行為。
</details>

---

## Step 4 — 代進 Bradley-Terry，看 $Z(x)$ 消失（魔術在這）

🎯 **目標**：偏好機率 $P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$。
把 Step 2/3 的 $r_\theta$ 代進去，化簡 $r_\theta(x,y_w)-r_\theta(x,y_l)$。

💡 **提示**：$y_w$ 和 $y_l$ 共用同一個 $x$，所以它們的 $\beta\log Z(x)$ 是**一樣的**。相減會怎樣？

<details><summary>▶ 看答案</summary>

$$r_\theta(x,y_w)-r_\theta(x,y_l)=\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}+\beta\log Z(x)-\Big[\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}+\beta\log Z(x)\Big]$$

$$=\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}.$$

**那個算不出來的 $Z(x)$ 被相減消掉了。** 這就是 DPO 不用跑 RL、不用 reward model 的根本原因——
整個方法的魔術就在這一步。
</details>

---

## Step 5 — 收尾成損失

🎯 **目標**：對偏好資料取「負對數概似」，寫出最終的 $\mathcal{L}_\text{DPO}$。

<details><summary>▶ 看答案</summary>

$$\mathcal{L}_\text{DPO}=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big].$$

對照 `pipeline/06_dpo.py`：`margin = (pc-rc)-(pr-rr)` 就是中括號裡那團、`loss = -logsigmoid(beta*margin)`
就是這個式子。你剛剛親手把那行程式碼推出來了。✅
</details>

---

## 收尾自測（四動作練習法）

不看答案，試著回答：

1. **講解**：用一句話說，DPO 為什麼不需要 reward model？（關鍵字：哪個東西被消掉了？）
2. **預測**：若拿掉 $\beta\,D_{\mathrm{KL}}$ 那一項（沒有參考模型當錨），Step 1 的 $\pi^*$ 會變怎樣？這對應到我們在 RLHF 實測看到的什麼現象？
3. **弄壞**：如果 $\pi_\theta=\pi_\text{ref}$（policy 還沒動），$\mathcal{L}_\text{DPO}$ 等於多少？（提示：$\sigma(0)$）
4. **改**：把 $\beta$ 調很小，根據 [[derivations]] §3，margin 會被推到多大的尺度？漂移會變大還小？

<details><summary>▶ 對答案</summary>

1. 偏好相減時，**配分函數 $Z(x)$（要對所有回答求和、算不出來的那個）被消掉**了，所以不必顯式建 reward、也不必跑 RL。
2. 沒有 KL 項 → $\pi^*\propto\exp(r/\beta)$，**完全不被 $\pi_\text{ref}$ 錨住** → 可以為了高獎勵任意亂跑。這正是 RLHF 實測 β=0 時的 **reward hacking / mode collapse**（「方言，的方言…」那個）。
3. margin$=0$ → $\sigma(0)=\tfrac12$ → $\mathcal{L}=-\log\tfrac12=\log 2\approx 0.693$。（這也是 `tests/test_dpo.py` 守的不變量。）
4. margin 尺度 $\sim 1/\beta$ 變**大** → policy 被推離 $\pi_\text{ref}$ 更遠 → **漂移更大**（小 β＝更激進，不是更保守）。
</details>

---

## See Also

- [[derivations]] §1–2 — 完整版證明（這頁的「答案總集」）
- `pipeline/06_dpo.py` — 你剛推出來的那行 `dpo_loss()`
- [[theory-map]] — DPO 的出處（Rafailov et al. 2023）
