# Pi0.5 PyTorch 模型架构分析

基于配置:
```python
TrainConfig(
    name='pi05_moz',
    model=Pi0Config(
        action_dim=32, action_horizon=50, max_token_len=200,
        dtype='bfloat16', paligemma_variant='gemma_2b',
        action_expert_variant='gemma_300m', pi05=True,
        discrete_state_input=True
    ),
    batch_size=32, ...
)
```

---

## 一、模型整体模块构成

Pi0.5 PyTorch 实现 (`PI0Pytorch`) 由以下主要模块构成：

```
PI0Pytorch (nn.Module)
├── 1. PaliGemmaWithExpertModel  (双专家联合推理模块)
│   ├── 1a. PaliGemmaForConditionalGeneration  (视觉-语言模型, ~2.5B params)
│   │   ├── SigLIP Vision Tower  (So400m/14, ~400M params)
│   │   │   └── PatchEmbedding → 27层 ViT Encoder → MultiModalProjector(1152→2048)
│   │   ├── Token Embedder  (vocab=257152, dim=2048)
│   │   └── Gemma-2B Language Model  (18层 Transformer)
│   │       ├── 每层: RMSNorm → Attention(8 heads, head_dim=256, 1 KV head) → RMSNorm → MLP(2048→16384→2048)
│   │       └── Final RMSNorm
│   │
│   └── 1b. GemmaForCausalLM (Action Expert, gemma_300m, ~311M params)
│       └── Gemma-300M Model  (18层 Transformer, 无 embed_tokens)
│           ├── 每层: AdaRMSNorm → Attention(8 heads, head_dim=256, 1 KV head) → AdaRMSNorm → MLP(1024→4096→1024)
│           └── Final AdaRMSNorm
│
├── 2. action_in_proj   (nn.Linear: 32 → 1024)    动作输入投影
├── 3. action_out_proj  (nn.Linear: 1024 → 32)     动作输出投影
├── 4. time_mlp_in      (nn.Linear: 1024 → 1024)   时间步 MLP 层1
└── 5. time_mlp_out     (nn.Linear: 1024 → 1024)   时间步 MLP 层2
```

### 各模块详细配置

| 模块 | 变体 | 隐藏维度 | 层数 | MLP维度 | 注意力头数 | KV头数 | Head Dim |
|------|------|---------|------|---------|-----------|--------|----------|
| **PaliGemma LLM** | gemma_2b | 2048 | 18 | 16,384 | 8 | 1 | 256 |
| **Action Expert** | gemma_300m | 1024 | 18 | 4,096 | 8 | 1 | 256 |
| **SigLIP Vision** | So400m/14 | 1152 | 27 | 4,304 | 16 | - | 72 |

---

## 二、数据流详细分析

### 2.1 Prefix（LLM 前缀）构造

```
输入图像 (3张 x [B, 3, 224, 224])
   │
   ├─→ SigLIP Vision Encoder (So400m/14)
   │    patch_size=14, image_size=224
   │    num_patches = (224/14)² = 256 tokens/image
   │    → [B, 256, 1152] (SigLIP hidden)
   │    → MultiModalProjector: Linear(1152 → 2048)
   │    → [B, 256, 2048] (per image)
   │
   │   3张图像共: 3 × 256 = 768 image tokens → [B, 768, 2048]
   │
   ├─→ Language Token Embedding
   │    tokenized_prompt: [B, max_token_len=200]
   │    → Embedding(257152, 2048) → × sqrt(2048)
   │    → [B, 200, 2048]
   │
   └─→ Concat → prefix_embs: [B, 968, 2048]
              prefix 总长度 = 768 (images) + 200 (language) = 968 tokens
```

### 2.2 Suffix（Action Expert 后缀）构造

```
Pi0.5 模式 (pi05=True, discrete_state_input=True):
  ─ 无 state token（state 已编码在 language prompt 中）

noisy_actions: [B, 50, 32]
   → action_in_proj: Linear(32 → 1024)
   → action_embs: [B, 50, 1024]

timestep: [B]
   → sinusoidal_pos_embedding(dim=1024)
   → time_emb: [B, 1024]
   → time_mlp_in: Linear(1024 → 1024) → SiLU
   → time_mlp_out: Linear(1024 → 1024) → SiLU
   → adarms_cond: [B, 1024]  (用于 AdaRMSNorm 调制)

suffix_embs = action_embs: [B, 50, 1024]
suffix 总长度 = 50 tokens (action_horizon)
```

### 2.3 联合注意力（双专家 Shared Attention）

**这是 LLM 到 Action Expert 数据传输的核心机制。**

两个专家在**每一层**都通过**拼接注意力**进行信息交换：

```
对于每一层 (共 18 层):

┌──────────────────────────────────────────────────────────────────┐
│  Expert 0 (PaliGemma):                                          │
│    hidden_states: [B, 968, 2048]                                │
│    → input_layernorm (标准 RMSNorm)                              │
│    → Q_proj(2048 → 8×256=2048): [B, 968, 8, 256]               │
│    → K_proj(2048 → 1×256=256):  [B, 968, 1, 256]               │
│    → V_proj(2048 → 1×256=256):  [B, 968, 1, 256]               │
│                                                                  │
│  Expert 1 (Action Expert):                                       │
│    hidden_states: [B, 50, 1024]                                  │
│    → input_layernorm (AdaRMSNorm, conditioned on adarms_cond)    │
│    → Q_proj(1024 → 8×256=2048): [B, 50, 8, 256]                │
│    → K_proj(1024 → 1×256=256):  [B, 50, 1, 256]                │
│    → V_proj(1024 → 1×256=256):  [B, 50, 1, 256]                │
│                                                                  │
│  ━━━ 拼接 (concat along sequence dim) ━━━                       │
│    Q_cat: [B, 8, 968+50=1018, 256]                              │
│    K_cat: [B, 1, 1018, 256]  (GQA: 1 KV head → broadcast to 8) │
│    V_cat: [B, 1, 1018, 256]                                     │
│                                                                  │
│  ━━━ RoPE + Attention 计算 ━━━                                   │
│    logits = Q_cat @ K_cat^T: [B, 8, 1018, 1018]                │
│    + attention_mask (prefix⇄prefix, suffix→prefix+suffix单向)    │
│    output = softmax(logits) @ V_cat: [B, 8, 1018, 256]         │
│                                                                  │
│  ━━━ 分割 (split) ━━━                                            │
│    Expert 0 attn_out: [B, 968, 8×256] → o_proj(2048→2048)      │
│    Expert 1 attn_out: [B, 50, 8×256]  → o_proj(2048→1024)      │
│                                                                  │
│  ━━━ 残差 + MLP ━━━                                              │
│    Expert 0: residual → post_attn_layernorm → MLP(2048→16384→2048) → residual │
│    Expert 1: gated_residual → AdaRMSNorm → MLP(1024→4096→1024) → gated_residual │
└──────────────────────────────────────────────────────────────────┘
```

### 2.4 注意力掩码模式

```
Attention Mask (1018 × 1018):

              ┌─── prefix (968) ────┐┌── suffix (50) ──┐
              img_0  img_1  img_2  lang  action_tokens
prefix  968 │ ■■■■  ■■■■  ■■■■  ■■■■ │  ░░░░░░░░░░░  │  prefix 互相可见，不能看 suffix
suffix   50 │ ■■■■  ■■■■  ■■■■  ■■■■ │  ▼▼▼▼▼▼▼▼▼▼▼  │  suffix 可以看 prefix + 因果自注意力

■ = 可见 (attend)
░ = 不可见 (masked)
▼ = 因果注意力 (causal, 只能看当前及之前)
```

**关键**: Action Expert 的 50 个 token **可以注意到所有 968 个 prefix token**，这是 LLM→Action Expert 信息传输的通道。

---

## 三、LLM 到 Action Expert 数据传输量分析

### 3.1 每层传输量（KV 通道）

在每一层的共享注意力中，Action Expert 通过注意力机制从 LLM 提取信息：

| 数据项 | 形状 | 数据量 (bfloat16) |
|--------|------|-------------------|
| **LLM K 向量** | [B, 1, 968, 256] | B × 968 × 256 × 2B = B × 496 KB |
| **LLM V 向量** | [B, 1, 968, 256] | B × 968 × 256 × 2B = B × 496 KB |
| **K broadcast 到 8 heads** | [B, 8, 968, 256] | B × 968 × 256 × 8 × 2B = B × 3.97 MB |
| **V broadcast 到 8 heads** | [B, 8, 968, 256] | B × 968 × 256 × 8 × 2B = B × 3.97 MB |

**每层 LLM→Expert 实际传输量**: 对于 batch_size=1:
- K + V (GQA 格式, 1 KV head): **968 × 256 × 2 = 495,616 elements = ~0.97 MB** (bfloat16)
- 经过 GQA broadcast 后实际参与计算: **968 × 256 × 8 × 2 = 3,964,928 elements = ~7.94 MB**

### 3.2 注意力矩阵（每层）

Action Expert 的 50 个 query 与 LLM 968 个 key 的交互:

| 数据项 | 形状 | 数据量 (bfloat16) |
|--------|------|-------------------|
| **Action→Prefix 注意力 logits** | [B, 8, 50, 968] | B × 8 × 50 × 968 × 2B = B × 775 KB |
| **Action→Prefix 注意力 output** | [B, 8, 50, 256] | B × 8 × 50 × 256 × 2B = B × 200 KB |

### 3.3 18 层总传输量 (batch_size=32)

```
单层 (batch_size=1):
  K 传输:   968 × 256 × 2 bytes = 495,616 B ≈ 0.47 MB
  V 传输:   968 × 256 × 2 bytes = 495,616 B ≈ 0.47 MB
  注意力矩阵: 8 × 50 × 968 × 2 bytes = 774,400 B ≈ 0.74 MB
  注意力输出: 8 × 50 × 256 × 2 bytes = 204,800 B ≈ 0.20 MB
  ─────────────────────────────────────────────
  每层小计: ≈ 1.88 MB

18 层总计 (batch_size=1): ≈ 33.8 MB
18 层总计 (batch_size=32): ≈ 1.08 GB
```

### 3.4 AdaRMSNorm 调制数据

Pi0.5 独有的 AdaRMSNorm 是另一条信息注入通道（但不是从 LLM 来的，而是从 timestep 来的）：

```
adarms_cond: [B, 1024]  (来自 timestep, 不来自 LLM)

每层 AdaRMSNorm:
  Dense(1024 → 1024×3 = 3072)
  → split → scale[B,1,1024], shift[B,1,1024], gate[B,1,1024]
  applied to: hidden_states [B, 50, 1024]

每层 × 2 (input_layernorm + post_attention_layernorm):
  modulation 数据: 2 × 3072 × 2 bytes = 12,288 B ≈ 12 KB (per sample per layer)
18 层总计: ≈ 216 KB (per sample), ≈ 6.75 MB (batch_size=32)
```

### 3.5 汇总表 (batch_size=32, bfloat16)

| 传输通道 | 方向 | 每层/样本 | 18层/batch=32 |
|----------|------|----------|---------------|
| **K vectors (GQA)** | LLM → 共享注意力 | 0.47 MB | 270 MB |
| **V vectors (GQA)** | LLM → 共享注意力 | 0.47 MB | 270 MB |
| **注意力 logits** | Expert Query × LLM Key | 0.74 MB | 425 MB |
| **注意力加权输出** | Expert 从 LLM V 提取 | 0.20 MB | 115 MB |
| **AdaRMSNorm 调制** | Timestep → Expert | 0.01 MB | 6.75 MB |
| **总计** | | **1.89 MB** | **≈ 1.09 GB** |

### 3.6 推理阶段特殊优化

推理时（`sample_actions`），LLM 的 prefix 只计算一次，KV 被缓存：

```
阶段1: Prefix 单次前向传播
  LLM 处理 968 tokens → 缓存 KV

  KV Cache 大小 (18层):
    K: 18 × [B, 1, 968, 256] × 2B = 18 × 0.47 MB = 8.53 MB (per sample)
    V: 18 × [B, 1, 968, 256] × 2B = 18 × 0.47 MB = 8.53 MB (per sample)
    总 KV Cache: ≈ 17.1 MB/sample, ≈ 546 MB (batch=32)

阶段2: 迭代去噪 (默认 10 步)
  每步只需要 Action Expert 前向:
    - 输入 suffix_embs: [B, 50, 1024]
    - 使用缓存的 prefix KV
    - 每步数据量: 仅 suffix attention (不重新计算 prefix)

  10步去噪总计 KV 访问: ~17.1 MB × 10 = 171 MB (per sample)
```

---

## 四、关键架构洞见

### 4.1 信息传输的本质

LLM 到 Action Expert 的信息传输 **不是** 传统的 cross-attention，而是通过**序列拼接共享注意力**实现的：

1. 两个专家的 Q/K/V 在序列维度上**拼接**
2. 在统一的注意力矩阵中计算交互
3. 通过 attention mask 控制信息流向（单向：LLM → Expert）
4. 每个专家使用**独立的 O_proj** 将共享注意力输出映射回各自隐空间

### 4.2 维度不对称的处理

| | LLM (gemma_2b) | Action Expert (gemma_300m) |
|---|---|---|
| 隐藏维度 | 2048 | 1024 |
| Q proj | 2048 → 2048 (8×256) | 1024 → 2048 (8×256) |
| K proj | 2048 → 256 (1×256) | 1024 → 256 (1×256) |
| V proj | 2048 → 256 (1×256) | 1024 → 256 (1×256) |
| O proj | 2048 → 2048 | 2048 → 1024 |

**关键**: 虽然两个模型隐藏维度不同 (2048 vs 1024)，但它们共享相同的 **head_dim=256** 和 **num_heads=8**，使得注意力空间维度一致（QKV 都在 256 维空间），从而实现无缝拼接。

### 4.3 Pi0.5 vs Pi0 的关键区别

| 特性 | Pi0 | Pi0.5 |
|------|-----|-------|
| 状态输入 | 连续向量投影为 1 个 suffix token | 离散语言 token（在 prefix 中） |
| 时间步嵌入 | 与 action 拼接后过 MLP | 通过 AdaRMSNorm 调制 Expert 每层归一化 |
| suffix 长度 | 1 (state) + 50 (actions) = 51 | 50 (actions only) |
| max_token_len | 48 | 200 |
| Expert 归一化 | 标准 RMSNorm | AdaRMSNorm (scale + shift + gate) |

---

## 五、模块参数量估算

| 模块 | 参数量 |
|------|--------|
| SigLIP Vision Tower (So400m/14) | ~400M |
| MultiModalProjector (1152→2048) | ~2.4M |
| Token Embedding (257152×2048) | ~527M |
| Gemma-2B LLM (18 layers) | ~1.5B |
| Gemma-300M Action Expert (18 layers) | ~311M |
| action_in_proj (32→1024) | ~33K |
| action_out_proj (1024→32) | ~33K |
| time_mlp_in + time_mlp_out (1024→1024 ×2) | ~2.1M |
| AdaRMSNorm Dense layers (1024→3072, 每层×2, 共18层) | ~113M |
| **总计** | **~2.85B** |

---

## 六、数据流图（端到端）

```
                              ┌──────────────────────────┐
                              │     3×RGB Images         │
                              │   [B, 3, 224, 224] ×3    │
                              └───────────┬──────────────┘
                                          │
                                    SigLIP Vision Tower
                                    (So400m/14, 27 layers)
                                          │
                              ┌───────────▼──────────────┐
                              │  Image Tokens             │
                              │  [B, 768, 2048]           │
                              │  (3 images × 256 patches) │
                              └───────────┬──────────────┘
                                          │
                              ┌───────────▼──────────────┐
                              │  Language Tokens           │
                              │  Embed(prompt) × sqrt(d)   │
                              │  [B, 200, 2048]            │
                              └───────────┬──────────────┘
                                          │
                              ┌───────────▼──────────────┐
                              │     PREFIX                 │
                              │  [B, 968, 2048]            │
                              │  (LLM Expert 0 input)      │
                              └───────────┬──────────────┘
                                          │
    ┌─────────────────┐                   │
    │  Noisy Actions   │                   │
    │  [B, 50, 32]     │                   │
    │       │          │                   │
    │  action_in_proj  │                   │
    │  Linear(32→1024) │                   │
    │       │          │                   │
    │  [B, 50, 1024]   │                   │
    │                  │                   │
    │  ┌─────────────┐ │                   │
    │  │ Timestep [B] │ │                   │
    │  │ → sincos emb │ │                   │
    │  │ → time_mlp   │ │                   │
    │  │ [B, 1024]    │ │                   │
    │  │ (adarms_cond)│ │                   │
    │  └──────┬──────┘ │                   │
    │         │        │                   │
    │    SUFFIX        │                   │
    │  [B, 50, 1024]   │                   │
    └────────┬────────┘                    │
             │                             │
             ▼                             ▼
    ┌─────────────────────────────────────────────────────┐
    │              18 × Shared Attention Layer             │
    │                                                     │
    │   Expert 0 (LLM):        Expert 1 (Action Expert):  │
    │   [B, 968, 2048]         [B, 50, 1024]              │
    │        │                       │                    │
    │   RMSNorm               AdaRMSNorm(adarms_cond)     │
    │   Q,K,V proj             Q,K,V proj                 │
    │        │                       │                    │
    │        └──────── concat ───────┘                    │
    │              [B, 8, 1018, 256]                      │
    │                     │                               │
    │              Shared Attention                        │
    │              (with causal mask)                      │
    │                     │                               │
    │        ┌──────── split ────────┐                    │
    │        │                       │                    │
    │   o_proj(2048→2048)      o_proj(2048→1024)          │
    │   + residual             + gated_residual           │
    │   RMSNorm                AdaRMSNorm                 │
    │   MLP(2048→16384→2048)   MLP(1024→4096→1024)        │
    │   + residual             + gated_residual           │
    │                                                     │
    │   (repeat × 18 layers)                              │
    └───────────────────────────────────┬─────────────────┘
                                        │
                              ┌─────────▼──────────────┐
                              │  Expert 1 Output         │
                              │  [B, 50, 1024]           │
                              │  (last 50 tokens)        │
                              └─────────┬──────────────┘
                                        │
                                  action_out_proj
                                  Linear(1024→32)
                                        │
                              ┌─────────▼──────────────┐
                              │  v_t (velocity pred)     │
                              │  [B, 50, 32]             │
                              └──────────────────────────┘
```
