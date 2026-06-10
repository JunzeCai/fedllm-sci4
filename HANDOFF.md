# SPECTRA-FedCore 服务器交接文档

> **收件人**：龙虾（服务器端实验负责人）  
> **日期**：2026-06-10  
> **仓库**：https://github.com/JunzeCai/fedllm-sci4  
> **项目状态**：数据准备层已完成，核心训练框架待实现  

---

## 1. 项目一句话定位

**SPECTRA-FedCore** 是一个面向 IIoT（工业物联网）边缘入侵检测的**可审计、客户端差分隐私联邦学习框架**。核心思想是：只利用**公开的小语言模型骨干权重**做奇异值分解（SVD）来构造适配器基，训练极小的层-wise 核心矩阵，在客户端本地裁剪并加噪后上传，从而在不泄露原始遥测数据的前提下，协作提升边缘安全助手的检测能力。

**不要把这个项目理解为"用 LLM 打败 XGBoost"**。论文的核心主张是：**在客户端发布 DP 的强隐私设定下，公开谱结构能多大程度恢复联邦适配的效用**。

---

## 2. 当前完成状态（截至 2026-06-10）

### ✅ 已完成

| 模块 | 状态 | 说明 |
|---|---|---|
| **研究设计文档** | ✅ | `docs/superpowers/specs/` 下 1313 行的 v0.3 文献校准版，包含完整数学公式、实验矩阵、消融设计、风险登记册 |
| **数据准备层** | ✅ | `src/fedllm_data/` + `tests/`，8 个 pytest 全部通过 |
| **Edge-IIoTset 数据集** | ✅ | 已下载并解压到 `data/raw/edgeiiotset/full/`，包含 14 个攻击 CSV、10 个正常 CSV、2 个合并 CSV |
| **SNLI 数据集** | ✅ | 已下载并解压到 `data/raw/snli/snli_1.0/`，用于 Fed-SB 基线复现 |
| **数据产物** | ✅ | `data/processed/` 下已生成清单、拆分方案、标签清单、提示样本 |
| **Fed-SB 基线代码** | ✅ | 已克隆到 `data/external/fed-sb/`，commit `e9e64982`，SNLI 已 symlink 到官方位置 |
| **数据集准备 CLI** | ✅ | `scripts/prepare_datasets.py` 可一键重新生成所有产物 |

### ⏳ 待实现（接下来的关键路径）

| 优先级 | 任务 | 预估工作量 | 阻塞项 |
|---|---|---|---|
| **P0** | 复现 Fed-SB 官方 SNLI 隐私联邦实验 | 1-2 天 | 服务器 GPU 环境 |
| **P0** | 实现公开谱基构造模块（SVD → 正交基 → 核心参数化） | 2-3 天 | 需要 PyTorch + transformers + PEFT |
| **P0** | 实现单服务器 FL 模拟框架（K=10，IID + Dirichlet Non-IID） | 2-3 天 | 需要确定 FL 引擎（自研 vs Flower） |
| **P1** | 实现客户端 DP 协议（裁剪、高斯噪声、RDP 会计） | 2-3 天 | 需要 opacus 或自研 accountant |
| **P1** | 层-wise 自适应秩/噪声分配 | 1-2 天 | 依赖 P0 的谱基模块 |
| **P1** | 本地残差个性化 | 1 天 | 依赖 P0 的核心训练框架 |
| **P2** | Edge-IIoT 非 DP 冒烟测试 | 0.5 天 | 依赖 P0 的谱基模块 |
| **P2** | 非 LLM IDS 基线（XGBoost、RF、FT-Transformer） | 2-3 天 | 可用 sklearn + pytorch-tabular |
| **P3** | 完整实验矩阵 + 论文图表 | 1-2 周 | 依赖所有 P0/P1 模块 |

---

## 3. 仓库结构速览

```
fedllm-sci4/
├── HANDOFF.md                          ← 本文档
├── pyproject.toml                      # 项目元数据，pytest 配置
├── .gitignore                          # 排除了 data/raw/ 和 data/external/
│
├── configs/data/
│   ├── edgeiiot.json                   # Edge-IIoT 拆分种子、比例
│   └── snli.json                       # SNLI 路径配置
│
├── data/
│   ├── README.md                       # 数据集清单、SHA-256、下载说明
│   ├── processed/                      # 生成的产物（已提交到 git）
│   │   ├── edgeiiot/
│   │   │   ├── file_manifest.json
│   │   │   ├── source_split_seed20260531.json
│   │   │   ├── label_inventory.json
│   │   │   └── prompt_smoke_samples.jsonl
│   │   └── snli/manifest.json
│   ├── raw/                            # ← 原始数据集（未提交，需自行准备）
│   │   ├── edgeiiotset/full/           # Edge-IIoTset 完整解压内容
│   │   └── snli/snli_1.0/              # SNLI 1.0 解压内容
│   └── external/fed-sb/                # ← Fed-SB 基线代码（未提交，需自行克隆）
│
├── docs/superpowers/
│   ├── specs/2026-05-31-spectra-dp-fedcore-design.md   ← 研究设计核心文档
│   └── plans/2026-05-31-data-preparation.md            ← 数据准备实施计划（已完成）
│
├── scripts/
│   ├── prepare_datasets.py             # 数据集准备 CLI
│   ├── run_fedsb_snli_fed_private.sh   # Fed-SB 联邦隐私复现脚本
│   └── run_fedsb_snli_central_private.sh # Fed-SB 集中式隐私复现脚本
│
├── src/fedllm_data/                    # 数据准备 Python 包
│   ├── edgeiiot.py                     # 文件发现、源拆分、标签清单
│   ├── labels.py                       # 15 类标签标准化
│   ├── prompts.py                      # 指令提示渲染
│   └── snli.py                         # SNLI 清单构造
│
└── tests/                              # 8 个 pytest 测试
    ├── test_edgeiiot_*.py
    ├── test_labels.py
    ├── test_prepare_datasets_cli.py
    ├── test_prompts.py
    └── test_snli_manifest.py
```

---

## 4. 服务器环境准备

### 4.1 克隆仓库

```bash
git clone https://github.com/JunzeCai/fedllm-sci4.git
cd fedllm-sci4
```

### 4.2 Python 环境

项目要求 **Python ≥ 3.10**。建议用 conda/venv：

```bash
python3 -m venv .venv
source .venv/bin/activate

# 数据准备层依赖（当前已足够跑通测试）
pip install pytest

# 后续核心训练需要（请根据实际安装情况逐步添加）
pip install torch transformers peft accelerate bitsandbytes datasets
pip install opacus  # DP 会计（可选，也可自研）
pip install scikit-learn xgboost lightgbm  # 非 LLM 基线
```

### 4.3 验证数据准备层

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

**期望输出**：8 passed。

### 4.4 数据集准备

**注意**：`data/raw/` 和 `data/external/` 被 `.gitignore` 排除，未提交到 git。服务器端需要自行准备。

#### Edge-IIoTset

已在本地准备好。迁移到服务器的方式二选一：

- **方式 A**：直接从本地 `data/raw/edgeiiotset/` 打包 SCP 到服务器：
  ```bash
  # 在本地执行
  tar czf edgeiiotset_raw.tar.gz -C /Users/ze/fedllm/data/raw edgeiiotset
  scp edgeiiotset_raw.tar.gz <server>:~/fedllm-sci4/data/raw/
  # 在服务器执行
  cd ~/fedllm-sci4/data/raw && tar xzf edgeiiotset_raw.tar.gz
  ```

- **方式 B**：服务器上从 Kaggle 重新下载（需 Kaggle API key）：
  ```bash
  pip install kagglehub
  python3 -c "import kagglehub; kagglehub.dataset_download('mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot')"
  ```
  下载后解压到 `data/raw/edgeiiotset/full/`。

#### SNLI

```bash
mkdir -p data/raw/snli
cd data/raw/snli
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
ln -s snli_1.0 current
cd ../../..
```

#### Fed-SB 基线代码

```bash
git clone https://github.com/CERT-Lab/fed-sb.git data/external/fed-sb
cd data/external/fed-sb
git checkout e9e649822ce63437535f72fbf06c146a22b9e527  # 已验证的 commit

# 链接 SNLI 数据到 Fed-SB 官方期望位置
mkdir -p fed_sb/DP/SNLI/data
ln -s ../../../../../../raw/snli/snli_1.0 fed_sb/DP/SNLI/data/snli_1.0
cd ../../..
```

### 4.5 重新生成数据产物（可选）

如果上面的数据集路径与默认一致，可直接运行：

```bash
PYTHONPATH=src python3 scripts/prepare_datasets.py \
  --edge-root data/raw/edgeiiotset/full_dataset \
  --snli-root data/raw/snli/current \
  --out-dir data/processed \
  --count-rows
```

产物应与 `data/processed/` 下已提交的内容一致。

---

## 5. 关键文档索引

### 必读（按顺序）

1. **`docs/superpowers/specs/2026-05-31-spectra-dp-fedcore-design.md`**  
   **这是整个项目的宪法**。1313 行，涵盖：
   - 研究问题（RQ1-RQ5）和可检验假设（H1-H5）
   - 与 Fed-SB 的碰撞边界（什么不能声称、什么可以声称）
   - 完整数学公式（谱基、核心参数化、DP 协议、层自适应噪声分配）
   - 实验矩阵（6 张表）
   - 消融设计和预期图表
   - 风险登记册（9 项风险及缓解）
   - 论文骨架、贡献措辞、审稿人 Q&A

2. **`data/README.md`**  
   数据集清单、SHA-256 校验值、下载地址、已生成产物说明。

3. **`docs/superpowers/plans/2026-05-31-data-preparation.md`**  
   数据准备层的详细实施计划（已执行完毕，可作为后续模块的参考模板）。

### 快速参考

| 你想知道什么 | 去哪里找 |
|---|---|
| 方法数学公式 | specs 第 5 节 "Method" |
| DP 协议细节 | specs 第 5.4-5.6 节 |
| 实验矩阵 | specs 第 9.1 节 "SCI Experiment Matrix" |
| 消融设计 | specs 第 11 节 "Ablations" |
| 复现性清单 | specs 第 12 节 "Reproducibility Checklist" |
| 论文贡献措辞 | specs 第 13 节 "Paper Contribution Wording" |
| 风险与缓解 | specs 第 14 节 "Risk Register" |
| 下一步做什么 | specs 第 15 节 "Immediate Next Steps" |
| 15 类标签定义 | `src/fedllm_data/labels.py` |
| 提示模板 | `src/fedllm_data/prompts.py` |
| 源拆分逻辑 | `src/fedllm_data/edgeiiot.py` |

---

## 6. 接下来要做的具体实验（龙虾的重点工作）

### 阶段 1：环境验证（第 1-2 天）

1. **跑通数据准备层测试**：`PYTHONPATH=src pytest tests/ -v` → 8 passed
2. **复现 Fed-SB SNLI 联邦隐私实验**：
   ```bash
   bash scripts/run_fedsb_snli_fed_private.sh
   ```
   目的：验证 Fed-SB 代码在服务器 GPU 上能跑通、验证隐私会计路径、记录基线性能。
3. **确认 Edge-IIoTset 数据完整性**：核对 `data/raw/edgeiiotset/full/` 下有 14 attack + 10 normal + 2 selected CSV，总约 26 个文件。

### 阶段 2：核心模块实现（第 3-10 天）

4. **公开谱基构造**（`src/spectra/` 或类似目录）：
   - 加载 `google/gemma-4-E2B-it`（4-bit 量化）
   - 对目标层（先 `q_proj`, `v_proj`）做截断 SVD
   - 保存 `U_{l,p}`、`V_{l,p}` 和谱能量 `E_l(p)`
   - 实现核心参数化 `Delta W = gamma * U * C * V^T`

5. **最小可复现 Edge-IIoT 流水线**：
   - 数据转换：把 CSV 行转为指令格式
   - 本地训练：冻结骨干 + 训练公开谱核心（非 DP，先冒烟）
   - 评估：15 类分类 Macro-F1
   - 目标：先让非 DP 版本在本地数据上能学到东西

6. **单服务器 FL 模拟**：
   - K=10 客户端
   - IID 分区（均匀随机）做 sanity check
   - Dirichlet(α=0.5) 标签偏斜做主 Non-IID 实验
   - 每轮：服务器广播全局核心 → 客户端本地训练 → 客户端上传 → 服务器平均聚合

### 阶段 3：DP 与高级模块（第 11-20 天）

7. **客户端 DP 协议**：
   - 全局裁剪 + 均匀噪声基线
   - RDP 会计（保守无采样放大版本）
   - 验证 accountant 输出与 epsilon 目标一致

8. **层自适应分配**：
   - 谱能量驱动的 `p_l` 分配
   - 层自适应裁剪半径 `R_l` 和噪声 `s_l`
   - 消融：uniform `p_l` + uniform noise → spectrum-adaptive `p_l` + uniform noise → spectrum-adaptive `p_l` + layer-adaptive noise

9. **本地残差 + 服务器后处理**：
   - 每个客户端维护一个不上传的 `P_{k,l}`
   - 服务器对聚合核心做收缩平滑

### 阶段 4：完整实验（第 21-35 天）

10. **主实验矩阵**：按 specs 第 9.1 节的 Table 1-6 逐一运行
11. **非 LLM 基线**：RF、XGBoost、FT-Transformer、MLP-FedAvg
12. **结果整理**：脚本化所有图表和表格，确保种子固定、可复现

---

## 7. 重要注意事项与常见陷阱

### 7.1 数据集相关

- **不要上传 `data/raw/` 到 git**。原始数据集共约 12GB，已用 `.gitignore` 排除。
- **Edge-IIoTset 标签标准化**：`OS_Fingerprinting` 必须映射为 `Fingerprinting`，代码中已在 `src/fedllm_data/labels.py` 处理。
- **主要评估协议**：使用 `ML-EdgeIIoT-dataset.csv` 做 80/10/10 分层行级拆分，**不是**按文件拆分。文件/源感知拆分仅用于鲁棒性检查。
- **泄露诊断**：主 CSV 中可能存在重复行（同一流量被不同传感器捕获）。需要在训练后跑重复诊断，如果泄露严重，报告去敏后的结果。

### 7.2 方法相关

- **不要声称记录级 DP**。当前设计是客户端级 DP（每个客户端上传前裁剪加噪）。如果实现了 DP-SGD（per-example clipping），才能声称记录级 DP。
- **不要声称"我们是第一个训练小方阵核心"**。Fed-SB / LoRA-SB 已经做了类似的事情。我们的独特卖点是**纯公开谱构造 + 边缘预算分配 + IIoT 评估协议**。
- **不要使用私有数据来选择 `p_l` 或 `R_l`**。所有超参数要么来自公开骨干谱，要么来自固定公开网格。如果必须用私有数据估计分位数，该估计本身必须被 DP 会计覆盖。
- **RDP 会计**：第一版用保守的无采样放大会计。采样放大（如 Poisson sampling）可以作为附录变体，但不是主设定。

### 7.3 基线相关

- **Fed-SB 是最强算法基线**，必须公平对比。对比维度三个：相同隐私预算、相同通信量、相同可训练参数。
- 如果 Fed-SB 直接适配 Gemma 4 E2B 失败，保留一个"Fed-SB-style fixed-core baseline"在我们自己的代码路径中实现，并明确标注差异。

### 7.4 论文相关

- 审稿人最可能问的问题及回答模板，见 specs 第 13.2 节。
- 措辞纪律：用"improves the trade-off"而不是"outperforms all baselines"；用"edge-deployable inference"而不是"every client trains on-device"。

---

## 8. 核心代码入口（未来新增模块的预期位置）

| 模块 | 建议目录 | 说明 |
|---|---|---|
| 谱基构造 | `src/spectra/basis.py` | SVD、截断、基保存/加载 |
| 核心适配器 | `src/spectra/adapter.py` | 核心参数化、前向传播 |
| 本地训练 | `src/spectra/trainer.py` | 单客户端训练循环 |
| FL 模拟 | `src/spectra/federation.py` | 服务器聚合、客户端采样、广播 |
| DP 协议 | `src/spectra/privacy.py` | 裁剪、加噪、RDP 会计、隐私账本 |
| 分配优化 | `src/spectra/allocation.py` | 谱能量驱动的 `p_l` 和噪声分配 |
| 评估 | `src/spectra/eval.py` | Macro-F1、混淆矩阵、每类召回 |
| 实验运行 | `scripts/run_fl.py` | 主实验入口 |
| 绘图制表 | `scripts/make_tables.py` | 论文图表生成 |

> 以上目录仅为建议，具体实现时可根据实际依赖调整。保持模块化，方便消融。

---

## 9. 沟通与问题排查

### 如果你遇到以下问题……

| 问题 | 排查步骤 |
|---|---|
| `pytest` 失败 | 确认 `PYTHONPATH=src`；确认 Python ≥ 3.10；删除 `__pycache__` 重试 |
| Edge-IIoTset 文件数不对 | 检查 `data/raw/edgeiiotset/full/` 下是否有 `Attack traffic/`、`Normal traffic/`、`Selected dataset for ML and DL/` 三个子目录 |
| Fed-SB 脚本跑不通 | 检查 `data/external/fed-sb/fed_sb/DP/SNLI/data/snli_1.0` symlink 是否正确；检查 CUDA/GPU 可用性 |
| 数据集路径不一致 | 修改 `configs/data/edgeiiot.json` 和 `configs/data/snli.json`，或传命令行参数 |
| git 提交时发现 data/raw 被追踪 | 确认 `.gitignore` 生效：`git rm -r --cached data/raw` |

### 需要决策的事项（请与作者确认）

1. **FL 引擎选择**：自研轻量级 FL 模拟（如 specs 暗示）还是引入 Flower/FedML？建议自研以保持可控。
2. **DP 会计库**：Opacus、Google DP Library、还是自研 RDP accountant？建议先自研保守版本，确保透明可控。
3. **量化方案**：是否使用 bitsandbytes 4-bit 训练？Gemma 4 E2B 较小，也可能用 8-bit 或 fp16。
4. **计算预算**：服务器 GPU 型号和数量？这决定了 batch size、层扩展范围、以及是否跑 K=20 敏感性实验。

---

## 10. 变更日志

| 日期 | 变更 | 作者 |
|---|---|---|
| 2026-05-31 | 研究设计 v0.3 定稿；数据准备层完成；数据集就位 | Junze |
| 2026-06-10 | 仓库初始化并推送到 GitHub；交接文档编写 | Junze |

---

**祝实验顺利！有任何问题请直接开 Issue 或联系作者。**
