# SPECTRA-FedCore Research Design

Date: 2026-05-31
Status: v0.3 literature-calibrated SCI-oriented research design
Working title: SPECTRA-FedCore: Public-Spectral Private Federated Core Adaptation for Edge IIoT Security

Name note: use `SPECTRA-FedCore` as the short working method name during experiments. The final paper title can add "client-side DP" or "differentially private" after results and framing are settled.

## 1. One-Sentence Positioning

SPECTRA-FedCore is an auditable, edge-budget-aware DP federated adaptation framework that turns an edge-deployable small language model into an IIoT intrusion-detection assistant by using only public backbone spectral information to define, allocate, train, privatize, and aggregate tiny core adapters.

The paper should not be framed as "we are the first to train only a small square LoRA core." Fed-SB already makes a closely related claim. The defensible framing is:

> Existing private federated LoRA methods optimize generic language tasks; existing IIoT IDS methods optimize lightweight classifiers. SPECTRA-FedCore bridges the missing middle: a public-basis, budget-aware, auditable DP-FL method for edge small language models in IIoT security.

## 1.1 Target SCI Positioning

The paper should be written as a method-plus-systems paper for privacy-preserving edge intelligence, not as a pure LLM adapter paper and not as a pure IDS classifier paper.

Best-fit journal families:

1. IEEE Internet of Things Journal / IEEE Transactions on Industrial Informatics:
   Emphasize IIoT gateways, edge deployment, privacy-preserving collaborative learning, communication budget, and intrusion detection.

2. Computers & Security / Journal of Information Security and Applications:
   Emphasize privacy auditability, threat model, IDS rigor, attack-class recall, and leakage-aware evaluation.

3. Future Generation Computer Systems / Information Fusion:
   Emphasize edge-cloud systems, federated learning, model efficiency, and multi-metric trade-offs.

The manuscript should avoid a "LLM beats all IDS models" claim. The stronger SCI story is:

> We study whether an edge-deployable small language model can be privately federated into an IIoT security assistant under strict client-side DP, and how much utility can be recovered by public spectral structure and edge-budget-aware adaptation.

## 1.2 Research Questions

RQ1: Under client-side release DP, can public-spectral core adaptation achieve a better utility-privacy-communication trade-off than FedAvg-LoRA and public Fed-SB-style baselines?

RQ2: Does public spectrum-driven rank allocation improve rare-attack recall and macro-F1 under fixed communication and trainable-parameter budgets?

RQ3: Does layer-adaptive clipping/noise reduce the performance loss caused by client-side DP compared with global clipping and uniform noise?

RQ4: Does a local residual core improve personalized IIoT gateway performance without degrading global generalization?

RQ5: Under a mainstream Edge-IIoTset closed-set protocol, how stable are the conclusions across simulated FL client partitions, duplicate diagnostics, and raw-file robustness checks?

## 1.3 Testable Hypotheses

H1: At the same `(epsilon, delta)` and communication budget, SPECTRA-FedCore is expected to improve macro-F1 over FedAvg-LoRA and to be competitive with, or better than, a Fed-SB-style fixed-core baseline under client-side release DP.

H2: Public spectrum-adaptive `p_l` is expected to improve macro-F1 per uploaded MB compared with uniform-rank core adapters.

H3: Layer-adaptive clipping/noise is expected to improve low-epsilon performance, especially at `epsilon <= 4`, by reducing over-noising of informative layers.

H4: Local residual cores are expected to improve client-personalized F1 under device-skew and attack-skew without requiring extra privacy budget because they are never uploaded.

H5: The same method ordering is expected to hold under duplicate-controlled and raw-file robustness checks, even if absolute scores differ from the primary selected-CSV protocol.

## 2. Collision Boundary

### What We Must Not Claim

Do not claim the following as unique contributions:

- Training and uploading only a small square matrix between frozen adapter factors.
- Exact aggregation through direct averaging of that small matrix.
- Linear DP noise without LoRA two-factor cross-noise as the only novelty.

Fed-SB already states that LoRA-SB learns a small square matrix between fixed adapters, direct averaging guarantees exact updates, and private settings benefit from fewer trainable parameters and avoiding LoRA noise amplification.

### 2.1 Novelty Matrix

| Dimension | FedAvg-LoRA | LoRA-XS / PiSSA | Fed-SB / LoRA-SB | DP-FL IDS classifiers | SPECTRA-FedCore |
| --- | --- | --- | --- | --- | --- |
| Small trainable adapter | yes | yes | yes | no | yes |
| Trains only square core | no | yes | yes | no | yes |
| Public-backbone-only basis | no | partial | no, update-based initialization is task-driven | not applicable | yes |
| Client-side release DP as main setting | sometimes | no | private setting exists | often | yes |
| Layer-wise public spectral rank allocation | no | no | no | no | yes |
| Layer-wise DP noise allocation for core adapters | no | no | no | no | yes |
| Local residual for IIoT non-IID gateways | no | no | limited/personalization-dependent | sometimes | yes |
| Edge-IIoT instruction IDS protocol | no | no | no | no | yes |
| Leakage-aware IDS splits | no | no | no | varies | yes |

The paper's defensible novelty is the combination of private-data-free public spectral construction, client-side DP utility recovery, and IIoT-specific evaluation. Any single component by itself is not enough for a strong SCI submission.

### What We Can Claim

The non-colliding contribution should be:

1. Private-data-free spectral basis construction:
   The adapter bases are computed only from released backbone weights, not from private task data, private client gradients, or a task-specific warm-start set.

2. Public spectrum-driven edge allocation:
   Layer-wise core dimensions and privacy noise are selected from public singular-value spectra under explicit communication, parameter, memory, and privacy budgets.

3. Auditable DP-FL protocol for edge IIoT IDS:
   The privacy unit, clipping rules, RDP accounting, client participation, data splits, and attack metrics are fixed and reproducible.

4. Federated instruction IDS benchmark:
   Edge-IIoTset is converted into a small-language-model instruction classification benchmark with a mainstream stratified closed-set split, simulated IID/non-IID FL partitions, duplicate diagnostics, raw-file robustness checks, rare-attack recall, latency, communication, and memory reporting.

## 3. Problem Story

Industrial edge gateways observe sensitive telemetry: packet features, device activity, logs, alerts, and system metrics. Uploading raw telemetry to a cloud model is often unacceptable. Traditional IDS models such as XGBoost, Random Forest, CNN, or MLP are strong for classification, but they are narrow tools: they do not naturally support local explanation, natural-language operator interaction, remediation suggestions, or policy generation.

Small language models can provide that unified security assistant interface, but edge-side private federated adaptation has four pressure points:

- Memory: even small LLM fine-tuning is heavier than inference.
- Communication: gateways cannot upload large LoRA factors every round.
- Non-IID data: each gateway sees different devices, protocols, benign baselines, and attack classes.
- Privacy auditability: "federated" alone does not provide a formal privacy guarantee.

SPECTRA-FedCore answers this by constraining adaptation to public spectral subspaces of the released backbone, training only tiny per-layer cores, and releasing clipped/noised core updates with explicit privacy accounting.

### 3.1 Introduction Narrative Arc

The introduction should move through four steps:

1. Edge IIoT security needs collaborative learning because individual gateways see incomplete attack distributions.
2. Raw telemetry and local traffic logs are sensitive, so FL is attractive but not sufficient without auditable DP.
3. Small language models are useful security assistants, but generic private federated LoRA methods are not designed around edge budgets, public-only initialization, or IIoT leakage risks.
4. SPECTRA-FedCore uses public backbone spectra to define the adaptation subspace and to allocate rank/noise budgets, then evaluates the full privacy-utility-communication story on a realistic IIoT benchmark.

## 4. Backbone and Deployment Assumption

Primary backbone:

```text
google/gemma-4-E2B-it
```

Use a 4-bit quantized frozen backbone for training experiments where supported, with trainable public-spectral core adapters. Deployment is framed as edge inference with merged or separately loaded adapters.

Important wording:

- Claim: Gemma 4 E2B is suitable for edge/on-device inference evaluation.
- Do not claim that full fine-tuning, or even PEFT training, is guaranteed to fit on the most constrained edge devices.
- Use the phrase "edge or near-edge gateway adaptation" for training, and "edge deployment" for inference.

## 5. Method

### 5.1 Public Spectral Basis

For each target linear layer:

```text
W0_l in R^{d_out x d_in}
```

compute a truncated SVD from the public released backbone:

```text
W0_l = U_l Sigma_l V_l^T
```

Select the first `p_l` singular vectors:

```text
U_{l,p} in R^{d_out x p_l}
V_{l,p} in R^{d_in x p_l}
```

These bases are public, fixed, not trained, not uploaded by clients, and reproducible from the same backbone checkpoint.

The shared adapter update is:

```text
Delta W_{l,k} = gamma_l U_{l,p} C_{l,k} V_{l,p}^T
C_{l,k} in R^{p_l x p_l}
```

### 5.2 Global Core Plus Local Residual

Each client uses two core paths:

```text
G_l^{(t)}: global shared core broadcast by the server
D_{k,l}^{(t)}: client global update path, uploaded after DP release
P_{k,l}: local personalized residual core, never uploaded
```

At the beginning of each communication round, `D_{k,l}^{(t)}` is initialized to zero or to a server-specified update buffer, then trained locally as the uploadable global-path delta. `P_{k,l}` persists across rounds on that client and is never included in the upload vector.

Local forward adapter:

```text
Delta W_{l,k}^{(t)}
  = gamma_l U_{l,p}
    (G_l^{(t)} + D_{k,l}^{(t)} + lambda_p P_{k,l})
    V_{l,p}^T
```

The personalized residual is included to handle IIoT non-IID behavior: device baselines, protocol mixes, and local attack exposure differ by gateway. The residual must be ablated because it can improve local accuracy while weakening global generalization if overused.

### 5.3 Layer-Wise Core Size Allocation

For each target layer, define public spectral energy:

```text
E_l(p) = sum_{i=1}^p sigma_{l,i}^2 / sum_i sigma_{l,i}^2
```

Choose `p_l` by solving a small bounded allocation problem:

```text
maximize   sum_l a_l E_l(p_l)
subject to sum_l p_l^2 <= B_param
           sum_l b p_l^2 <= B_comm
           p_l in P_candidates
```

where:

- `B_param` is the trainable parameter budget.
- `B_comm` is the uploaded scalar budget per round.
- `b` is bits per scalar after serialization.
- `a_l` can be uniform in the first version, then replaced by public layer heuristics or a DP-safe validation estimate.

This is a key difference from Fed-SB: the allocation is public-spectrum and edge-budget driven, rather than generic fixed-rank adapter training.

Default SCI version:

- Candidate ranks: `p_l in {4, 8, 16, 32}`.
- First-pass target modules: `q_proj` and `v_proj`.
- Expanded target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Full ablation target modules: attention projections plus selected MLP projections if compute allows.
- Initial layer weights: `a_l = 1`.
- Main public-spectrum weights: `a_l = E_l(p_l)` or normalized spectral tail gain. This remains private-data-free.

The choice between `E_l(p_l)` and `E_l(p_l) / p_l^2` should be treated as an ablation:

- `E_l(p_l)` favors high-energy layers and may maximize accuracy.
- `E_l(p_l) / p_l^2` favors parameter efficiency and may improve uploaded-MB Pareto curves.

For the first SCI submission, report both in an allocation ablation and use whichever wins on validation under a predeclared, seed-averaged criterion.

### 5.4 Differential Privacy Protocol

Use client-level DP as the main paper guarantee because it is rigorous and practical for FL.

Record-level DP is possible only if the local training procedure itself uses DP-SGD with per-example clipping and noise. Do not claim record-level DP from only noising the final client update.

Main trust model:

- Client-side release DP: each client clips and noises its upload before sending it to the server. This protects against an honest-but-curious server observing individual uploads.
- Secure-aggregation central DP is not the main setting. It can be mentioned as a future or appendix variant, but the paper's primary goal is to reduce utility loss under the stronger client-side release setting.

Baseline global-clipping protocol:

1. Client trains local global-path cores `D_{k,l}^{(t)}` and local residuals `P_{k,l}`.
2. Client concatenates the uploadable global-path core updates into a vector:

```text
u_k^{(t)} = concat_l vec(D_{k,l}^{(t)})
```

3. Client clips:

```text
u_k^{c,t} = u_k^{(t)} min(1, R_t / ||u_k^{(t)}||_2)
```

4. Client releases:

```text
tilde u_k^{(t)} = u_k^{c,t} + z_k^{(t)}
z_k^{(t)} ~ N(0, s_t^2 I)
```

For replace-one client adjacency, sensitivity is `Delta_t = 2 R_t`. The Gaussian mechanism RDP at order `alpha` is:

```text
epsilon_t(alpha) = alpha Delta_t^2 / (2 s_t^2)
                 = 2 alpha R_t^2 / s_t^2
```

For add/remove client adjacency, use `Delta_t = R_t`, giving:

```text
epsilon_t(alpha) = alpha R_t^2 / (2 s_t^2)
```

Total client-level RDP for a participating client is:

```text
epsilon_total(alpha) = sum_{t in T_k} epsilon_t(alpha)
```

Convert to `(epsilon, delta)`:

```text
epsilon(delta) = min_{alpha > 1}
  epsilon_total(alpha) + log(1 / delta) / (alpha - 1)
```

Sampling amplification can be added later, but the first reproducible version should report conservative no-amplification accounting and optionally an amplified accountant in an appendix.

### 5.5 Utility Preservation Under Client-Side Release DP

Because every participating client uploads an already noised update, utility preservation is a first-class method goal rather than a minor implementation detail. SPECTRA-FedCore uses five DP-compatible mechanisms:

1. Dimension reduction before privacy:
   Release only public-spectral core updates instead of LoRA factors or dense updates. This reduces the number of noised coordinates before any DP accounting.

2. Public spectrum-driven layer allocation:
   Assign larger cores to layers whose public backbone spectra indicate higher retained energy under a fixed edge communication budget.

3. Layer-adaptive clipping and noise:
   Avoid drowning small but useful layers by assigning per-layer clipping radii and noise scales under a composed RDP budget.

4. Personalized local residual:
   Keep client-specific residual cores local and unuploaded. This recovers non-IID local utility without spending communication or privacy budget on purely local patterns.

5. Server-side denoising as post-processing:
   Apply shrinkage or moving-average smoothing to noised aggregate core updates using only public information and prior DP transcripts. This consumes no additional privacy budget, though it must be ablated to verify that it improves utility.

The paper should present these as a coordinated answer to the stronger client-side release setting. The main research question becomes:

> How much of the utility lost to client-side DP release can be recovered by public spectral structure, budget-aware allocation, personalization, and post-processing?

### 5.6 Layer-Adaptive Privacy Allocation

The full method uses per-layer clipping and layer-adaptive noise by default. Global clipping with uniform noise is retained as a baseline and implementation sanity check.

Per-layer release:

```text
D_{k,l}^{c,t}
  = D_{k,l}^{(t)}
    min(1, R_{l,t} / ||D_{k,l}^{(t)}||_F)

tilde D_{k,l}^{(t)}
  = D_{k,l}^{c,t} + Z_{k,l}^{(t)}

Z_{k,l}^{(t)} ~ N(0, s_{l,t}^2 I_{p_l x p_l})
```

If releases are clipped and noised per layer, with per-layer sensitivity `Delta_l`, the RDP composition is:

```text
sum_l alpha Delta_l^2 / (2 s_l^2) <= rho_alpha
```

Let `d_l = p_l^2` and minimize weighted expected perturbation:

```text
minimize sum_l w_l d_l s_l^2
subject to sum_l alpha Delta_l^2 / (2 s_l^2) <= rho_alpha
```

The closed-form variance allocation is:

```text
s_l^2 =
  sqrt((alpha Delta_l^2 / 2) / (w_l d_l))
  * (sum_j sqrt((alpha Delta_j^2 / 2) w_j d_j))
  / rho_alpha
```

This should be introduced after the global-clipping version is working. The main ablation should compare:

- uniform `p_l`, uniform noise
- spectrum-adaptive `p_l`, uniform noise
- spectrum-adaptive `p_l`, layer-adaptive noise

Default clipping/noise policy for the full method:

- Use per-layer clipping radii `R_l = c_R sqrt(d_l)` for the first reproducible version.
- Select `c_R` from a small public grid using a non-private public proxy only if available; otherwise use the same `c_R` for all compared methods and report clipping rates.
- Do not tune `R_l` using raw private client update quantiles unless the quantile estimation is itself DP-accounted.
- Use `w_l = E_l(p_l)` for the first full method and report `w_l = E_l(p_l) / d_l` as an efficiency-oriented ablation.

The paper should report clipping rate by layer and round. If a layer is clipped almost always or almost never, that is evidence that `R_l` needs adjustment or that the adaptive mechanism is not meaningful.

### 5.7 Server Aggregation

The server receives noised core updates and aggregates:

```text
bar u^{(t)} = sum_{k in I_t} q_{k,t} tilde u_k^{(t)}
```

Then unpack to each layer:

```text
G_l^{(t+1)} = G_l^{(t)} + bar D_l^{(t)}
```

Optional post-processing shrinkage:

```text
hat D_l = omega_l bar D_l
omega_l in [0, 1]
```

This is privacy-cost-free post-processing if `omega_l` is derived from public transcript statistics or from DP-safe statistics. The first version should include shrinkage as a utility-preservation component, but it must be separately ablated.

## 6. Theory Claims

### Claim 1: Unique Coordinates in a Public Subspace

For:

```text
M_l = { U_{l,p} C V_{l,p}^T : C in R^{p_l x p_l} }
```

if `U_{l,p}` and `V_{l,p}` have orthonormal columns, then every `Delta W in M_l` has unique coordinates:

```text
C = U_{l,p}^T Delta W V_{l,p}
```

This avoids LoRA factor gauge ambiguity inside the chosen public subspace.

### Claim 2: Core Aggregation Equals Product-Space Aggregation

```text
sum_k q_k U C_k V^T = U (sum_k q_k C_k) V^T
```

This is exact, but not unique to this paper. The paper should state it as a property needed for correctness, then contrast the source of the basis and allocation with Fed-SB.

### Claim 3: Frobenius Norm Preservation

Because `U` and `V` have orthonormal columns:

```text
|| U (C - Chat) V^T ||_F = || C - Chat ||_F
```

This lets the paper analyze perturbation, clipping, and truncation in core space.

### Claim 4: Linear DP Perturbation

The noised update is:

```text
U (C + Z) V^T - U C V^T = U Z V^T
```

and:

```text
E || U Z V^T ||_F^2 = E || Z ||_F^2 = p_l^2 s_l^2
```

Again, the paper should avoid claiming this linearity alone is new; it is part of the correctness and utility analysis.

## 7. Dataset and Task Protocol

Dataset:

```text
Edge-IIoTset
```

Main tasks:

1. Binary IDS: benign vs malicious.
2. Threat-level classification: 5 threat families.
3. Fine-grained attack classification: benign plus 14 attacks.

Primary task should be fine-grained attack classification because binary IDS is often too easy and does not show the value of federated knowledge sharing or rare-class handling.

Instruction format:

```text
System:
You are an edge IIoT security assistant running on a local gateway.
Classify local telemetry without sending raw data to the cloud.

User:
Device and traffic telemetry:
[network]
proto={proto}; service={service}; duration={duration_bin};
src_bytes={src_bytes_bin}; dst_bytes={dst_bytes_bin};
tcp_flags={tcp_flags}; packet_rate={packet_rate_bin};

[system]
cpu={cpu_bin}; memory={memory_bin}; process_count={proc_bin};

[log]
alert_count={alert_count_bin}; error_count={error_count_bin};

Allowed labels:
{label_list}

Assistant:
label =
```

Training should supervise only the label answer. Explanations can be evaluated later, but the main paper metrics must be classification metrics.

Feature handling:

- Main prompt uses a compact curated schema of roughly 20-35 protocol and traffic-behavior fields.
- Exclude identifier-like or high-leakage fields from the main prompt: `frame.time`, `ip.src_host`, `ip.dst_host`, payload/message fields, URI/full-URI fields, raw referer fields, and label columns.
- Numeric features should be rendered consistently; training-set-only quantile binning can be used if it improves token stability, with bin boundaries saved to disk.
- Keep prompt length at or below 512 tokens unless ablated.
- Full-feature and shorter-feature prompts are ablations, not the main setting.

## 8. Federated Splits

### 8.0 Verified Metadata Status

Verification on 2026-05-31:

- The Edge-IIoTset paper lists the selected feature columns as packet/protocol features plus `Attack_label` and `Attack_type`; it does not list an explicit `session_id`, `scenario_id`, or `device_id` field in the selected 63-column table.
- A publicly mirrored `ML-EdgeIIoT-dataset.csv` header confirms columns such as `frame.time`, `ip.src_host`, `ip.dst_host`, protocol fields, `Attack_label`, and `Attack_type`, but no explicit scenario/session identifier.
- A public preprocessing note reports that the full dataset package contains separate raw files under `Normal traffic/<sensor_name>/...` and `Attack traffic/<attack_name>...`, plus selected merged `DNN-EdgeIIoT-dataset.csv` and `ML-EdgeIIoT-dataset.csv` files.

Primary evaluation protocol:

1. Main closed-set benchmark:
   Use `ML-EdgeIIoT-dataset.csv` with an 80/10/10 stratified row-level split over the normalized 15-class label space. This follows the common Edge-IIoTset IDS practice and keeps the experimental protocol familiar to reviewers.

2. Main FL simulation:
   Use the same train/validation/test split, then partition only the training set into simulated clients on a single server. The validation and test sets remain global and fixed across methods so method differences are not confounded by different evaluation data.

3. Client partitions:
   Run one IID sanity partition and one primary non-IID label-skew partition. If compute allows, add a source/protocol-skew partition as a secondary robustness check.

4. Leakage diagnostics:
   Report exact duplicate or feature-hash overlap across train/validation/test after removing excluded columns. If duplicate leakage is material, report a deduplicated sensitivity result.

5. Raw-file robustness:
   Use the full raw CSV structure for a per-file 8/1/1 row-level split as a robustness check. This is not an unseen-attack or file-heldout task; it is a traditional within-file split using the original file organization.

Not the main protocol:

- Entire-file heldout, unseen-attack classification, and time-aware/client-aware partitioning are not main experiments because they would make the task less comparable to common Edge-IIoTset IDS papers. They can be mentioned as future stricter OOD settings if results or space permit.

Default single-server FL simulation:

```text
K = 10 clients
participation = 100% for initial sanity, then 20%-50% for FL sweeps
rounds = {20, 50} depending on compute
local epochs = 1
seeds = 3
primary non-IID = Dirichlet label-skew with alpha = 0.5
secondary sensitivity = K = 20 clients if server time allows
```

## 9. Baselines

### Public Baseline Strategy

The strongest public algorithmic baseline should be Fed-SB, using the authors' released implementation when possible. Fed-SB is the closest collision-risk method because it trains a small square matrix between fixed adapter factors, supports federated private fine-tuning, and provides scripts for private federated experiments. Its public repository is `https://github.com/CERT-Lab/fed-sb`.

Verification on 2026-05-31:

- The Fed-SB README documents privacy-preserving fine-tuning on SNLI with `DP/SNLI/scripts/central_private.sh` and `DP/SNLI/scripts/fed_private.sh`.
- The repository tree contains `fed_sb/DP/SNLI/fed_trainer.py`, `fed_sb/DP/SNLI/trainer.py`, config files, and the private federated shell scripts.

Reproduction strategy:

1. Reproduce Fed-SB's official private federated SNLI experiment first, without changing the method. This verifies the environment, privacy path, and expected outputs.
2. Record the exact commit hash, dependencies, command line, GPU type, and privacy parameters.
3. Adapt Fed-SB to the Edge-IIoT instruction dataset with minimal changes: data loader, prompt formatter, label parsing, and metrics.
4. Run `Fed-SB-original` using its recommended settings where compatible.
5. Run `Fed-SB-budget-matched` under our communication/trainable-parameter/privacy budgets.
6. If direct adaptation to Gemma 4 E2B fails, implement a Fed-SB-style baseline in our code path and clearly mark the difference.

The comparison should not only match LoRA rank. Rank can be misleading because different methods have different trainable parameters, communication payloads, initialization costs, and privacy mechanisms. Use three fair comparison regimes:

1. Same privacy budget:
   Compare all DP methods at the same `(epsilon, delta)`, same client participation schedule, same number of rounds, and same accountant convention.

2. Same communication budget:
   Match uploaded scalars or uploaded MB per client per round. If Fed-SB uses rank `r`, choose SPECTRA layer budgets so total uploaded core scalars are comparable.

3. Same trainable-parameter budget:
   Match the total number of trainable adapter parameters. This separates "better representation" from "more parameters".

Fed-SB should be run in two forms if compute allows:

- Fed-SB-original: faithful public implementation with its update-based initialization and recommended settings, adapted only as needed to the same backbone and Edge-IIoT prompts.
- Fed-SB-budget-matched: adjusted to match SPECTRA-FedCore's trainable parameter and communication budgets.

This avoids handicapping the baseline while still showing whether the public-spectral, private-data-free design is competitive under edge and privacy budgets.

### 9.1 SCI Experiment Matrix

The experiment plan should be staged so that every table answers one reviewer question.

Table 1: Main utility under realistic FL.

```text
Dataset split: ML-EdgeIIoT 80/10/10 stratified row-level split
FL setting: single-server simulation, K=10 clients, IID sanity and label-skew non-IID
Task: 15-class fine-grained IDS
Privacy: epsilon in {8, 4, 2, 1}, delta = 1e-5 or 1/N_clients^1.1
Methods: FedAvg-LoRA, Fed-SB-original, Fed-SB-budget-matched, SPECTRA-LDP-Core, full SPECTRA-FedCore
Metrics: Macro-F1, balanced accuracy, rare attack recall, uploaded MB/round
```

Table 2: Fairness-controlled comparison.

```text
Regime A: same epsilon, same rounds, same participation
Regime B: same uploaded MB/client/round
Regime C: same trainable adapter parameters
Report: Macro-F1, rare recall, peak VRAM, wall-clock time
```

Table 3: Non-LLM IDS context.

```text
Methods: Random Forest, XGBoost/LightGBM, FT-Transformer, MLP-FedAvg, SPECTRA-FedCore
Purpose: show that classical models remain strong on pure tabular classification, while SPECTRA targets private federated edge assistant trade-offs
```

Table 4: Ablation of utility recovery under client-side DP.

```text
SPECTRA-Core no DP
SPECTRA-LDP global clipping + uniform noise
+ public spectrum rank allocation
+ layer-adaptive clipping/noise
+ local residual
+ shrinkage post-processing
```

Table 5: Reproducibility and leakage diagnostics.

```text
Main stratified split duplicate diagnostics
Deduplicated sensitivity if feature-hash overlap is material
Raw full-package per-file 8/1/1 robustness split
K=20 client sensitivity if compute allows
Report: duplicate rate, performance shift, and whether method ranking changes
```

Table 6: Edge deployment cost.

```text
Trainable parameters
Uploaded MB/client/round
Total transmitted MB
Peak VRAM during adapter training
Inference latency with adapter
Prompt length distribution
```

Minimum publishable package:

- Main utility table with Fed-SB and FedAvg-LoRA.
- Ablation table showing why the full method recovers client-side DP utility.
- Non-LLM context table.
- Duplicate diagnostics and raw-file robustness table.
- Reproducibility package with fixed seeds, split indices, configs, and privacy ledgers.

### Non-LLM IDS Baselines

- Random Forest
- XGBoost or LightGBM
- FT-Transformer or TabTransformer
- MLP-FedAvg with and without DP

These are mandatory because Edge-IIoTset is tabular/network-flow data.

### LLM and Adapter Baselines

Use the same Gemma 4 E2B instruction backbone where possible:

- Zero-shot or prompt-only Gemma classifier
- Centralized LoRA
- Local-only public-spectral core adapter
- FedAvg-LoRA
- FFA-LoRA
- Fed-SB or a faithful reimplementation if compute permits
- FedASK if implementation budget allows
- AS-LoRA if implementation budget allows
- SPECTRA-FedCore without DP
- SPECTRA-FedCore without adaptive allocation
- Full SPECTRA-FedCore

If Fed-SB cannot be fully reproduced, include a clearly labeled "Fed-SB-style fixed-core baseline" and cite the limitation.

## 10. Main Metrics

Utility:

- Macro-F1
- Balanced accuracy
- Per-class recall
- Rare attack recall
- AUROC for binary and one-vs-rest settings
- Confusion matrix

Privacy and security:

- Report `(epsilon, delta)` with RDP order grid.
- Membership inference attack AUC on released global models if feasible.
- Clipping rate per round.
- Noise multiplier per round.

Efficiency:

- Trainable parameters
- Uploaded MB per client per round
- Total communication
- Peak VRAM
- Wall-clock training time
- Edge inference latency
- Prompt token length

Robustness:

- Non-IID severity sweep.
- Client participation sweep: 100%, 20%, 10%.
- Rare-class imbalance sweep if feasible.

Statistical reporting:

- Run at least 3 seeds for all main tables.
- Report mean plus standard deviation.
- Use paired tests or bootstrap confidence intervals for the main comparison against Fed-SB-budget-matched.
- For rare attacks, report per-class recall rather than only aggregate macro-F1.
- Always report the exact privacy accountant variant and RDP order grid.

## 11. Ablations

Core ablations:

- Random orthogonal basis vs public SVD basis vs PiSSA-style basis.
- Uniform `p_l` vs public spectrum-adaptive `p_l`.
- Uniform noise vs layer-adaptive noise.
- No local residual vs local residual.
- No DP vs epsilon in `{8, 4, 2, 1}`.
- Global clipping vs per-layer clipping.
- With and without post-processing shrinkage.
- Main stratified selected-CSV split vs deduplicated sensitivity vs raw per-file 8/1/1 robustness split.

The most important plots:

- Macro-F1 vs epsilon.
- Macro-F1 vs uploaded MB per round.
- Rare attack recall heatmap.
- Per-layer selected `p_l`.
- Clipping rate and noise multiplier over rounds.
- Communication/VRAM/latency Pareto frontier.

## 12. Reproducibility Checklist

The repository should eventually include:

- Exact dataset download instructions and checksum.
- A frozen label map.
- Saved train/validation/test indices for every split and seed.
- Saved numeric bin boundaries.
- Prompt renderer versioned as code.
- Model checkpoint ID and revision.
- Adapter target module list.
- Hyperparameter YAML files for every method.
- DP accountant implementation and exported privacy ledger.
- Serialized per-round metrics.
- Environment file with CUDA, PyTorch, Transformers, PEFT, bitsandbytes, and Python versions.
- Scripts for all tables and plots.

Artifact expectations for SCI review:

- `configs/methods/fedsb_original.yaml`
- `configs/methods/fedsb_budget_matched.yaml`
- `configs/methods/spectra_ldp_global.yaml`
- `configs/methods/spectra_dp_full.yaml`
- `configs/splits/device_skew_seed*.json`
- `configs/splits/attack_skew_seed*.json`
- `outputs/<run_id>/privacy_ledger.json`
- `outputs/<run_id>/communication.json`
- `outputs/<run_id>/metrics_by_class.json`
- `outputs/<run_id>/hardware_profile.json`

Minimum run command structure:

```text
python scripts/prepare_edgeiiot.py --config configs/data/edgeiiot.yaml
python scripts/run_fl.py --config configs/experiments/spectra_fedcore.yaml
python scripts/evaluate.py --run-dir outputs/<run_id>
python scripts/make_tables.py --runs outputs/
```

## 13. Paper Contribution Wording

Recommended contribution text:

1. We propose SPECTRA-FedCore, a public-spectral federated core adaptation method for edge small language models. It constructs all adapter bases from released backbone weights, so basis construction does not depend on private client samples or gradients.

2. We introduce an edge-budget-aware allocation scheme that selects layer-wise core sizes and privacy noise scales from public singular-value spectra under communication, parameter, and RDP privacy constraints.

3. We provide an auditable client-level DP federated training protocol for core adapters, with explicit clipping, Gaussian release, RDP accounting, participation tracking, and reproducible privacy ledgers.

4. We build a federated instruction intrusion-detection benchmark on Edge-IIoTset, using a mainstream stratified closed-set split, single-server simulated FL clients, duplicate diagnostics, raw-file robustness checks, rare attack recall, and edge deployment metrics.

## 13.1 Manuscript Skeleton

Recommended title:

```text
SPECTRA-FedCore: Public-Spectral Client-Side Differentially Private Federated Adaptation of Edge Small Language Models for IIoT Intrusion Detection
```

Shorter title option:

```text
SPECTRA-FedCore for Private Edge-IIoT Security Assistants
```

Abstract structure:

1. Problem: IIoT gateways need collaborative intrusion detection, but raw telemetry cannot be shared and client-side DP damages utility.
2. Gap: Existing private federated LoRA methods are not designed around public-only basis construction, edge budget allocation, and leakage-aware IIoT evaluation.
3. Method: Public spectral core adapters, layer-wise rank/noise allocation, local residual personalization, and post-processing denoising.
4. Evaluation: Edge-IIoTset instruction IDS under a mainstream stratified split, simulated IID/non-IID FL clients on one server, client-side DP, and public Fed-SB baselines.
5. Claim: Better privacy-utility-communication trade-off, not absolute dominance over all tabular IDS classifiers.

Suggested section outline:

```text
1. Introduction
2. Related Work
   2.1 IIoT intrusion detection and Edge-IIoTset
   2.2 Federated and differentially private IDS
   2.3 PEFT and federated LoRA for LLMs
   2.4 Public spectral adapters and Fed-SB boundary
3. Problem Formulation and Threat Model
4. SPECTRA-FedCore
   4.1 Public spectral core parameterization
   4.2 Edge-budget-aware rank allocation
   4.3 Client-side DP release and RDP accounting
   4.4 Layer-adaptive noise allocation
   4.5 Local residual personalization and post-processing
5. Theoretical Analysis
6. Experimental Protocol
7. Results
8. Discussion and Limitations
9. Conclusion
```

Reviewer-facing claim discipline:

- Use "improves the trade-off" instead of "outperforms all baselines" unless the results support the stronger claim.
- Use "edge-deployable inference" and "edge/near-edge adaptation" instead of implying every client can train Gemma 4 E2B on-device.
- Use "client-level DP" consistently unless a DP-SGD record-level variant is implemented.
- Say "public-spectral construction" rather than "novel square core adapter" as the novelty anchor.

## 13.2 Expected Reviewer Questions and Answers

Q1: Is this just Fed-SB with a different basis?

A: Fed-SB is the closest baseline and must be evaluated directly. The difference is that SPECTRA constructs bases and rank/noise allocation solely from public backbone spectra, while Fed-SB relies on update-based initialization and generic language-task settings. The paper's novelty is the private-data-free public spectral design plus client-side DP utility recovery in Edge-IIoT IDS.

Q2: Why use an LLM for tabular IDS when XGBoost is strong?

A: The claim is not pure tabular accuracy dominance. The LLM provides a unified edge security assistant interface for classification, explanation, and future policy interaction. Classical IDS baselines are included to calibrate classification performance.

Q3: Does noising final client updates provide record-level DP?

A: No. The main guarantee is client-level DP for uploaded updates. Record-level DP is only claimed if local DP-SGD is implemented and accounted for.

Q4: Does public SVD spectrum really indicate downstream layer importance?

A: This is an empirical hypothesis, not an assumption. The paper tests random basis, uniform rank, SVD/PiSSA basis, and spectrum-adaptive allocation under fixed budgets.

Q5: Is the Edge-IIoT evaluation leaking duplicates?

A: The primary protocol intentionally follows common Edge-IIoTset IDS practice: a stratified row-level closed-set split on the selected ML CSV. To avoid overstating the result, the paper removes identifier-like fields from prompts, fixes all split seeds, reports exact duplicate or feature-hash overlap across splits, and includes a raw full-package per-file 8/1/1 robustness check.

Q6: Is the baseline comparison fair?

A: The paper reports same-privacy, same-communication, and same-trainable-parameter regimes, including Fed-SB-original and Fed-SB-budget-matched where feasible.

## 14. Risk Register

Risk 1: Fed-SB overlap.
Mitigation: Include Fed-SB as a baseline and frame our novelty around public spectral construction, edge-budget allocation, and IIoT protocol.

Risk 2: LLM underperforms XGBoost.
Mitigation: Do not claim absolute IDS accuracy dominance. Claim privacy-communication-memory trade-off and security-assistant extensibility.

Risk 3: Record-level DP proof is invalid.
Mitigation: Main paper uses client-level DP. Record-level DP is only claimed if implemented with DP-SGD.

Risk 4: Dataset leakage through random split.
Mitigation: Use the mainstream stratified selected-CSV split as the main comparable benchmark, but remove identifier-like fields, report duplicate diagnostics, and include raw per-file 8/1/1 robustness. Do not claim unseen-source or unseen-attack generalization from the main split.

Risk 5: Compute budget too high.
Mitigation: Start with target modules `q_proj` and `v_proj`, p in `{4, 8, 16}`, K=10 or K=20. Expand after first sanity results.

Risk 6: Fed-SB-original is hard to adapt to Gemma 4 E2B or Edge-IIoT prompts.
Mitigation: Keep two baselines: faithful Fed-SB where possible, and Fed-SB-style budget-matched core baseline implemented in the same code path as SPECTRA for controlled comparison.

Risk 7: Client-side DP destroys low-epsilon performance.
Mitigation: Make this part of the research question. Report graceful degradation, not only best-case accuracy. Highlight which utility-preservation component recovers the most performance.

Risk 8: Public spectrum allocation gives weak gains.
Mitigation: Keep it as one component in the full utility-recovery stack. If gains are small, emphasize that private-data-free allocation is competitive with stronger task-dependent baselines under stricter privacy assumptions.

Risk 9: Edge-IIoTset selected CSV metadata is insufficient for true scenario split.
Mitigation: Do not make true scenario-aware claims from the selected CSV. Treat stricter source/time/client-aware splits as future OOD work unless reliable metadata is available. Use the full raw package only for within-file robustness in the main paper.

## 15. Immediate Next Steps

1. Freeze the main data protocol: selected `ML-EdgeIIoT-dataset.csv`, normalized 15 labels, 80/10/10 stratified split, fixed seeds, and saved split indices.
2. Implement compact prompt rendering with identifier-like fields excluded and prompt length recorded.
3. Implement single-server FL simulation: `K=10` clients, IID sanity, Dirichlet label-skew primary non-IID, and optional `K=20` sensitivity.
4. Run Fed-SB official SNLI private FL reproduction from the public checkout before adapting it to Edge-IIoT.
5. Build the smallest reproducible Edge-IIoT pipeline:
   data conversion, prompt rendering, one local public-spectral core adapter, one simulated FL round, and one accountant output.
6. Run a first non-DP Edge-IIoT sanity test before spending compute on DP sweeps.
7. Add the client-side DP utility-preservation stack incrementally: dimension reduction, adaptive allocation, residual personalization, then shrinkage.

## 16. Sources Checked

- Gemma 4 model overview: https://ai.google.dev/gemma/docs/core
- Gemma 4 E2B model card: https://huggingface.co/google/gemma-4-E2B
- LoRA-XS: https://arxiv.org/abs/2405.17604
- PiSSA: https://arxiv.org/abs/2404.02948
- LoRA original ICLR paper: https://openreview.net/pdf?id=nZeVKeeFYf9
- LoRA-SB: https://arxiv.org/abs/2411.19557
- Fed-SB: https://arxiv.org/abs/2502.15436
- Fed-SB OpenReview PDF: https://openreview.net/pdf/6cafce957866c5389892ff0269c45952cf7d1df4.pdf
- FedAvg original AISTATS paper page: https://research.google.com/pubs/pub44822.html
- FedProx MLSys paper: https://proceedings.mlsys.org/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf
- FedALT AAAI paper page: https://ojs.aaai.org/index.php/AAAI/article/view/39054
- LoRA-FAIR ICCV paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Bian_LoRA-FAIR_Federated_LoRA-Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.pdf
- DP-DyLoRA paper page: https://huggingface.co/papers/2405.06368
- FedASK: https://arxiv.org/abs/2507.09990
- AS-LoRA: https://arxiv.org/abs/2605.05769
- Edge-IIoTset paper metadata: https://napier-repository.worktribe.com/output/2969370/edge-iiotset-a-new-comprehensive-realistic-cyber-security-dataset-of-iot-and-iiot-applications-for-centralized-and-federated-learning
- Edge-IIoTset paper PDF feature list: https://d197for5662m48.cloudfront.net/documents/publicationstatus/163328/preprint_pdf/ac0f306e8bd32b5626f494d51ec34daf.pdf
- Edge-IIoTset FL paper: https://www.mdpi.com/2103016
- Hybrid LLM IDS paper on Edge-IIoTset and ToN_IoT: https://pmc.ncbi.nlm.nih.gov/articles/PMC12944543/
- Public Edge-IIoTset preprocessing/file-list note: https://github.com/nickjeffrey/ensemble_learning/blob/main/Edge-IIoTset2023_dataset_preprocessing.md
- Public mirrored ML CSV header used for metadata check: https://github.com/Rayan-Ali1083/Layer7Defend/blob/main/ML-EdgeIIoT-dataset.csv
- Fed-SB public code: https://github.com/CERT-Lab/fed-sb

## 17. Local Dataset Status

Prepared on 2026-05-31:

- Full Edge-IIoTset Kaggle archive downloaded and verified as a valid zip.
- Edge-IIoTset CSV/PDF/README files extracted to `data/raw/edgeiiotset/full/`.
- PCAP files were not extracted; they remain available inside the archive.
- The extracted CSV structure includes `Attack traffic`, `Normal traffic`, and `Selected dataset for ML and DL`, enabling file/source-aware split design.
- SNLI 1.0 downloaded and extracted to `data/raw/snli/snli_1.0/` for Fed-SB private FL reproduction.
- Dataset inventory and checksums are recorded in `data/README.md`.
- Reproducible data-preparation code now exists under `src/fedllm_data/` with tests under `tests/`.
- Processed artifacts were generated under `data/processed/`, including Edge-IIoT file manifest, source-aware split plan, normalized label inventory, prompt smoke samples, and SNLI manifest.
- Edge-IIoT label normalization currently maps `OS_Fingerprinting` to `Fingerprinting` to align raw source files with the selected ML/DNN CSV label space.
- Fed-SB public code was cloned under `data/external/fed-sb` at commit `e9e649822ce63437535f72fbf06c146a22b9e527`, with SNLI linked into the official `DP/SNLI/data` location for server-side reproduction.

## 18. Literature-Calibrated v0.3 Refinement

This section records the refined paper design after checking recent Edge-IIoTset IDS papers, FL foundations, DP-FL practice, and federated LoRA/PEFT papers. The guiding decision is:

> Keep the method innovative; keep the experimental protocol familiar, reproducible, and defensible.

### 18.1 What We Borrow From Peer Papers

Edge-IIoTset IDS literature:

- The Edge-IIoTset original paper evaluates centralized ML and FedAvg-style FL after dataset processing, feature selection, stratified cross-validation, and held-out test evaluation. This supports a conventional closed-set IDS benchmark rather than an exotic unseen-file task.
- Recent Edge-IIoTset IDS papers commonly use stratified row-level train/test splits, often 80/20 or train/validation/test variants, with all preprocessing fitted only on training data.
- Recent "leakage-safe" IDS writing emphasizes: split first, fit scalers/PCA/feature selection/oversampling only on training data, keep the test set untouched, and discuss duplicate or capture-context leakage.

FL and DP-FL foundations:

- FedAvg remains the default FL baseline and motivates reporting local epochs, communication rounds, client count, and participation.
- FedProx-style literature makes non-IID client heterogeneity a required experiment dimension, but does not require us to implement every heterogeneity algorithm.
- Client-side or user-level DP papers report the privacy unit, clipping norm, noise multiplier, participation schedule, accountant, and epsilon/delta. This level of reporting is mandatory for our paper.

Federated LoRA/PEFT literature:

- Fed-SB is the closest method and must be treated as the strongest algorithmic baseline, not only as related work.
- LoRA-XS and PiSSA make public or SVD-informed low-rank subspaces mainstream enough that we should not overclaim "SVD adapters" as novel by itself.
- Newer federated LoRA papers such as FedALT and LoRA-FAIR emphasize heterogeneity, personalization, and communication efficiency. We cite them to motivate local residual personalization and fair communication-controlled comparisons, but we do not need to implement every new variant for the first SCI submission.

### 18.2 Refined Central Thesis

The paper should be written around this thesis:

> Under a mainstream Edge-IIoTset closed-set IDS protocol, client-side DP makes federated LLM adaptation difficult. SPECTRA-FedCore improves the privacy-utility-communication trade-off by using only public backbone spectra to choose compact core-adapter subspaces and privacy-aware layer budgets, while keeping the experimental protocol comparable to standard IDS studies.

The strongest story is not:

```text
LLMs beat XGBoost on tabular IDS.
```

The strongest story is:

```text
Small LLM security assistants can be privately federated under edge budgets when adaptation is restricted to public-spectral cores, and the loss from client-side DP can be reduced without relying on private-data-dependent adapter bases.
```

### 18.3 Final Main Data Protocol

Primary dataset:

```text
data/raw/edgeiiotset/ML-EdgeIIoT-dataset-full.csv
```

Primary task:

```text
15-class closed-set classification
```

Normalized label space:

```text
Normal
Backdoor
DDoS_HTTP
DDoS_ICMP
DDoS_TCP
DDoS_UDP
Fingerprinting
MITM
Password
Port_Scanning
Ransomware
SQL_injection
Uploading
Vulnerability_scanner
XSS
```

Main split:

```text
80/10/10 stratified row-level split
seed = 20260531
```

Rules:

- Save train/validation/test indices to disk.
- Fit all transforms only on the training split.
- Apply class balancing only to training data if used.
- Do not use the test set for prompt feature choice, quantile bins, feature selection, hyperparameter tuning, or DP parameter tuning.
- Report duplicate diagnostics across splits after removing excluded identifier-like columns.

Robustness checks:

1. Exact duplicate or feature-hash overlap check on the main split.
2. Deduplicated sensitivity result if cross-split exact duplicates are material.
3. Raw full-package per-file 8/1/1 split as a conventional robustness check.

Do not use unseen-file, unseen-source, unseen-attack, or time-aware heldout as the main paper task.

### 18.4 Prompt and Feature Design

Main prompt uses compact tabular-to-instruction rendering:

```text
System:
You are an edge IIoT security assistant. Classify telemetry locally.

User:
Traffic features:
tcp.flags={...}; tcp.len={...}; tcp.dstport={...}; ...

Allowed labels:
Normal | Backdoor | ... | XSS

Assistant:
{label}
```

Main feature set:

- 20-35 curated protocol and traffic-behavior fields.
- Exclude `frame.time`, `ip.src_host`, `ip.dst_host`, raw payload/message fields, URI/full-URI fields, referer fields, and label fields.
- Prefer stable packet/protocol statistics: TCP flags, TCP/UDP length and ports if not too identifier-like, DNS query length/type, MQTT length/type/topic length, Modbus length/unit fields, and protocol flags.
- Record prompt token length distribution.

Inference:

- Use constrained label decoding or exact label extraction to avoid free-form answer ambiguity.
- The supervised target is only the label string.
- Explanation generation can appear as qualitative examples, not as a main metric.

Ablations:

1. Compact prompt vs shorter prompt.
2. Compact prompt vs broader feature prompt.
3. Numeric-only classical baselines vs LLM prompt model.

### 18.5 Single-Server FL Simulation

The user will provide one server, so all FL experiments are simulated on one machine.

Simulation design:

```text
K = 10 clients
local_epochs = 1
rounds = 20 for sanity, 50 for main runs if compute allows
participation = 100% for debugging, then 20%-50% for reported FL sweeps
primary non-IID = Dirichlet label-skew, alpha = 0.5
secondary sensitivity = K = 20 if server time allows
```

Client partitions:

1. IID sanity:
   Stratified client shards to verify the implementation and upper-bound behavior.

2. Label-skew non-IID:
   Dirichlet partition over labels. This is the main FL setting because it is standard, controllable, and easy to reproduce.

3. Source/protocol-skew secondary:
   Use source/protocol hints only if easy and stable. This should not dominate the paper.

Evaluation:

- Keep validation and test sets global and fixed across methods.
- Report client-weighted aggregation by local sample count.
- Save per-client label histograms for every partition seed.
- Report convergence curves, not only final scores.

### 18.6 Refined Method Components

Core method:

1. Public spectral basis:
   Compute layer bases from public frozen backbone weights only.

2. Budget-aware rank allocation:
   Choose `p_l` from public spectral energy under trainable-parameter and communication budgets.

3. Client-side DP release:
   Clip and noise only uploaded global-path core updates on the client before server aggregation.

4. Layer-adaptive clipping/noise:
   Allocate clipping/noise by public layer spectra and core dimensions. Do not tune private clipping thresholds from raw client updates unless accounted for.

5. Local residual core:
   Keep a small client-local residual unuploaded for personalization under non-IID data.

6. Optional shrinkage:
   Apply deterministic post-processing to noised aggregate cores. Since DP is immune to post-processing, this does not spend extra privacy budget.

What not to overclaim:

- Training only a square core is not unique.
- Exact aggregation of the core is not unique.
- SVD-informed subspaces are not unique.
- The contribution is the public-only spectral allocation and DP utility recovery stack in a single Edge-IIoT FL protocol.

### 18.7 Baseline Hierarchy

Minimum baselines:

1. Classical IDS context:
   `XGBoost` or `LightGBM`, `Random Forest`, and `MLP`.

2. Central and local adapter context:
   Central LoRA, local-only LoRA or local-only SPECTRA core, and prompt-only Gemma.

3. Federated adapter baselines:
   FedAvg-LoRA and Fed-SB.

4. DP baselines:
   FedAvg-LoRA-DP, Fed-SB-DP, and SPECTRA-FedCore-DP.

Optional baselines:

- FedProx-LoRA if FedAvg-LoRA is unstable under label skew.
- DP-DyLoRA if implementation is straightforward.
- FedALT or LoRA-FAIR only if code and compute budget make them feasible; otherwise cite them in related work and explain that Fed-SB is the closest communication/DP collision baseline.

Fairness regimes:

1. Same epsilon and delta.
2. Same uploaded MB per client per round.
3. Same trainable adapter parameter budget.
4. Same backbone, split, prompt renderer, optimizer budget, and FL schedule where possible.

### 18.8 Main Tables and Figures

Table 1: Main closed-set FL result.

```text
Dataset: ML-EdgeIIoT, 80/10/10 stratified split
Clients: K=10, IID and Dirichlet alpha=0.5
Methods: FedAvg-LoRA, Fed-SB, SPECTRA-FedCore
Privacy: non-DP and epsilon in {8, 4, 2, 1}
Metrics: macro-F1, rare-attack recall, weighted-F1, uploaded MB/round
```

Table 2: Fair controlled comparison.

```text
Same epsilon
Same communication
Same trainable parameters
Report macro-F1, rare recall, peak VRAM, wall time
```

Table 3: Classical IDS context.

```text
Random Forest
XGBoost/LightGBM
MLP
Central LoRA
SPECTRA-FedCore non-DP
```

Table 4: Ablation.

```text
Random basis vs public SVD basis
Uniform rank vs spectral rank allocation
Uniform noise vs layer-adaptive noise
No residual vs local residual
No shrinkage vs shrinkage
Compact prompt vs short/broad prompt
```

Table 5: Reproducibility and leakage diagnostics.

```text
Duplicate rate across main split
Deduplicated sensitivity if needed
Raw per-file 8/1/1 robustness
Method ranking stability
```

Figure 1: System diagram.

```text
Public backbone -> public spectral bases -> client local training -> client-side DP release -> server aggregate -> global core + local residual
```

Figure 2: Privacy-utility-communication Pareto.

```text
x-axis: uploaded MB/round
y-axis: macro-F1 or rare recall
markers: epsilon values
```

Figure 3: DP degradation curve.

```text
epsilon vs macro-F1
methods: FedAvg-LoRA-DP, Fed-SB-DP, SPECTRA-FedCore-DP
```

Figure 4: Ablation waterfall.

```text
Uniform core under DP -> spectral rank -> adaptive noise -> residual -> shrinkage
```

### 18.9 Manuscript Writing Pattern

Introduction should follow this structure:

1. IIoT gateways need collaborative IDS because single sites see incomplete attacks.
2. Raw telemetry cannot be centralized; FL helps, but FL alone does not provide formal privacy.
3. Client-side DP is stronger and auditable, but it sharply hurts LLM adapter utility.
4. Existing private federated LoRA methods are generic and do not exploit public backbone spectra for edge-budgeted IIoT security.
5. SPECTRA-FedCore uses public spectral cores, budget-aware allocation, and DP-compatible personalization to recover utility under strict communication and privacy budgets.

Related work should be organized by reviewer expectation:

1. Edge-IIoTset and IIoT IDS evaluation practice.
2. Federated and differentially private IDS.
3. PEFT and spectral adapters.
4. Federated LoRA and Fed-SB collision boundary.

Experimental section should open with reproducibility:

- dataset version and checksum
- label normalization
- split seed and indices
- excluded fields
- client partition seeds
- backbone checkpoint
- adapter target modules
- DP accountant
- hardware and runtime

Discussion should be explicit:

- Classical models may remain stronger for pure tabular classification.
- The LLM value is the private federated security-assistant path and future operator interaction.
- Main split is closed-set and comparable, not an unseen-source deployment proof.
- Client-side DP utility loss is expected; the contribution is reducing it under fixed budgets.

### 18.10 Minimum Publishable SCI Package

If server time is limited, the minimum publishable package is:

1. Main selected-ML split with saved indices and duplicate diagnostics.
2. Classical baselines: Random Forest, XGBoost/LightGBM, MLP.
3. FedAvg-LoRA and Fed-SB on the same prompt/split.
4. SPECTRA-FedCore non-DP and DP at epsilon `{8, 4, 2, 1}`.
5. Ablation of public SVD basis, spectral allocation, adaptive noise, and local residual.
6. Communication, trainable parameter, memory, and latency reporting.
7. Raw per-file 8/1/1 robustness check if compute allows after main tables.

This keeps the paper focused: method novelty is high, evaluation style is familiar, and rigor is visible without turning the benchmark into a niche OOD study.
