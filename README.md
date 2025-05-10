# course-project-03-ASSEMBLY


# Project ASSEMBLY — Energy-Aware Adaptive Batch-Level Scheduler for IIoT Fog–Cloud Environments

> **Core Contribution:** A **two-tier scheduler** combining:
- **BDRL-PPO (Bayesian Tiny Fast-Adaptive Proximal Policy Optimiser)** for high-level resource partitioning between fog and cloud.
- **ABS (Adaptive Batch Scheduler)** based on **Genetic Programming** for fine-grained task scheduling.

The system aims to **minimize energy consumption** while ensuring that batch workflows meet their soft deadlines (**QoS compliance**).

---

## 1. Motivation

Industrial IoT (IIoT) systems often run **batch workflows** during off-peak hours or shifts. These workflows can consist of **100s to 1000s of parallel tasks** that process large sensor logs and produce quality reports.

While these tasks are not real-time, their **energy cost** is significant due to their scale. Traditional **static heuristics** fail to adapt to changing conditions like workload size and node availability.

**ASSEMBLY** solves this with a **learn-and-adapt** approach:
- **BDRL-PPO** learns and adapts the **fog vs. cloud allocation ratio** quickly.
- **ABS** evolves a **fine-grained schedule** respecting BDRL's guidance.

---

## 2. Formal Problem Definition

### 2.1 System Model

- **Nodes:** Set of **fog nodes** (𝑭) and **cloud nodes** (𝑪). Each has:
  - CPU speed *Sₙ* [MIPS]
  - Idle & busy power: *Pᵢₙ*, *P_bₙ* [W]
  - DVFS levels: *Lₙ*
- **Network:** Bandwidth *Bₙₘ* and latency *RTTₙₘ* between any pair
- **Workload:** DAG *G = (V,E)* where:
  - *wᵥ*: Task computation (MI)
  - *eᵥ*: Input data size (MB)
- **Deadline:** Soft deadline Δᴮ for entire batch

### 2.2 Decision Variables

| Variable | Domain | Meaning |
|--------|--------|---------|
| *ρ* | [0,1] | Fog allocation ratio from BDRL |
| *(xᵥ,tᵥ)* | (node, slot) | ABS assigns task *v* to node *xᵥ* at time *tᵥ* |
| *lᵥ* | DVFS level | Chosen for executing task *v* |

### 2.3 Objective

Minimize total energy **E_total**, including compute and network costs, subject to meeting the deadline:

$$
\min_{ρ,x,t,l}\;E_\text{total}=\sum_{v∈V}\Big(P_b^{x_v}(l_v)·\frac{w_v}{S_{x_v}(l_v)}+\sum_{(v,u)∈E}\frac{e_{vu}}{B_{x_vx_u}}·P_\text{net}\Big)\]

Plus a penalty if the makespan exceeds the deadline:
$$
λ·\max(0,\text{finish}-Δ^B)
$$

---

## 3. Bayesian DRL Component (BDRL-PPO)

### 3.1 Why Bayesian?

Standard PPO models have fixed weights that don’t adapt well to new regimes. In contrast, **Bayesian PPO** treats weights as Gaussian distributions (mean + variance), allowing:

- **Uncertainty estimation**: High variance means uncertain decisions (useful for adaptation).
- **Few-shot adaptation**: One gradient update after a small rollout adapts the policy fast.

This allows **tiny model size (<200KB)** and **fast retraining** (under 10ms on edge devices).

### 3.2 States, Actions, Rewards

| Symbol | Shape | Description |
|--------|-------|-------------|
| *sₜ* | 7-D vector | ⟨pending_tasks, Fog_CPU%, Cloud_CPU%, avg_DVFS_level, SoC_battery, mean_RTT, time_to_deadline⟩ |
| *aₜ=ρₜ* | scalar | Fraction of next H tasks to prefer fog |
| *rₜ* | scalar | Reward: −α·E_window − β·lateness_window |

Where *H* = planning horizon (~50 tasks). *E_window* is total energy used in the current window. *lateness_window* penalizes missed deadlines.

### 3.3 Training Loss

Uses **Bayesian Advantage-Weighted PPO**:

$$
\mathcal L(µ,σ)=\mathbb E_{θ∼\mathcal N(µ,σ^2)}\Big[\mathcal L_\text{clip}(θ)+c_1·\text{MSE}(V_θ,R)+c_2·\text{entropy}(π_θ)\Big]−β_{KL}·KL\big(\mathcal N(µ,σ^2)||\mathcal N(µ_0,σ_0^2)\big)
$$

This includes:
- Clipped surrogate objective (`PPO`)
- Value function loss (`MSE`)
- Entropy bonus for exploration
- KL divergence regularization (Bayesian prior)

Gradients use **reparameterization trick**.

### 3.4 Online Few-Shot Adaptation

When a new batch arrives:
1. Collect a **small rollout** (≤ 2×H steps)
2. Update the Bayesian parameters (*µ, σ*) using one ELBO gradient step
3. Takes **<10ms** on an ARM-based edge device

This ensures rapid adaptation to new workloads without full retraining.

---

## 4. Adaptive Batch Scheduler (ABS) via Genetic Programming

### What It Does:

ABS evolves **program trees** that take `(task_features, node_features)` and output `(node_id, slot, DVFS_level)`.

- **Terminals:** e.g., `fog_load`, `task_MI`, `earliest_slot`, `ρ`
- **Operators:** {+, −, ×, ÷, min, max, if-less}

Each evolved tree represents a **scheduling strategy**.

### Initialization:

Population initialized so that **fog preference = ρ** from BDRL → cross-layer consistency.

### Fitness Function:

$$
F = E_\text{total} + λ·\text{lateness} + ξ·\text{variance}(node\_energy)
$$

This balances:
- Energy usage
- Deadline violations
- Load balancing across nodes

---

## 5. Pseudo-Code Suite

### 5.1 BDRL-PPO (Offline Training)

```text
Algorithm BDRL_Train(trace, µ₀, σ₀)
1  initialise µ ← µ₀, σ ← σ₀
2  for each episode = one historical batch do
3      Rollout policy π_{µ,σ} for T steps (sample θ ∼ N(µ,σ²))
4      Compute advantages Â via GAE
5      Update µ ← µ + η·∇_µ L(µ,σ)   // Adam optimiser
6      Update σ ← σ + η·∇_σ L(µ,σ)
7  end for
8  save µ,σ as prior
```

**What it does:** Trains the Bayesian DRL policy using historical batches. Updates both mean and variance of the weight distribution.

---

### 5.2 BDRL On-Line Adaptation (per arriving batch)

```text
Algorithm BDRL_Adapt(cur_state, µ, σ)
1  Sample θ ∼ N(µ, σ²); act ρ ← π_θ(cur_state)
2  Execute ρ for horizon H; collect tuple (s,a,r,s')
3  Compute one-step ELBO gradients g_µ, g_σ
4  µ ← µ + η_adapt·g_µ; σ ← σ + η_adapt·g_σ
5  return ρ, µ, σ
```

**What it does:** Adapts the Bayesian model in real-time based on current state and feedback from execution.

---

### 5.3 ABS Genetic Programming Loop

```text
Algorithm ABS_Evolve(G, ρ, PopSize N, Gen G_max)
1  P ← Initialise(N, bias_fog=ρ)
2  for g = 1…G_max do
3      for χ in P do
4          χ ← DVFS_Repair(χ)
5          fit(χ) ← EnergySim(G, χ)
6      end
7      P ← Tournament_Select(P)
8      P ← Crossover_Mutate(P)
9  end
10 return argmin_χ fit(χ)
```

**What it does:** Evolves a population of scheduling strategies (trees), evaluates them, selects the best, and evolves new ones over generations.

---

### 5.4 DVFS-Aware Repair

```text
DVFS_Repair(χ)
for task v in χ:
    node ← x_v; slot ← t_v
    choose lowest DVFS level l such that finish_time ≤ Δᴮ
    if not feasible: migrate v to fastest node in F∪C
return χ
```

**What it does:** Ensures every scheduled task respects the deadline by choosing appropriate DVFS levels. If not possible, migrates to faster node.

---

### 5.5 Energy Simulator Stub

```text
EnergySim(G, χ)
E_total ← 0; lateness ← 0
for each task v in topological order:
    calc start, finish according to χ & predecessors
    P ← P_busy^{x_v}(l_v); E_total += P * (finish-start)
    update node timeline
lateness = max(0, max_finish - Δᴮ)
return E_total + λ*lateness + ξ*variance(node_energy)
```

**What it does:** Simulates the execution of a schedule, computes energy, checks deadline, adds penalties, returns fitness value.

---

### 5.6 ASSEMBLY Master Scheduler

```text
Scheduler_Run(batch G, µ, σ)
1  cur_state ← telemetry()
2  ρ, µ, σ ← BDRL_Adapt(cur_state, µ, σ)
3  χ* ← ABS_Evolve(G, ρ, N=30, G_max=100)
4  enact χ* on runtime orchestrator
5  record energy stats for next batch
```

**What it does:** Coordinates the whole workflow:
1. Gets system telemetry
2. Uses BDRL to adapt fog-cloud preference
3. Runs genetic programming to evolve best schedule
4. Executes the schedule
5. Records results for future learning

---

## 6. Variable Glossary (Complete)

| Symbol | Meaning | Units |
|--------|---------|-------|
| *G=(V,E)* | Batch workflow DAG | – |
| *v,u* | Task indices | – |
| *wᵥ* | CPU work of task | MI |
| *eᵥ* | Data size | MB |
| *Sₙ(l)* | MIPS at DVFS level *l* | MI/s |
| *Pᵢₙ, P_bₙ(l)* | Idle & busy power draw | W |
| *ρ* | Fog preference ratio | – |
| *µ,σ²* | Mean & variance of Bayesian policy weights | – |
| *θ* | Concrete weight sample | – |
| *H* | Horizon length for BDRL action | tasks |
| *α,β* | Reward weights (energy, lateness) | – |
| *λ,ξ* | ABS fitness weights | – |
| *η,η_adapt* | Learning rates (train, adapt) | – |
| *N, G_max* | ABS population & generation counts | – |
| *lᵥ* | DVFS level for task *v* | index |
| *Δᴮ* | Soft deadline for batch | s |

---

## 7. Dataset & Evaluation Details

- **Dataset:** Alibaba Cluster Trace 2018
  - Generate 14 workload sizes from 100 to 3000 tasks
- **Simulator:** iFogSim 2 or cloudsim extended with:
  - Per-node DVFS
  - Bayesian policy plugin

- **Metrics:**
  - Total energy (E_total)
  - Makespan
  - QoS violation %
  - Adaptation time
  - Policy uncertainty (avg σ)

---


## 9. Deliverables

- **Source Code:** PyTorch + Java implementation with Makefile
- **Documentation:** README, API docs, this design doc
- **Visualizations:** Plots of energy curves, uncertainty evolution, convergence

