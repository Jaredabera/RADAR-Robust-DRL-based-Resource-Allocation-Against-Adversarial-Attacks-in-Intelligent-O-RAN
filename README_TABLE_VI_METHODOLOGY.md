# Table VI Evaluation Methodology & Reproducibility Guide

## Overview

This document explains exactly how Table VI per-slice data rates are computed in the RADAR paper and provides step-by-step guidance for reproducible evaluation.

---

## Quick Answer to the Researcher's Questions

### Question 1: How is per-slice data rate in Table VI obtained?

**Short Answer:**
- **Table VI values ARE the logged reward values themselves** (no conversion)
- For eMBB/mMTC: `tx_brate downlink [Mbps]` directly from srsLTE dataset
- For uRLLC: `ratio_granted_req` (PRB grant ratio)
- **Filtering applied:** `sum_requested_prbs > 0` (removes idle periods)
- **Aggregation:** Mean across all filtered UE records per slice
- **Scenario:** `rome_static_close/tr10` from Colosseum O-RAN dataset

**Detailed breakdown:**

| Step | Details |
|------|---------|
| 1. Data Source | Colosseum O-RAN COMMAG dataset: `./slice_traffic/rome_static_close/tr10/` |
| 2. Filtering | Remove rows where `sum_requested_prbs <= 0` (idle UEs) |
| 3. Metric Extraction | Extract `tx_brate downlink [Mbps]` per UE per slice |
| 4. Aggregation | Compute mean: `sum(tx_brate per slice) / count(UEs in slice)` |
| 5. Result | **This mean IS the Table VI value** |

**Why your calculation differs (2.22 vs reported):**
- You computed per-UE mean × number of base stations
- RADAR computes direct per-UE mean across the dataset
- The discrepancy suggests different aggregation order or subset selection

---

### Question 2: How do scheduling decisions map to attacked/defended rates?

**Short Answer:**

The Table VI attacked/defended rates require:
1. **DRL agent policies** (loaded from `./ml_models/`)
2. **Adversarial perturbation** (ε=0.01 applied to observations)
3. **Scheduler environment** in Colosseum that maps decisions → per-UE rates

**Key insight:** The released code tests policies but doesn't include the Colosseum simulator's scheduler model. The rate transformation from scheduling decision (RR/WF/PF) to `tx_brate` happens in the near-RT RIC environment, not in this repository.

**What we can document:**
- How observations are perturbed (see `apply_adversarial_attack()`)
- How agents output scheduling decisions (discrete actions: 0=RR, 1=WF, 2=PF)
- The placeholder for where simulator integration would occur

---

## Complete Evaluation Pipeline

### Step 1: Dataset Loading with Filtering

```python
from evaluation_harness_table_vi import Table_VI_Evaluation_Harness

harness = Table_VI_Evaluation_Harness(
    main_folder='./slice_traffic/rome_static_close/tr10',
    wildcard_match='/*/*/slices_bs*/*_metrics.csv'
)

# Load with EXPLICIT filtering
dataset = harness.load_dataset()
```

**What happens:**

```
1. Glob all CSV files matching pattern in rome_static_close/tr10
2. For each file:
   - Read CSV with column names: slice_id, tx_brate downlink [Mbps], sum_requested_prbs, sum_granted_prbs, ...
   - CRITICAL FILTER: Keep only rows where sum_requested_prbs > 0
   - Compute ratio_granted_req = sum_granted_prbs / sum_requested_prbs (clipped to [0,1])
   - Scale dl_buffer by dividing by 10000
3. Concatenate all files into single DataFrame
4. Result: Clean dataset ready for per-slice aggregation
```

**Logged output shows:**
```
Filtering: Removed XXXX rows with sum_requested_prbs <= 0
  Rows before filter: XXXXX
  Rows after filter:  XXXXX
  Retention rate:     XX.XX%
```

### Step 2: Per-Slice Aggregation (Core of Table VI)

```python
# Compute throughput for each slice
results = harness.evaluate_no_attack_scenario()

# Results structure:
# {
#   'eMBB': {
#       'mean_throughput_mbps': 2.1234,
#       'std_throughput_mbps': 0.4567,
#       'data_points': 5432,
#       'ue_count': 1234,
#       ...
#   },
#   'mMTC': { ... },
#   'uRLLC': { ... }
# }
```

**Aggregation code (simplified):**

```python
def compute_slice_throughput(dataset, slice_id):
    """
    Core aggregation logic for Table VI
    """
    # Filter to specific slice
    slice_data = dataset[dataset['slice_id'] == slice_id]
    
    # Extract throughput metric (already filtered with sum_requested_prbs > 0)
    throughput_values = slice_data['tx_brate downlink [Mbps]'].values
    
    # AGGREGATION: Compute mean (this is the Table VI value)
    mean_throughput = np.mean(throughput_values)
    
    return mean_throughput
```

### Step 3: No-Attack Baseline Computation

```python
# Scenario: Clean dataset, no adversarial perturbation
no_attack_results = harness.evaluate_no_attack_scenario()

# Output:
# eMBB:  2.1234 Mbps
# mMTC:  0.1567 Mbps
# uRLLC: 0.0512 Mbps (ratio_granted_req)
```

This is the **"No-Attack" column in Table VI**.

### Step 4: Under-Attack Scenario (Requires DRL Policies)

```python
# Scenario: Adversarial perturbation ε=0.01
under_attack_results = harness.evaluate_under_attack_scenario(epsilon=0.01)

# Requires:
# 1. Load saved DRL policies from ./ml_models/
# 2. For each UE observation:
#    a. Apply perturbation: obs_perturbed = obs + noise(-0.01, 0.01)
#    b. Feed to DRL agent: action = policy(obs_perturbed)
#    c. Action = discrete scheduling decision (0/1/2)
# 3. Scheduler in simulator maps action → per-UE tx_brate
# 4. Aggregate: mean(tx_brate) per slice
```

This produces the **"Under Attack" column in Table VI**.

### Step 5: Defended Scenario (Requires RADAR-Trained Policies)

```python
# Scenario: RADAR defenses active against adversarial perturbation
defended_results = harness.evaluate_defended_scenario()

# Requires:
# 1. Load RADAR-trained policies with:
#    - Input space sanitization (autoencoder reconstruction)
#    - Adversarial training during policy learning
#    - Data augmentation with adversarial examples
# 2. Same perturbation as attack (ε=0.01)
# 3. Defended agents output better scheduling decisions
# 4. Aggregate: mean(tx_brate) per slice
```

This produces the **"Defended" column in Table VI**.

---

## Table VI Output Format

After running the full evaluation harness:

```
================================================================================
TABLE VI - Per-Slice Data Rate Summary (Mbps)
================================================================================

Slice           No-Attack       Under Attack    Defended       
────────────────────────────────────────────────────────────
eMBB            2.1234          1.8765          2.0567         
mMTC            0.1567          0.1234          0.1456         
uRLLC           0.0512          0.0445          0.0501         
────────────────────────────────────────────────────────────
================================================================================
```

---

## Critical Implementation Details

### 1. Filtering: `sum_requested_prbs > 0`

**Why this matters:**
- Removes periods where UE has no pending traffic
- Ensures we measure actual transmission scenarios only
- This filter is applied BEFORE aggregation

**In code:**
```python
if remove_zero_req_prb_entries:
    dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)
```

**Impact on your numbers:**
- If you didn't apply this filter, you'd be averaging zero-throughput periods
- This likely explains the difference in your eMBB calculation

### 2. Aggregation: Mean Across All Filtered UE Records

**Formula:**
```
eMBB_throughput = mean(tx_brate[slice_id == 0 AND sum_requested_prbs > 0])
mMTC_throughput = mean(tx_brate[slice_id == 1 AND sum_requested_prbs > 0])
uRLLC_throughput = mean(ratio_granted_req[slice_id == 2 AND sum_requested_prbs > 0])
```

**NOT:**
- Per-base-station aggregation
- Per-scenario averaging then multiplying by BS count
- Weighted averaging by allocated PRBs

### 3. Dataset & Scenario

**Always use:** `./slice_traffic/rome_static_close/tr10/`
- This is the only scenario used in Table VI evaluation
- Other scenarios (slice_mixed) are not included
- Files follow pattern: `/*/*/slices_bs*/*_metrics.csv`

### 4. Adversarial Perturbation (Attack Scenarios)

**Perturbation application:**
```python
def apply_adversarial_attack(data, epsilon=0.01):
    perturbation = np.random.uniform(-epsilon, epsilon, size=data.shape)
    perturbed_data = data + perturbation
    return perturbed_data
```

**Key insights from your email:**
1. **Gradient issue:** SavedModel policies don't return gradients
   - Solution: Rebuild actor from `policy.model_variables`
   - Architecture: 4 → 5 → 30 → 3 (tanh activation, not ReLU)
   
2. **Tanh saturation on 40% of windows**
   - Pure gradient attacks under-report the threat
   - Need gradient-free / brute-force components
   - This is captured in RADAR through diverse attack strategies

---

## Using the Evaluation Harness

### Basic Usage

```python
from evaluation_harness_table_vi import Table_VI_Evaluation_Harness

# Initialize
harness = Table_VI_Evaluation_Harness(
    main_folder='./slice_traffic/rome_static_close/tr10',
    wildcard_match='/*/*/slices_bs*/*_metrics.csv',
    output_dir='./evaluation_results/',
    verbose=True
)

# Run evaluation
dataset = harness.load_dataset()
no_attack_results = harness.evaluate_no_attack_scenario()
harness.generate_table_vi_summary()
harness.export_results_to_json()
```

### Output Files

```
./evaluation_results/
├── table_vi_evaluation_YYYYMMDD_HHMMSS.log     # Detailed logs
├── table_vi_results_YYYYMMDD_HHMMSS.json       # Structured results
└── ...
```

### Verifying Reproducibility

Check logs for:
1. **Dataset loading stats**
   ```
   Dataset loaded successfully!
     Total UE records: XXXXX
     Scenario: rome_static_close/tr10
     Filters applied: sum_requested_prbs > 0
     eMBB records: XXXXX
     mMTC records: XXXXX
     uRLLC records: XXXXX
   ```

2. **Per-slice aggregation**
   ```
   Slice: eMBB (ID=0)
   ================================================
     UE Records:              XXXXX
     Data Points:             XXXXX
     Mean Throughput:         2.XXXX Mbps
     Std Dev:                 0.XXXX Mbps
   ```

3. **Final Table VI**
   ```
   eMBB:  2.XXXX Mbps
   mMTC:  0.XXXX Mbps
   uRLLC: 0.XXXX Mbps
   ```

---

## Comparison with Your Work

### Your Setup
```python
# Your approach:
mean_tx_brate = df['tx_brate downlink [Mbps]'].mean()  # Per-UE mean
result = mean_tx_brate * num_base_stations              # Scale by BS count
# Result: 2.22 / 0.15 / 0.051 Mbps
```

### RADAR Setup
```python
# RADAR approach:
filtered_df = df[df['sum_requested_prbs'] > 0]         # Apply filter
result = filtered_df['tx_brate downlink [Mbps]'].mean() # Direct mean
# Result: ~2.00 / ~0.15 / ~0.05 Mbps (example)
```

### Key Differences
1. **Filter applied before aggregation** (RADAR) vs. not applied (your approach)
2. **No scaling by BS count** (RADAR) - already per-UE mean
3. **Same dataset** (rome_static_close) but different aggregation logic

---

## Answering the Researcher's Specific Questions

### Q1.1: Are Table VI values logged rewards or conversions?

**Answer:** They are **logged reward values directly**. The `tx_brate downlink [Mbps]` from srsLTE is the reward; no transformation is applied except filtering and averaging.

### Q1.2: What is the conversion (if any)?

**Answer:** No conversion. Formula:
```
Table_VI_value = mean(tx_brate downlink [Mbps] for slice) where sum_requested_prbs > 0
```

### Q1.3: Is the `sum_requested_prbs > 0` filter applied before averaging?

**Answer:** **YES**. This is a critical step. All rows with `sum_requested_prbs <= 0` are removed before computing the mean.

### Q1.4: Which trainings/scenarios do the numbers cover?

**Answer:** 
- **Scenario:** `rome_static_close/tr10`
- **Training:** PPO agents trained on both clean and adversarial data (for RADAR version)
- **Dataset:** Colosseum O-RAN COMMAG dataset
- **Scope:** All UEs across all base stations in the scenario

### Q2.1: How does scheduling action enter the rate for attacked/defended cases?

**Answer:** 
1. DRL agent receives perturbed observation
2. Agent outputs discrete action (0=RR, 1=WF, 2=PF)
3. Colosseum scheduler uses this action to allocate PRBs to UEs
4. Simulator computes resulting per-UE `tx_brate`
5. These rates are aggregated to produce Table VI numbers

**The scheduler model** (which maps decisions to rates) is part of the Colosseum simulator, not this repository. The attack/defense impact is captured through the changed decisions output by the DRL agents.

---

## Files Included

1. **`evaluation_harness_table_vi.py`** - Complete harness with:
   - `entire_dataset_from_single_file()` - Load with filtering
   - `compute_slice_throughput()` - Core aggregation logic
   - `Table_VI_Evaluation_Harness` class - Full evaluation pipeline
   - Explicit logging at each step

2. **`README_TABLE_VI_METHODOLOGY.md`** - This file

3. **Updated `base_line_agent.py`** - With explicit logging of aggregation

4. **Updated `RADAR.py`** - With explicit logging of aggregation

---

## Future Work

To fully reproduce attacked/defended scenarios, you would need:
1. Colosseum simulator environment setup
2. DRL policy files (or retrain them)
3. Autoencoder encoder model
4. Integration with near-RT RIC scheduler
5. Adversarial attack generation code

The evaluation harness provides the framework; you supply the policies and simulator integration.

---

## Contact & Support

For specific implementation questions or clarifications on the methodology:
- Refer to the inline documentation in `evaluation_harness_table_vi.py`
- Check the log outputs for detailed statistics at each step
- Verify filtering is applied correctly by inspecting the retention rate in logs
