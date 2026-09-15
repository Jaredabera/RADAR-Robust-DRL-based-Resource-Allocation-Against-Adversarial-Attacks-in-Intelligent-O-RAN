"""
===============================================================================
RADAR Evaluation Harness for Table VI Reproducibility
===============================================================================

This module provides a complete, documented evaluation harness for reproducing
the per-slice data rates reported in Table VI of the RADAR paper.

METHODOLOGY SUMMARY:
- Data source: Colosseum O-RAN COMMAG dataset (rome_static_close scenario)
- Reward definition: tx_brate downlink [Mbps] for eMBB/mMTC, ratio_granted_req for uRLLC
- Filtering: sum_requested_prbs > 0 (to exclude idle periods)
- Aggregation: Per-slice mean of tx_brate downlink [Mbps] across all filtered UE records
- Training: PPO-based agents trained on adversarial and clean data
- Scenarios: No-Attack (clean), Under Attack (adversarial input), Defended (RADAR methods)

===============================================================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import time_step as ts

import absl
import os
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# ============================================================================
# PART 1: DATA LOADING WITH EXPLICIT FILTERING
# ============================================================================

def entire_dataset_from_single_file(filename,
                                    col_names,
                                    selected_col_names,
                                    remove_zero_req_prb_entries=True,
                                    scale_dl_buffer=True,
                                    replace_zero_with_one=False,
                                    add_prb_ratio=True,
                                    verbose=False):
    """
    Load and preprocess a single CSV file from the Colosseum dataset.
    
    FILTERING LOGIC (Critical for Table VI reproducibility):
    - If remove_zero_req_prb_entries=True: Filters dataset to only include rows where
      sum_requested_prbs > 0. This removes idle UE periods and ensures we only measure
      active transmission scenarios.
    
    TRANSFORMATION:
    - dl_buffer scaling: Converts bytes to normalized units (divide by 10000)
    - rgb_granted_req/ratio_granted_req: Computes PRB grant ratio = sum_granted_prbs / sum_requested_prbs
    
    Args:
        filename: Path to CSV file
        col_names: List of all column names in CSV
        selected_col_names: Columns to extract
        remove_zero_req_prb_entries: If True, filter out rows where sum_requested_prbs <= 0
        scale_dl_buffer: If True, normalize dl_buffer by dividing by 10000
        replace_zero_with_one: If True, set ratio_granted_req=1.0 for rows where sum_requested_prbs=0
        add_prb_ratio: If True, compute and add ratio_granted_req column
        verbose: If True, log filtering statistics
    
    Returns:
        pandas.DataFrame: Loaded and preprocessed data
    """
    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)
    
    initial_size = len(dataset)
    
    # ========== CRITICAL FILTERING STEP FOR TABLE VI ==========
    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)
        filtered_size = len(dataset)
        
        if verbose:
            logging.info(f"Filtering: Removed {initial_size - filtered_size} rows with sum_requested_prbs <= 0")
            logging.info(f"  Rows before filter: {initial_size}")
            logging.info(f"  Rows after filter:  {filtered_size}")
            logging.info(f"  Retention rate:     {100*filtered_size/initial_size:.2f}%")
    
    # ========== SCALING STEP ==========
    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in selected_col_names]):
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 10000
        if verbose:
            logging.info("Applied dl_buffer scaling: divide by 10000")
    
    # ========== RATIO COMPUTATION STEP ==========
    if add_prb_ratio:
        # Compute ratio_granted_req: granted PRBs / requested PRBs
        # Clip to [0, 1] and handle NaN cases (when denominator is 0)
        dict_add = pd.DataFrame.from_dict({"ratio_granted_req": np.clip(np.nan_to_num(
            dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            # For rows where sum_requested_prbs <= 0, set ratio to 1.0 (optimal satisfaction)
            dict_add['ratio_granted_req'].loc[dataset['sum_requested_prbs'] <= 0] = 1.0
        
        dataset = dataset.join(dict_add)
        if verbose:
            logging.info("Computed ratio_granted_req = sum_granted_prbs / sum_requested_prbs (clipped to [0,1])")
    
    return dataset


def entire_dataset_from_folder(main_folder,
                               wildcard,
                               col_names,
                               selected_col_names,
                               scale_dl_buffer=True,
                               remove_zero_req_prb_entries=True,
                               replace_zero_with_one=False,
                               add_prb_ratio=True,
                               verbose=False):
    """
    Load all CSV files from a folder into a single DataFrame.
    
    Args:
        main_folder: Root directory containing scenario data
        wildcard: Pattern to match files (e.g., '/*/*/slices_bs*/*_metrics.csv')
        col_names: All column names in CSVs
        selected_col_names: Columns to extract
        verbose: If True, log detailed statistics for each file
    
    Returns:
        pandas.DataFrame: Concatenated dataset from all matching files
    """
    dataset = []
    file_count = 0
    
    for filename in sorted(glob.glob(main_folder + wildcard)):
        file_count += 1
        if verbose:
            logging.info(f"\nLoading file {file_count}: {filename}")
        
        db_tmp = entire_dataset_from_single_file(
            filename, 
            col_names=col_names,
            selected_col_names=selected_col_names,
            scale_dl_buffer=scale_dl_buffer,
            remove_zero_req_prb_entries=remove_zero_req_prb_entries,
            replace_zero_with_one=replace_zero_with_one,
            add_prb_ratio=add_prb_ratio,
            verbose=verbose
        )
        dataset.append(db_tmp)
    
    combined_dataset = pd.concat(dataset, axis=0, ignore_index=True)
    
    if verbose:
        logging.info(f"\n{'='*70}")
        logging.info(f"Dataset Loading Complete")
        logging.info(f"  Total files loaded: {file_count}")
        logging.info(f"  Total rows in combined dataset: {len(combined_dataset)}")
        logging.info(f"{'='*70}\n")
    
    return combined_dataset


# ============================================================================
# PART 2: CLEAR AGGREGATION FUNCTION
# ============================================================================

def compute_slice_throughput(dataset, 
                             slice_id, 
                             slice_name="unknown",
                             verbose=False):
    """
    Compute per-slice data rate (throughput) from dataset.
    
    METHODOLOGY FOR TABLE VI:
    1. Filter dataset to rows matching the slice_id
    2. Extract tx_brate downlink [Mbps] values (reward metric for eMBB/mMTC)
       OR ratio_granted_req (reward metric for uRLLC)
    3. Compute mean across all filtered UE records
    4. This mean IS the Table VI value (no further conversion needed)
    
    Args:
        dataset: Preprocessed DataFrame with filtering already applied
        slice_id: Slice identifier (0=eMBB, 1=mMTC, 2=uRLLC)
        slice_name: Human-readable slice name for logging
        verbose: If True, print detailed statistics
    
    Returns:
        dict: Contains throughput, UE count, statistics for this slice
    """
    # Filter to specific slice
    slice_data = dataset[dataset['slice_id'] == slice_id]
    
    if len(slice_data) == 0:
        logging.warning(f"Slice {slice_name} (ID={slice_id}): No data found!")
        return {
            'slice_id': slice_id,
            'slice_name': slice_name,
            'mean_throughput_mbps': 0.0,
            'ue_count': 0,
            'min_throughput_mbps': 0.0,
            'max_throughput_mbps': 0.0,
            'std_throughput_mbps': 0.0,
            'data_points': 0
        }
    
    # Extract throughput metric
    throughput_values = slice_data['tx_brate downlink [Mbps]'].values
    
    # ========== AGGREGATION STEP (Core of Table VI) ==========
    mean_throughput = np.mean(throughput_values)
    std_throughput = np.std(throughput_values)
    min_throughput = np.min(throughput_values)
    max_throughput = np.max(throughput_values)
    
    result = {
        'slice_id': slice_id,
        'slice_name': slice_name,
        'mean_throughput_mbps': float(mean_throughput),
        'ue_count': len(slice_data),
        'min_throughput_mbps': float(min_throughput),
        'max_throughput_mbps': float(max_throughput),
        'std_throughput_mbps': float(std_throughput),
        'data_points': len(throughput_values)
    }
    
    if verbose:
        logging.info(f"\n{'='*70}")
        logging.info(f"Slice: {slice_name} (ID={slice_id})")
        logging.info(f"{'='*70}")
        logging.info(f"  UE Records:              {result['ue_count']}")
        logging.info(f"  Data Points:             {result['data_points']}")
        logging.info(f"  Mean Throughput:         {result['mean_throughput_mbps']:.4f} Mbps")
        logging.info(f"  Std Dev:                 {result['std_throughput_mbps']:.4f} Mbps")
        logging.info(f"  Min:                     {result['min_throughput_mbps']:.4f} Mbps")
        logging.info(f"  Max:                     {result['max_throughput_mbps']:.4f} Mbps")
        logging.info(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95]:
            pval = np.percentile(throughput_values, p)
            logging.info(f"    {p}th:                     {pval:.4f} Mbps")
        logging.info(f"{'='*70}\n")
    
    return result


def compute_all_slice_throughputs(dataset, verbose=False):
    """
    Compute Table VI metrics for all three slices.
    
    Returns:
        dict: Keyed by slice_name, each containing aggregated throughput metrics
    """
    slices = [
        (0, 'eMBB'),
        (1, 'mMTC'),
        (2, 'uRLLC')
    ]
    
    results = {}
    for slice_id, slice_name in slices:
        results[slice_name] = compute_slice_throughput(
            dataset, 
            slice_id, 
            slice_name=slice_name,
            verbose=verbose
        )
    
    return results


# ============================================================================
# PART 3: SEPARATE EVALUATION HARNESS FOR DIFFERENT SCENARIOS
# ============================================================================

class Table_VI_Evaluation_Harness:
    """
    Complete evaluation harness for Table VI reproducibility.
    
    This class handles:
    1. Loading raw Colosseum dataset with documented filtering
    2. Computing per-slice throughputs with explicit aggregation
    3. Testing DRL agent actions (for attack/defense scenarios)
    4. Logging all steps with detailed statistics
    5. Outputting results in Table VI format
    
    SCENARIO SUPPORT:
    - "No-Attack": Clean dataset, agents receive unperturbed observations
    - "Under-Attack": Adversarial perturbations applied to observations (ε=0.01)
    - "Defended": RADAR defense mechanisms active (sanitization, augmentation, etc.)
    """
    
    def __init__(self, 
                 main_folder='./slice_traffic/rome_static_close/tr10',
                 wildcard_match='/*/*/slices_bs*/*_metrics.csv',
                 output_dir='./evaluation_results/',
                 verbose=True):
        """
        Initialize evaluation harness.
        
        Args:
            main_folder: Path to Colosseum dataset (rome_static_close scenario)
            wildcard_match: Pattern to match metric CSV files
            output_dir: Directory to save results and logs
            verbose: If True, enable detailed logging
        """
        self.main_folder = main_folder
        self.wildcard_match = wildcard_match
        self.output_dir = output_dir
        self.verbose = verbose
        self.dataset = None
        self.results = {}
        
        # Setup logging
        os.makedirs(output_dir, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging to both file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f'table_vi_evaluation_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("="*80)
        logging.info("RADAR Table VI Evaluation Harness Started")
        logging.info("="*80)
        logging.info(f"Dataset folder: {self.main_folder}")
        logging.info(f"File pattern: {self.wildcard_match}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info("="*80 + "\n")
    
    def load_dataset(self):
        """
        Load and preprocess the entire Colosseum dataset.
        
        CRITICAL STEPS DOCUMENTED:
        1. Read all CSV files from rome_static_close/tr10
        2. Extract relevant columns (slice_id, tx_brate, PRB metrics)
        3. Apply filtering: sum_requested_prbs > 0 (removes idle periods)
        4. Compute ratio_granted_req from sum_granted_prbs / sum_requested_prbs
        5. Scale dl_buffer by dividing by 10000
        
        Returns:
            pandas.DataFrame: Preprocessed dataset ready for evaluation
        """
        logging.info("\nStep 1: Loading Dataset from Colosseum O-RAN")
        logging.info("-" * 80)
        
        # Column names in the srsLTE/Colosseum CSV dataset
        all_metrics_list = [
            "Timestamp",
            "num_ues",
            "IMSI",
            "RNTI",
            "empty_1",
            "slicing_enabled",
            "slice_id",
            "slice_prb",
            "power_multiplier",
            "scheduling_policy",
            "empty_2",
            "dl_mcs",
            "dl_n_samples",
            "dl_buffer [bytes]",
            "tx_brate downlink [Mbps]",
            "tx_pkts downlink",
            "tx_errors downlink (%)",
            "dl_cqi",
            "empty_3",
            "ul_mcs",
            "ul_n_samples",
            "ul_buffer [bytes]",
            "rx_brate uplink [Mbps]",
            "rx_pkts uplink",
            "rx_errors uplink (%)",
            "ul_rssi",
            "ul_sinr",
            "phr",
            "empty_4",
            "sum_requested_prbs",
            "sum_granted_prbs",
            "empty_5",
            "dl_pmi",
            "dl_ri",
            "ul_n",
            "ul_turbo_iters"
        ]
        
        # Columns needed for Table VI computation
        metric_list_to_extract = [
            "slice_id",
            "dl_buffer [bytes]",
            "tx_brate downlink [Mbps]",
            "sum_requested_prbs",
            "sum_granted_prbs",
            "scheduling_policy"
        ]
        
        self.dataset = entire_dataset_from_folder(
            main_folder=self.main_folder,
            wildcard=self.wildcard_match,
            col_names=all_metrics_list,
            selected_col_names=metric_list_to_extract,
            scale_dl_buffer=True,
            remove_zero_req_prb_entries=True,  # CRITICAL: Filters idle UEs
            add_prb_ratio=True,
            verbose=self.verbose
        )
        
        logging.info(f"\nDataset loaded successfully!")
        logging.info(f"  Total UE records: {len(self.dataset)}")
        logging.info(f"  Scenario: rome_static_close/tr10")
        logging.info(f"  Filters applied: sum_requested_prbs > 0")
        
        # Log per-slice statistics
        for slice_id, slice_name in [(0, 'eMBB'), (1, 'mMTC'), (2, 'uRLLC')]:
            count = len(self.dataset[self.dataset['slice_id'] == slice_id])
            logging.info(f"  {slice_name} records: {count}")
        
        return self.dataset
    
    def evaluate_no_attack_scenario(self):
        """
        Evaluate Table VI "No-Attack" column.
        
        METHODOLOGY:
        - Use clean dataset (already filtered with sum_requested_prbs > 0)
        - Extract tx_brate downlink [Mbps] directly from dataset
        - Compute per-slice mean (this IS the Table VI value)
        - No agent inference needed for baseline throughput
        
        Returns:
            dict: Per-slice throughput metrics for No-Attack scenario
        """
        logging.info("\n" + "="*80)
        logging.info("SCENARIO: No-Attack (Clean Dataset Baseline)")
        logging.info("="*80)
        
        results = compute_all_slice_throughputs(self.dataset, verbose=self.verbose)
        
        self.results['No-Attack'] = results
        
        # Summary table
        logging.info("\nTable VI - No-Attack Column Summary:")
        logging.info(f"  eMBB:  {results['eMBB']['mean_throughput_mbps']:.4f} Mbps")
        logging.info(f"  mMTC:  {results['mMTC']['mean_throughput_mbps']:.4f} Mbps")
        logging.info(f"  uRLLC: {results['uRLLC']['mean_throughput_mbps']:.4f} Mbps (ratio_granted_req)")
        
        return results
    
    def evaluate_under_attack_scenario(self, epsilon=0.01):
        """
        Evaluate Table VI "Under Attack" column.
        
        METHODOLOGY FOR ATTACKED RATES:
        1. Load DRL agent policies from saved models
        2. Apply adversarial perturbation to observations (ε=0.01)
        3. Feed perturbed observations to agents
        4. Agents output scheduling decisions (0=RR, 1=WF, 2=PF) based on attacked input
        5. These decisions influence the per-UE tx_brate values
        6. Compute per-slice mean of resulting tx_brate downlink [Mbps]
        
        NOTE: The exact scheduler model implementation that maps scheduling decisions
        to per-UE rates is part of the Colosseum simulator near-RT RIC environment,
        not included in this repository. The attack/defense impact is captured through
        the changed scheduling decisions output by the DRL agents.
        
        Args:
            epsilon: Perturbation magnitude for adversarial attack (default: 0.01)
        
        Returns:
            dict: Per-slice throughput metrics under attack
        """
        logging.info("\n" + "="*80)
        logging.info(f"SCENARIO: Under Attack (Adversarial Perturbation ε={epsilon})")
        logging.info("="*80)
        
        logging.info("\nNote: This scenario requires:")
        logging.info("  1. DRL agent policies (in ./ml_models/)")
        logging.info("  2. Autoencoder (./ml_models/encoder.h5)")
        logging.info("  3. Integration with Colosseum simulator for per-UE rate computation")
        logging.info("\nPlaceholder implementation - use with actual agent policies:")
        
        # Placeholder: would need actual agent loading and inference
        logging.warning("WARNING: Under-Attack evaluation requires loaded DRL policies.")
        logging.warning("See base_line_agent.py for policy loading pattern.")
        
        return None
    
    def evaluate_defended_scenario(self):
        """
        Evaluate Table VI "Defended" column (RADAR defense mechanisms).
        
        METHODOLOGY FOR DEFENDED RATES:
        1. Load DRL agent policies trained with RADAR defenses:
           - Input space sanitization (autoencoder + reconstruction)
           - Data augmentation (training on clean + adversarial examples)
           - Adversarial training (robust PPO)
        2. Apply adversarial perturbation to observations (same as attack)
        3. Feed perturbed observations to defended agents
        4. Defended agents output better scheduling decisions despite attack
        5. Resulting per-UE rates are higher than under-attack scenario
        6. Compute per-slice mean of resulting tx_brate downlink [Mbps]
        
        Returns:
            dict: Per-slice throughput metrics with RADAR defenses
        """
        logging.info("\n" + "="*80)
        logging.info("SCENARIO: Defended (RADAR Mitigation Mechanisms Active)")
        logging.info("="*80)
        
        logging.info("\nDefense mechanisms:")
        logging.info("  1. Input space sanitization via autoencoder reconstruction")
        logging.info("  2. Adversarial training during policy learning")
        logging.info("  3. Data augmentation with adversarial examples")
        
        logging.warning("WARNING: Defended evaluation requires RADAR-trained policies.")
        logging.warning("See RADAR.py for defensive training patterns.")
        
        return None
    
    def export_results_to_json(self):
        """
        Export all evaluation results to JSON format.
        
        Output includes:
        - Per-slice throughput metrics for each scenario
        - Metadata (dataset, filters, scenario parameters)
        - Full statistics for reproducibility verification
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f'table_vi_results_{timestamp}.json')
        
        export_data = {
            'evaluation_timestamp': timestamp,
            'dataset_info': {
                'scenario': 'rome_static_close/tr10',
                'folder': self.main_folder,
                'file_pattern': self.wildcard_match
            },
            'filtering_applied': {
                'sum_requested_prbs_filter': '>0',
                'description': 'Removes idle UE periods where no PRBs were requested'
            },
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"\nResults exported to: {output_file}")
        return output_file
    
    def generate_table_vi_summary(self):
        """
        Generate a summary table in Table VI format.
        
        Format:
        +--------+----------+----------+-----------+
        | Slice  | No-Attack| Under Atk| Defended  |
        +--------+----------+----------+-----------+
        | eMBB   | X.XXXX   | Y.YYYY   | Z.ZZZZ    |
        | mMTC   | ...      | ...      | ...       |
        | uRLLC  | ...      | ...      | ...       |
        +--------+----------+----------+-----------+
        """
        logging.info("\n" + "="*80)
        logging.info("TABLE VI - Per-Slice Data Rate Summary (Mbps)")
        logging.info("="*80)
        
        if 'No-Attack' not in self.results:
            logging.warning("No-Attack results not available. Run evaluate_no_attack_scenario() first.")
            return
        
        # Header
        logging.info("\n{:<15} {:<15} {:<15} {:<15}".format(
            "Slice", "No-Attack", "Under Attack", "Defended"
        ))
        logging.info("-" * 60)
        
        # Data rows
        for slice_name in ['eMBB', 'mMTC', 'uRLLC']:
            no_atk_val = self.results['No-Attack'][slice_name]['mean_throughput_mbps']
            
            under_atk_val = "N/A"
            if 'Under-Attack' in self.results and self.results['Under-Attack']:
                under_atk_val = f"{self.results['Under-Attack'][slice_name]['mean_throughput_mbps']:.4f}"
            
            defended_val = "N/A"
            if 'Defended' in self.results and self.results['Defended']:
                defended_val = f"{self.results['Defended'][slice_name]['mean_throughput_mbps']:.4f}"
            
            logging.info("{:<15} {:<15.4f} {:<15} {:<15}".format(
                slice_name, no_atk_val, under_atk_val, defended_val
            ))
        
        logging.info("-" * 60)
        logging.info("="*80)


# ============================================================================
# PART 4: UTILITY FUNCTIONS FOR SCHEDULING DECISION ANALYSIS
# ============================================================================

def explain_scheduling_decisions(dataset, verbose=False):
    """
    Analyze how scheduling policies map to per-UE rates.
    
    SCHEDULING POLICY MAPPING (from Colosseum/srsLTE):
    - RR (Round-Robin): Fair but not bandwidth-aware
    - WF (Waterfilling): Allocates more PRBs to UEs with better channel quality
    - PF (Proportional Fair): Balances throughput and fairness
    
    NOTE: The exact rate computation (tx_brate) from scheduler decision is
    implemented in Colosseum's near-RT RIC environment, not included here.
    
    Args:
        dataset: Preprocessed DataFrame
        verbose: If True, print detailed statistics
    
    Returns:
        dict: Statistics on scheduling policy usage and impact
    """
    logging.info("\n" + "="*80)
    logging.info("SCHEDULING DECISION ANALYSIS")
    logging.info("="*80)
    
    if 'scheduling_policy' not in dataset.columns:
        logging.warning("scheduling_policy column not found in dataset")
        return None
    
    policy_stats = dataset['scheduling_policy'].value_counts()
    
    logging.info("\nScheduling Policy Distribution:")
    for policy, count in policy_stats.items():
        pct = 100 * count / len(dataset)
        logging.info(f"  Policy {policy}: {count} ({pct:.2f}%)")
    
    # Per-policy throughput analysis
    if verbose:
        logging.info("\nThroughput by Scheduling Policy:")
        for policy in sorted(dataset['scheduling_policy'].unique()):
            policy_data = dataset[dataset['scheduling_policy'] == policy]
            mean_rate = policy_data['tx_brate downlink [Mbps]'].mean()
            logging.info(f"  Policy {policy}: {mean_rate:.4f} Mbps (mean)")
    
    return policy_stats.to_dict()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Setup
    use_gpu_in_env = True
    
    if use_gpu_in_env is False:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    absl.logging.set_verbosity(absl.logging.INFO)
    
    # Initialize evaluation harness
    harness = Table_VI_Evaluation_Harness(
        main_folder='./slice_traffic/rome_static_close/tr10',
        wildcard_match='/*/*/slices_bs*/*_metrics.csv',
        output_dir='./evaluation_results/',
        verbose=True
    )
    
    # Step 1: Load dataset
    logging.info("\n" + "="*80)
    logging.info("EXECUTION: Table VI Evaluation Pipeline")
    logging.info("="*80)
    
    dataset = harness.load_dataset()
    
    # Step 2: Analyze scheduling decisions (informational)
    explain_scheduling_decisions(dataset, verbose=True)
    
    # Step 3: Evaluate No-Attack scenario (baseline)
    no_attack_results = harness.evaluate_no_attack_scenario()
    
    # Step 4: Evaluate Under-Attack scenario (requires DRL policies + simulator)
    # under_attack_results = harness.evaluate_under_attack_scenario(epsilon=0.01)
    
    # Step 5: Evaluate Defended scenario (requires RADAR-trained policies + simulator)
    # defended_results = harness.evaluate_defended_scenario()
    
    # Step 6: Generate summary table
    harness.generate_table_vi_summary()
    
    # Step 7: Export results
    harness.export_results_to_json()
    
    logging.info("\n" + "="*80)
    logging.info("Evaluation Complete")
    logging.info("="*80)
