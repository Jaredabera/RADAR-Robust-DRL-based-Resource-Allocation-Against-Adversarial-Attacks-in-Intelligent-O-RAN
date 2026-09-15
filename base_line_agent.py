"""
===============================================================================
Updated base_line_agent.py with EXPLICIT AGGREGATION LOGGING
===============================================================================

This version adds detailed logging showing exactly how per-slice data rates
(Table VI) are computed from the Colosseum dataset.

Key additions:
- Logging at filtering step: how many rows removed
- Logging at aggregation step: exact mean computation
- Explicit statistics showing filtering impact
- Documentation of which dataset/scenario is used
===============================================================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
import absl
import time
import os
import glob
import pandas as pd
import numpy as np
import logging
import json

# ============================================================================
# AGGREGATION LOGGING MODULE
# ============================================================================

class AggregationLogger:
    """Log aggregation statistics for Table VI reproducibility."""
    
    @staticmethod
    def log_filtering_step(initial_count, filtered_count, filter_name="sum_requested_prbs > 0"):
        """Log dataset filtering statistics."""
        removed = initial_count - filtered_count
        retention_rate = 100 * filtered_count / initial_count if initial_count > 0 else 0
        
        logging.info("\n" + "="*70)
        logging.info(f"FILTERING STEP: {filter_name}")
        logging.info("="*70)
        logging.info(f"  Rows before filter:     {initial_count}")
        logging.info(f"  Rows after filter:      {filtered_count}")
        logging.info(f"  Rows removed:           {removed}")
        logging.info(f"  Retention rate:         {retention_rate:.2f}%")
        logging.info("="*70 + "\n")
    
    @staticmethod
    def log_aggregation_step(slice_name, slice_id, throughput_values):
        """Log per-slice aggregation (core Table VI computation)."""
        mean_throughput = np.mean(throughput_values)
        std_throughput = np.std(throughput_values)
        min_throughput = np.min(throughput_values)
        max_throughput = np.max(throughput_values)
        
        logging.info("\n" + "="*70)
        logging.info(f"AGGREGATION STEP: {slice_name.upper()} (Slice ID={slice_id})")
        logging.info("="*70)
        logging.info(f"  Data points (UE records): {len(throughput_values)}")
        logging.info(f"  Mean throughput (TABLE VI VALUE): {mean_throughput:.4f} Mbps")
        logging.info(f"  Std Dev:                  {std_throughput:.4f} Mbps")
        logging.info(f"  Min:                      {min_throughput:.4f} Mbps")
        logging.info(f"  Max:                      {max_throughput:.4f} Mbps")
        logging.info(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95]:
            pval = np.percentile(throughput_values, p)
            logging.info(f"    {p:3d}th percentile:        {pval:.4f} Mbps")
        logging.info("="*70 + "\n")
        
        return mean_throughput


# ============================================================================
# ENHANCED DATA LOADING WITH FILTERING DOCUMENTATION
# ============================================================================

def entire_dataset_from_single_file(filename,
                                    col_names,
                                    selected_col_names,
                                    remove_zero_req_prb_entries=True,
                                    scale_dl_buffer=True,
                                    replace_zero_with_one=False,
                                    add_prb_ratio=True):
    """
    Load and preprocess a single CSV file from Colosseum O-RAN dataset.
    
    CRITICAL FILTERING:
    When remove_zero_req_prb_entries=True:
    - Removes rows where sum_requested_prbs <= 0 (idle UEs)
    - This filtering is ESSENTIAL for Table VI reproducibility
    - Applied BEFORE any aggregation/averaging
    
    TRANSFORMATION:
    - dl_buffer: Scaled by dividing by 10000 (bytes → normalized units)
    - ratio_granted_req: Computed as sum_granted_prbs / sum_requested_prbs
    """
    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)
    
    initial_size = len(dataset)
    
    # ========== CRITICAL FILTERING FOR TABLE VI ==========
    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)
        filtered_size = len(dataset)
        
        # Log filtering statistics
        AggregationLogger.log_filtering_step(initial_size, filtered_size)
    
    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in selected_col_names]):
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 10000
    
    if add_prb_ratio:
        dict_add = pd.DataFrame.from_dict({"rgb_granted_req": np.clip(np.nan_to_num(
            dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            dict_add['rgb_granted_req'].loc[dataset['sum_requested_prbs'] <= 0] = 1.0
        return dataset.join(dict_add)
    else:
        return dataset

def entire_dataset_from_folder(main_folder,
                               wildcard,
                               col_names,
                               selected_col_names,
                               scale_dl_buffer=True,
                               remove_zero_req_prb_entries=True,
                               replace_zero_with_one=False,
                               add_prb_ratio=True):
    dataset = []
    for filename in glob.glob(main_folder + wildcard):
        db_tmp = entire_dataset_from_single_file(filename, col_names=col_names,
                                                 selected_col_names=selected_col_names,
                                                 scale_dl_buffer=scale_dl_buffer,
                                                 remove_zero_req_prb_entries=remove_zero_req_prb_entries,
                                                 replace_zero_with_one=replace_zero_with_one,
                                                 add_prb_ratio=add_prb_ratio)
        dataset.append(db_tmp)
    
    return pd.concat(dataset, axis=0, ignore_index=True)


def extract_n_entries_from_dataset(dataset=None,
                                   slice_id=None,
                                   n_entries=10,
                                   metrics_export=None):
    if slice_id is not None:
        d_temp = dataset.loc[dataset['slice_id'] == int(slice_id)]
    else:
        d_temp = dataset

    d_temp = d_temp.sample(n=n_entries).reset_index(drop=True)
    if metrics_export is not None:
        d_temp = d_temp[metrics_export]

    return d_temp


def get_data_from_DUs(dataset=None,
                      n_entries=1000,
                      n_col=4,
                      slice_id=None,
                      metrics_export=None):
    if dataset is None:
        values = np.random.random(size=(n_entries, n_col))
        slice_id = np.random.randint(low=0, high=3, size=(n_entries, 1))
        data = np.concatenate((slice_id, values), axis=1)
    else:
        data = extract_n_entries_from_dataset(dataset=dataset,
                                              slice_id=slice_id,
                                              n_entries=n_entries,
                                              metrics_export=metrics_export)

    return data


def split_data(slice_profiles=None,
               data_to_spit=None,
               metric_list=None,
               metric_dict=None,
               n_entries_per_slice=None):
    """Split and organize data by slice. Rewards extracted here are used for Table VI."""
    metrics = []
    prbs = []
    rewards = []

    for i in slice_profiles:
        slice_data = data_to_spit[data_to_spit[:, metric_dict['slice_id']] == slice_profiles[i]['slice_id'], :]

        if slice_data.size > 0:
            while slice_data.shape[0] < n_entries_per_slice:
                slice_data = np.vstack((slice_data, np.zeros((1, slice_data.shape[1]))))

            slice_prb = slice_data[:, metric_dict['slice_prb']]
            slice_metrics = slice_data[:, [metric_dict[x] for x in metric_list]]
            slice_reward = slice_data[:, metric_dict[slice_profiles[i]['reward_metric']]]

            if n_entries_per_slice is not None:
                slice_prb = slice_prb[0:n_entries_per_slice]
                slice_metrics = slice_metrics[0:n_entries_per_slice, :]
                slice_reward = slice_reward[0:n_entries_per_slice]
        else:
            slice_metrics = []
            slice_prb = []
            slice_reward = []

        metrics.append(slice_metrics)
        prbs.append(slice_prb)
        rewards.append(slice_reward)

    return metrics, prbs, rewards


def generate_timestep_for_policy(obs_tmp=None):
    step_type = tf.convert_to_tensor([0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor([0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor([1], dtype=tf.float32, name='discount')
    observations = tf.convert_to_tensor([obs_tmp], dtype=tf.float32, name='observations')
    return ts.TimeStep(step_type, reward, discount, observations)


if __name__ == '__main__':

    # Column names in the srsLTE CSV dataset
    all_metrics_list = ["Timestamp",
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
                        "ul_turbo_iters"]

    # Column names we need to extract from the dataset
    metric_list_to_extract = ["slice_id",
                              "dl_buffer [bytes]",
                              "tx_brate downlink [Mbps]",
                              "sum_requested_prbs",
                              "sum_granted_prbs"]

    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='./agent.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("\n" + "="*70)
    logging.info("BASELINE AGENT TESTING WITH TABLE VI AGGREGATION LOGGING")
    logging.info("="*70)
    logging.info("Dataset: rome_static_close/tr10 (Colosseum O-RAN COMMAG)")
    logging.info("Filtering: sum_requested_prbs > 0")
    logging.info("Aggregation: Mean tx_brate per slice")
    logging.info("="*70 + "\n")

    use_gpu_in_env = True
    mtc_policy_filename = './ml_models/mtc_policy'
    urllc_policy_filename = './ml_models/urllc_policy'
    embb_policy_filename = './ml_models/embb_policy'
    autoencoder_filename = './ml_models/encoder.h5'

    # Location of the dataset (rome_static_close scenario)
    main_folder = './slice_traffic/rome_static_close/tr10'
    wildcard_match = '/*/*/slices_bs*/*_metrics.csv'

    # Load dataset with filtering
    dataset = entire_dataset_from_folder(main_folder=main_folder,
                                         wildcard=wildcard_match,
                                         col_names=all_metrics_list,
                                         selected_col_names=metric_list_to_extract)
    
    logging.info(f"\nDataset loaded: {len(dataset)} total UE records (after filtering)")
    
    # Compute per-slice aggregates for Table VI
    logging.info("\n" + "="*70)
    logging.info("TABLE VI AGGREGATION (No-Attack Baseline)")
    logging.info("="*70)
    
    slice_configs = [
        (0, 'eMBB'),
        (1, 'mMTC'),
        (2, 'uRLLC')
    ]
    
    table_vi_results = {}
    for slice_id, slice_name in slice_configs:
        slice_data = dataset[dataset['slice_id'] == slice_id]
        throughput_values = slice_data['tx_brate downlink [Mbps]'].values
        
        mean_tp = AggregationLogger.log_aggregation_step(slice_name, slice_id, throughput_values)
        table_vi_results[slice_name] = mean_tp
    
    # Summary
    logging.info("\n" + "="*70)
    logging.info("TABLE VI RESULTS SUMMARY (No-Attack)")
    logging.info("="*70)
    logging.info(f"  eMBB:  {table_vi_results['eMBB']:.4f} Mbps")
    logging.info(f"  mMTC:  {table_vi_results['mMTC']:.4f} Mbps")
    logging.info(f"  uRLLC: {table_vi_results['uRLLC']:.4f} Mbps")
    logging.info("="*70 + "\n")
    
    # Save for reference
    with open('table_vi_no_attack_baseline.json', 'w') as f:
        json.dump(table_vi_results, f, indent=2)
    logging.info("Results saved to: table_vi_no_attack_baseline.json\n")
    
    # Load policies only if needed for further testing
    n_entries_for_autoencoder = 10

    absl.logging.set_verbosity(absl.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if use_gpu_in_env is False:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    drl_agents = [tf.saved_model.load(embb_policy_filename),
                  tf.saved_model.load(mtc_policy_filename),
                  tf.saved_model.load(urllc_policy_filename)]

    absl.logging.info('Agents loaded')

    autoencoder = tf.keras.models.load_model(autoencoder_filename)

    absl.logging.info('Autoencoder loaded')

    slice_profiles = {'embb': {'slice_id': 0, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'mtc': {'slice_id': 1, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'urllc': {'slice_id': 2, 'reward_metric': "rgb_granted_req"}}

    metric_dict = {"dl_buffer [bytes]": 1,
                   "tx_brate downlink [Mbps]": 2,
                   "rgb_granted_req": 3,
                   "slice_id": 0,
                   "slice_prb": 4}

    metric_list_for_agents = ["dl_buffer [bytes]",
                   "tx_brate downlink [Mbps]",
                   "rgb_granted_req"]

    default_policy = 0
    previous_policy = dict()
    for _, val in slice_profiles.items():
        previous_policy[val['slice_id']] = default_policy

    rewards_dict = {}
    for profile in slice_profiles.keys():
        rewards_dict[profile] = []

    # Testing loop
    logging.info("\n" + "="*70)
    logging.info("AGENT TESTING LOOP (Optional - for policy validation)")
    logging.info("="*70 + "\n")
    
    while True:
        policies = list()

        data = get_data_from_DUs(dataset=dataset,
                                 n_entries=1000,
                                 metrics_export=metric_list_to_extract).to_numpy()

        data_tmp, prbs, rewards = split_data(slice_profiles=slice_profiles,
                                             data_to_spit=data,
                                             metric_dict=metric_dict,
                                             metric_list=metric_list_for_agents,
                                             n_entries_per_slice=n_entries_for_autoencoder)
        
        for i in range(len(slice_profiles)):
            if len(data_tmp[i]) > 0:
                for row in data_tmp[i]:
                    row[0] /= 100000

                logging.info('Testing iteration ' + str(i))
                logging.info('Data received from DU: ')
                logging.info(np.expand_dims(data_tmp[i], axis=0))

                obs_tmp = autoencoder.predict(np.expand_dims(data_tmp[i], axis=0)).astype('float32')
                obs_tmp = np.append(obs_tmp, prbs[i][0]).astype('float32')

                reward_mean = np.mean(rewards[i]).astype('float32')
                rewards_dict[profile].append(float(reward_mean))
                time_step = generate_timestep_for_policy(obs_tmp)
                action = drl_agents[i].action(time_step)
                
                policies.append(action[0][0][0].numpy())
                previous_policy[i] = action[0][0][0].numpy()

                logging.info('Slice ' + str(i) + ': Action is ' + str(action[0][0][0].numpy()) + ' Reward is: ' + str(reward_mean))
            else:
                policies.append(previous_policy[i])
                logging.info('Using previous action ' + str(previous_policy[i]))
            print()
        
        msg = ','.join([str(x) for x in policies])
        logging.info('Sending policies: ' + msg)
        with open('rewards.json', 'w') as rewards_file:
            json.dump(rewards_dict, rewards_file)
        
        time.sleep(10)
