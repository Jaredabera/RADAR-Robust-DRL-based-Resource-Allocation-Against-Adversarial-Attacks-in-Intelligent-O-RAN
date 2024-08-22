from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import time_step as ts

import absl
import time
import os
import glob

import pandas as pd
import numpy as np
import json
import logging
# Generating dataset from csv file. Returns a Pandas DataFrame
def entire_dataset_from_single_file(filename,
                                    col_names,
                                    selected_col_names,
                                    remove_zero_req_prb_entries=True,
                                    scale_dl_buffer=True,
                                    replace_zero_with_one=False,
                                    add_prb_ratio=True):
    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)

    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)

    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in
                                selected_col_names]):
        # scale the dl_buffer
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 10000

    if add_prb_ratio:
        dict_add = pd.DataFrame.from_dict({"ratio_granted_req": np.clip(np.nan_to_num(
            dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            dict_add['ratio_granted_req'].loc[dataset['sum_requested_prbs'] <= 0] = 1.0
        return dataset.join(dict_add)
    else:
        return dataset

# return all csv files inside a single DataFrame
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


# take n entries from the DataFrame at random
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

# This function is used here to emulate a DU reporting real-time data. Replace this function with your DU
# FOR TESTING PURPOSES ONLY
def get_data_from_DUs(dataset=None,
                      n_entries=1000,
                      n_col=4,
                      slice_id=None,
                      metrics_export=None):

    if dataset is None:  # generate random data in case I do not have a dataset( if it is none first time, can not start from NULL)
        values = np.random.random(size=(n_entries, n_col))
        slice_id = np.random.randint(low=0, high=3, size=(n_entries, 1))
        data = np.concatenate((slice_id, values), axis=1)
    else:
        data = extract_n_entries_from_dataset(dataset=dataset,
                                              slice_id=slice_id,
                                              n_entries=n_entries,
                                              metrics_export=metrics_export)

    return data

# Return lists for metrics, rewards, prbs assigned to each slice.
# Ideally, the list is such that len(list) = num_slices
def split_data(slice_profiles=None,
               data_to_spit=None,
               metric_list=None,
               metric_dict=None,
               n_entries_per_slice=None):
    metrics = []
    prbs = []
    rewards = []

    # ordering here follows slice_profiles
    for i in slice_profiles:

        slice_data = data_to_spit[data_to_spit[:, metric_dict['slice_id']] == slice_profiles[i]['slice_id'], :]

        if slice_data.size > 0:
            # repmat on rows to reach needed dimension in case myfile do not have enough reporting data
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

# Used to generate the input to the DRL agent. It returns a TimeStep that contains (step_type, reward, discount, observations)
def generate_timestep_for_policy(obs_tmp=None):
    step_type = tf.convert_to_tensor(
        [0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [1], dtype=tf.float32, name='discount')
    observations = tf.convert_to_tensor(
        [obs_tmp], dtype=tf.float32, name='observations')
    return ts.TimeStep(step_type, reward, discount, observations)

# Function to calculate the reward
def calculate_reward(observation):
    reward_metric = slice_profiles[i]['reward_metric']  # Use slice_key here
    reward_index = metric_dict[reward_metric]
    return observation[0, reward_index]

def apply_adversarial_attack(data, epsilon=0.01):
    # Apply a noise-based adversarial perturbation
    perturbation = np.random.uniform(-epsilon, epsilon, size=data.shape)
    perturbed_data = data + perturbation
    return perturbed_data
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

    use_gpu_in_env = True
    mtc_policy_filename = './ml_models/mtc_policy'
    urllc_policy_filename = './ml_models/urllc_policy'
    embb_policy_filename = './ml_models/embb_policy'
    autoencoder_filename = './ml_models/encoder.h5'

    # Location of the dataset--> to use (valid in offline testing ONLY)
    main_folder = './slice_traffic/rome_static_close/tr10'
    wildcard_match = '/*/*/slices_bs*/*_metrics.csv'

    # get dataset for testing purposes only.
    # This is used as this code does not run with hardware components.
    # Not needed if getting data from real DUs
    dataset = entire_dataset_from_folder(main_folder=main_folder,
                                         wildcard=wildcard_match,
                                         col_names=all_metrics_list,
                                         selected_col_names=metric_list_to_extract)

    # Input size to the autoencoder for dimentionality reduction
    n_entries_for_autoencoder = 10

    # set logging level + enable TF2 behavior
    absl.logging.set_verbosity(absl.logging.INFO)
    # select which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if use_gpu_in_env is False:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        print("Num GPUs Available outside environments: ", len(gpu_devices))
        # load policy, these are the folder where saved_model.pb is stored
    drl_agents = [tf.saved_model.load(embb_policy_filename),
                  tf.saved_model.load(mtc_policy_filename),
                  tf.saved_model.load(urllc_policy_filename)]

    absl.logging.info('Agents loaded')

    autoencoder = tf.keras.models.load_model(autoencoder_filename)
    
    absl.logging.info('Autoencoder loaded')

    slice_profiles = {'embb': {'slice_id': 0, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'mtc': {'slice_id': 1, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'urllc': {'slice_id': 2, 'reward_metric': "ratio_granted_req"}}

    metric_dict = {"dl_buffer [bytes]": 1,
                   "tx_brate downlink [Mbps]": 2,
                   "ratio_granted_req": 3,
                   "slice_id": 0,
                   "slice_prb": 4}

    default_policy = 0
    previous_policy = dict()
    for _, val in slice_profiles.items():
        previous_policy[val['slice_id']] = default_policy

    previous_metrics = ''
    rewards_dict = {}  # Initialize rewards_dict outside the loop
    for profile in slice_profiles.keys():
        rewards_dict[profile] = []

    # Create an empty list to store the adversarial examples
    adversarial_examples = []
    previous_policy = dict()
    for _, val in slice_profiles.items():
        previous_policy[val['slice_id']] = default_policy

    previous_metrics = ''
    rewards_dict = {}  # Initialize rewards_dict outside the loop
    for profile in slice_profiles.keys():
        rewards_dict[profile] = []

   
    adversarial_examples = []
    while True:
        policies = list()

        # This is where data comes from the DUs.
        # As an example, we extract data from the static dataset.
        # Users may want to interface it with their own DUs
        data = get_data_from_DUs(dataset=dataset,
                                 n_entries=1000,
                                 metrics_export=metric_list_to_extract).to_numpy()

        data_tmp, prbs, rewards = split_data(slice_profiles=slice_profiles,
                                             data_to_spit=data,
                                             metric_dict=metric_dict,
                                             metric_list=["dl_buffer [bytes]", "tx_brate downlink [Mbps]", "ratio_granted_req"],
                                             n_entries_per_slice=n_entries_for_autoencoder)
        epsilon = 0.01  # Initial perturbation magnitude
        max_iterations = 15000  # Maximum number of attack iterations
        threshold_reward_increase = 1.1  # Threshold for overtaking reward
        
        for i in range(len(slice_profiles)):
            if len(data_tmp[i]) > 0:
                for row in data_tmp[i]:
                    row[0] /= 10000 # Convert from bytes to Mbps

                # Apply the adversarial attack to the observations
                perturbed_data = apply_adversarial_attack(data_tmp[i])

                # Append the adversarial example to the list
                adversarial_examples.append(perturbed_data)
                
                logging.info('Testing iteration ' + str(i))
                logging.info('Data received from DU (dl_buffer [bytes], tx_brate downlink [Mbps], ratio_granted_req): ')
                logging.info(np.expand_dims(perturbed_data, axis=0))

                obs_tmp = autoencoder.predict(np.expand_dims(perturbed_data, axis=0)).astype('float32')
                obs_tmp = np.append(obs_tmp, prbs[i][0]).astype('float32')

                reward_mean = np.mean(rewards[i]).astype('float32')
                rewards_dict[profile].append(float(reward_mean))
                time_step = generate_timestep_for_policy(obs_tmp)
                action = drl_agents[i].action(time_step)
                
                # append policies to send and store policy
                policies.append(action[0][0][0].numpy())
                previous_policy[i] = action[0][0][0].numpy()

                logging.info('Slice ' + str(i) + ': Action is ' + str(action[0][0][0].numpy()) + ' Reward is: ' + str(
                    reward_mean))
            else:
                # append previous policy
                policies.append(previous_policy[i])
                logging.info('Using previous action ' + str(previous_policy[i]) + ' for slice profile ' + str(i))
            print()
        # build message to send policies to the DU
        msg = ','.join([str(x) for x in policies])
        logging.info('Sending this message to the DU: ' + msg)
        # with open('rewards.json', 'w') as rewards_file:
        #     json.dump(rewards_dict, rewards_file)
        
        time.sleep(10)
            # Flatten the data_tmp list into a 2-dimensional array
        flattened_data = np.concatenate(data_tmp, axis=0)

        # Create a list to store the column names for the adversarial examples
        adversarial_columns = ["dl_buffer [bytes]", "tx_brate downlink [Mbps]", "ratio_granted_req"]

        # Create a DataFrame from the flattened adversarial examples
        adversarial_df = pd.DataFrame(flattened_data, columns=adversarial_columns)

        # Write the DataFrame to a CSV file
        adversarial_df.to_csv('adversarial_examples.csv', index=False)