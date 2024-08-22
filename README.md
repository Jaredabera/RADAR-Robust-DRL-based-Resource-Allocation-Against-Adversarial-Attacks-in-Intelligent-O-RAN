# RADAR-Robust-DRL-based-Resource-Allocation-Against-Adversarial-Attacks-in-Intelligent-O-RAN
RADAR is a DRL-based resource allocation mechanism for O-RAN, defending against adversarial attacks. It enhances resilience through input space sanitization, augmentation, and adversarial training. RADAR significantly recovers user data rates across eMBB, mMTC, and uRLLC slices.
![defense_resized-111](https://github.com/user-attachments/assets/d1c90f05-9212-43d6-a685-cc4734b722f7)

For detailed information on the *tested adversarial attacks*, please visit the following GitHub repository: Adversarial DRL ORAN:https://github.com/Jaredabera/adversarialdrlORAN.

# Testing the DRL agents

This repository contains the script test_agent_release.py, which is used to test the DRL agents we used in our work. The script executes in three phases:

Phase 1: loading agents and encoder from ml_models;
Phase 2: loading data from the CSV files in the repository;
Phase 3: feeding the DRL agents which compute the best action for the current state. This phase runs in a loop.
All required dependencies are included in the requirements.txt file.

*Remark 1*: anyone interested in feeding real-time data to the DRL agents must implement proper methods to (i) gather data from DUs (i.e., get_data_from_DUs()); (ii) feed it to the DRL agent (i.e., split_data()); and (iii) feed back the output of the DRL agent to the DUs (i.e., send_action_to_DU()).

**Phase 1**
We load the 3 DRL agents and the encoder portion of the autoencoder we used in the experimental section of our work. All models are stored in ml_models and loaded when starting the script. We have one DRL agent (i.e., the trained Proximal Policy Optimization (PPO) policy network) per slice. Rewards vary across the various DRL agents and are set as follows:

eMBB slice: Maximize throughput. This is done by setting the reward equal to tx_brate downlink [Mbps], which represents the downlink throughput in Mbps as measured by srsLTE;
MTC slice: Maximize throughput. This is done by setting the reward equal to tx_brate downlink [Mbps], which represents the downlink throughput in Mbps as measured by srsLTE;
URLLC slice: Minimize latency. This is done by setting the reward equal to ratio_granted_req, which represents the ratio between the number of PRBs allocated by the scheduler and those requested by the UEs. The higher the value, the faster requests are satisfied and traffic experience low latency.
These metrics are reported periodically by DUs and, in our case, are contained in the CSV repository included in this repository.

**Phase 2**
We load the CSV dataset included in the repository. CSV files are loaded into Pandas DataFrame structures, which are used in this case to feed the DRL agents with data. In real-world deployments, data is reported directly from DUs. In this case, and for testing purposes only, we provide functions to emulate such data by extracting it from the dataset we collected.

**Phase 3**
We run a loop that extracts data from the dataset and feeds it to each DRL agent. Data is taken from the dataset at random, grouped according to the slice they belong to, and fed to the corresponding DRL agent, which uses the PPO policy network to compute the best action to maximize the reward.

***Key Features***

- Online adversarial example generation
- PPO-based DRL agents for each network slice (eMBB, MTC, URLLC)
- Dimensionality reduction using autoencoders
- Continuous learning loop for enhanced robustness

**How It Works**
## Dataset structure
- ``slice_mixed``: UEs are randomly distributed across slices
- ``slice_traffic``: UEs are divided per slice based on traffic types:
  	- Slice 0: eMBB UEs
  	- Slice 1: MTC UEs
  	- Slice 2: URLLC UEs
![model_structure-Page-2](https://github.com/user-attachments/assets/6550f734-d10a-45d6-a454-8a761e23b549)

1. *Data Collection*: Simulates data from network slices (dl_buffer, tx_brate, ratio_granted_req). system collects data from DUs (Data Units) using the get_data_from_DUs function. In this case, it's simulating data collection from a static dataset.
2. *Adversarial Attack*: Applies small imperciptle  (ε=0.001) perturbations to create adversarial examples. The perturbed data is stored in the adversarial_examples list.
3. *Preprocessing*: Normalizes data and reduces dimensionality via autoencoder. This compressed representation is used as input for the DRL agent.
4. *DRL Decision-Making*: Feeds processed adversarial examples to slice-specific DRL agents. The agent generates an action (policy) for each slice based on the adversarial input.
The actions determine the resource allocation for each slice.
5. *Reward Calculation*: Computes and logs rewards based on slice-specific metrics. The rewards and actions are logged for each iteration.
6. *Continuous Learning*: Repeats the process, allowing agents to adapt to adversarial inputs. By training on these perturbed inputs, the DRL agent becomes more robust to potential variations or attacks on agent new observations/data.
