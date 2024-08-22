# RADAR-Robust-DRL-based-Resource-Allocation-Against-Adversarial-Attacks-in-Intelligent-O-RAN
RADAR is a DRL-based resource allocation mechanism for O-RAN, defending against adversarial attacks. It enhances resilience through input space sanitization, augmentation, and adversarial training. RADAR significantly recovers user data rates across eMBB, mMTC, and uRLLC slices.
![defense_resized-1](https://github.com/user-attachments/assets/7f47d301-26e8-4ab3-a2b2-57825f60a185)
**Key Features**

Online adversarial example generation
PPO-based DRL agents for each network slice (eMBB, MTC, URLLC)
Dimensionality reduction using autoencoders
Continuous learning loop for enhanced robustness

**How It Works**

1. *Data Collection*: Simulates data from network slices (dl_buffer, tx_brate, ratio_granted_req). system collects data from DUs (Data Units) using the get_data_from_DUs function. In this case, it's simulating data collection from a static dataset.
2. *Adversarial Attack*: Applies small perturbations to create adversarial examples.
3. *Preprocessing*: Normalizes data and reduces dimensionality via autoencoder.
4. *DRL Decision-Making*: Feeds processed adversarial examples to slice-specific DRL agents.
5. *Reward Calculation*: Computes and logs rewards based on slice-specific metrics.
6. *Continuous Learning*: Repeats the process, allowing agents to adapt to adversarial inputs.
