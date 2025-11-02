# human-feedback-llama2
Fine-tunes Llama 2 using RLHF for production chatbots


# Dataset 
For dataset it choosen 
Dahoas/rm-static it is availabe on github


**Phase 1: Supervised Fine-Tuning (SFT)**
    * Fine-tune base Llama 3.2 on high-quality instruction-following data
    * Use the Dahoas/rm-static dataset's chosen responses

**Phase 2: Reward Model Training**
    * Train a reward model to predict human preferences
    * Use comparison pairs from the dataset (chosen vs rejected)

**Phase 3: Reinforcement Learning (PPO)**
    * Use PPO algorithm to optimize policy using the reward model
    * Fine-tune the SFT model with RL