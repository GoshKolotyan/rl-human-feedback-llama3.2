---

# human-feedback-llama2

Fine-tunes **Llama 3.2 3B** using **RLHF** for production chatbots.

---

## Dataset

For dataset it’s chosen:
**Dahoas/rm-static** (available on GitHub)

---

## Training Phases

**Phase 1: Supervised Fine-Tuning (SFT)**

* Fine-tune base Llama 3.2 on high-quality instruction-following data
* Use the Dahoas/rm-static dataset's chosen responses

**Phase 2: Reward Model Training**

* Train a reward model to predict human preferences
* Use comparison pairs from the dataset (chosen vs rejected)

**Phase 3: Reinforcement Learning (PPO)**

* Use PPO algorithm to optimize policy using the reward model
* Fine-tune the SFT model with RL

---

## 🔄 Comparing Model Outputs (SFT)

📝 **Prompt 1:**
*Explain what attention mechanism is.*

---

### 🔵 BASE MODEL (Llama-3.2-3B — No Fine-tuning)

---

What is the difference between attention and RNN?
Attention is a mechanism that can be used with an RNN or other type of model. It is a way to focus on certain parts of the input sequence, while ignoring other parts.

What is the difference between attention and GRU?
GRU is a type of recurrent neural network (RNN) that can be used for sequence modeling. It is a type of RNN that uses a gated recurrent unit (GRU) to update its internal state. Attention is a mechanism that can be used with an RNN or other type of model. It is a way to focus on certain parts of the input sequence, while ignoring other parts.

What is the difference between attention and LSTM?
LSTM is...

---

### 🟢 FINE-TUNED SFT MODEL (After 2 Epochs of Training)

---

Attention mechanisms are neural networks designed to focus on specific parts of the input data while ignoring others.
They are used in deep learning architectures—such as Transformers—to allow the model to assign different importance to different tokens.
This selective focus helps the network capture dependencies more efficiently across long sequences.

---
