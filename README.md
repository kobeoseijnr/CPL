# CPL
Contrastive Preference Learning for Trajectory Optimization
# Contrastive Preference Learning for Trajectory Optimization

## Overview
This project explores **Contrastive Preference Learning (CPL)**, an advanced framework for optimizing trajectory rankings in **reinforcement learning (RL)** tasks. Unlike traditional RL methods that rely on explicitly defined reward functions, CPL leverages **human feedback mechanisms**—such as **pairwise comparisons, expert demonstrations, and emergency stop (e-stop) signals**—to construct a robust, preference-based ranking function. This approach enhances agent learning in environments where rewards are sparse, ambiguous, or difficult to specify.

## Key Features
- **Preference-Based Learning:** Uses human feedback to optimize agent behavior.
- **Multi-Modal Feedback Integration:** Incorporates **pairwise comparisons**, **expert demonstrations**, and **e-stop signals** to refine trajectory rankings.
- **Improved Policy Optimization:** Builds upon **maximum entropy RL** and **reward-rational implicit choice models** to enhance learning robustness.
- **Safety-Aware Learning:** Ensures agents avoid unsafe behaviors using **e-stop signals**.
- **Outperforms Traditional RL:** Demonstrates superior accuracy and robustness compared to Q-Learning and other standard RL methods.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install numpy torch matplotlib
```
If using CUDA for GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage
To run the project and generate training trajectories:
```bash
python create_traj.py
```
Ensure the custom environment `custom_env_w.py` is included in the directory.

## File Structure
```
├── create_traj.py     # Main script for trajectory generation
├── custom_env_w.py    # Custom RL environment
├── README.md          # Documentation
```

## Methodology
CPL optimizes trajectory rankings by training a **preference-based ranking function** with a neural network architecture:
- **Input Layer:** Processes trajectory features (e.g., position, velocity, state variables).
- **Hidden Layers:** Fully connected layers with **ReLU activation** for non-linearity.
- **Dropout Regularization:** Improves generalization by preventing overfitting.
- **Output Layer:** Generates a **preference score** for ranking trajectories.

### Feedback Mechanisms
1. **Pairwise Comparisons** - Allows ranking of trajectories based on desirability.
2. **Expert Demonstrations** - Ensures alignment with expert-defined optimal behaviors.
3. **Emergency Stop (e-stop) Signals** - Prevents unsafe trajectories in high-risk environments.

## Experimental Results
CPL was evaluated against Q-Learning using the following metrics:
- **Accuracy:** CPL achieved **95-98% ranking accuracy** across feedback types.
- **Reward Optimization:** Demonstrated superior performance in sparse feedback environments.
- **Safety Compliance:** E-stop signals significantly reduced unsafe trajectories.

## Future Work
- **Integrating CPL with PPO and SAC** for more advanced RL applications.
- **Developing active learning mechanisms** to enhance feedback efficiency.
- **Applying CPL to real-world scenarios**, such as **robotics, autonomous systems, and healthcare**.


