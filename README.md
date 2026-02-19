# Chef's Hat DQN Agent

## Assigned Variant
This implementation focuses on the **DQN (Deep Q-Network)** variant for reinforcement learning in the Chef's Hat card game environment. It includes enhancements such as Double-DQN, Dueling Networks, and Prioritized Experience Replay (PER).

## How to Run the Code
1. **Install Dependencies**:
   ```bash
   pip install chefshatgym torch numpy matplotlib
   ```

2. **Run Training**:
   ```bash
   python chefshat_dqn_agent.py --mode train --games 500
   ```

3. **Run Evaluation**:
   ```bash
   python chefshat_dqn_agent.py --mode eval --games 100
   ```

4. **Run Hyperparameter Experiment**:
   ```bash
   python chefshat_dqn_agent.py --mode experiment
   ```

5. **Run Simulation (Standalone)**:
   ```bash
   python chefshat_dqn_agent.py --mode simulate
   ```

## Experiments Conducted
- **Training**: The agent was trained for 500 games using the ChefsHatGYM environment.
- **Evaluation**: The trained model was evaluated over 100 games to measure performance.
- **Hyperparameter Experiments**: Multiple configurations of learning rate and epsilon decay were tested to identify the best-performing setup.
- **Simulation**: A standalone simulation was run to validate the code logic without requiring the ChefsHatGYM environment.

## Results Interpretation
- **Training Metrics**: The `train_training_metrics.png` file visualizes the learning curve, showing the agent's performance improvement over time.
- **Evaluation Metrics**: The `experiment_summary.json` file summarizes the win rates and other metrics for the trained agent.
- **Hyperparameter Results**: The `exp_experiment_results.png` file compares the performance of different hyperparameter configurations.
- **Simulation Results**: The `sim_training_metrics.png` and `sim_exp_experiment_results.png` files provide insights into the agent's behavior in standalone simulations.

These results demonstrate the effectiveness of the DQN agent in learning optimal strategies for the Chef's Hat card game.