This project implements a Q-learning agent to solve a custom 5×5 Grid World environment using reinforcement learning techniques. The agent learns to navigate the environment, avoid obstacles, leverage a special jump, and ultimately reach the terminal state to maximize cumulative rewards.

Environment Configuration

Grid size: 5 × 5 (bounded by borders)

Actions: Four possible moves — North (↑), South (↓), East (→), West (←)

Start state: Cell [2,1] (second row, first column)

Terminal state: Cell [5,5] (reward = +10)

Special jump: From cell [2,4] → [4,4] with a reward of +5

Obstacles (blocked cells):
(3,3), (3,4), (3,5), (4,3)

Rewards:
+10 for reaching the terminal state
+5 for special jump
−1 for every other move

Learning Rule

The state value update follows the standard Q-learning update rule:

V(Sₜ) ← V(Sₜ) + α [ V(Sₜ₊₁) − V(Sₜ) ]

Where:
α = learning rate (0 < α ≤ 1)
V(Sₜ) = current state value
V(Sₜ₊₁) = next state value

Features

Q-table representation of states and actions

Epsilon-greedy exploration to balance exploration and exploitation

Configurable learning rate (α) between 0 and 1

Training for 100 episodes with early stopping when the agent achieves an average cumulative reward > 10 over 30 consecutive episodes

Visualization of state values in the grid world

Tasks Implemented

(a) Explained exploration vs. exploitation
(b) Defined policy, environment, states, and actions
(c) Built Q-table with multiple learning rates
(d) Implemented Q-learning agent with epsilon-greedy strategy
(e) Trained agent under defined stopping criteria
(f) Visualized learned state values in the grid layout

Visualizations

Learned Q-values per state–action pair

Heatmap of state values across the grid

Training performance over episodes (cumulative rewards)
