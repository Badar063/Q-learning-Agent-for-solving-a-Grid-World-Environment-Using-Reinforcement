#Here, we import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start_state = (2, 1)  #2 is Row and Colon is 1
        self.terminal_state = (5, 5)
        self.special_jump = {(2, 4): (4, 4)}  #reward is +5
        self.obstacles = {(3, 3), (3, 4), (3, 5), (4, 3)}  #obstacles

        self.actions = [1, 2, 3, 4]  #1 for North/2 for South/3 for Eas/4 for West
        self.action_names = {1: 'North', 2: 'South', 3: 'East', 4: 'West'}
        self.action_deltas = {
            1: (-1, 0),  # North
            2: (1, 0),  # South
            3: (0, 1),  # East
            4: (0, -1)  # West
        }

    def is_valid_state(self, state):
        row, col = state
        return (1 <= row <= self.rows and
                1 <= col <= self.cols and
                state not in self.obstacles)

    def get_next_state(self, state, action):
        row, col = state
        if state in self.special_jump:  # First Jump
            next_state = self.special_jump[state]
            return next_state, 5, False

        if state == self.terminal_state:  # Checking terminal state
            return state, 0, True  # Zero Reward
        delta_row, delta_col = self.action_deltas[action]  # Calculate movement
        next_state = (row + delta_row, col + delta_col)

        if not self.is_valid_state(next_state):  # Check if valid
            next_state = state  # Stay in place if invalid
        if next_state == self.terminal_state:  # Here calculating reward
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return next_state, reward, done

    def reset(self):  # Resetting environment
        return self.start_state

    def render(self, state=None):  # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.rows + 1):
            ax.axhline(i, color='gray', linewidth=2, alpha=0.7)
        for j in range(self.cols + 1):
            ax.axvline(j, color='gray', linewidth=2, alpha=0.7)
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        for obstacle in self.obstacles:
            row, col = obstacle
            rect = patches.Rectangle((col - 1, self.rows - row), 1, 1,
                                     facecolor='#2c3e50', alpha=0.9,
                                     edgecolor='#34495e', linewidth=3,
                                     hatch='//')
            ax.add_patch(rect)
            ax.text(col - 0.5, self.rows - row + 0.5, 'X',
                    fontsize=20, ha='center', va='center', weight='bold', color='white')

        for from_state, to_state in self.special_jump.items():
            from_row, from_col = from_state
            to_row, to_col = to_state
            ax.annotate('', xy=(to_col - 0.5, self.rows - to_row + 0.5),
                        xytext=(from_col - 0.5, self.rows - from_row + 0.5),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                        linewidth=3, alpha=0.8))
            ax.text(from_col - 0.5, self.rows - from_row + 0.8, '+5',
                    fontsize=14, ha='center', color='#e74c3c',
                    weight='bold', bbox=dict(boxstyle="round,pad=0.3",
                                             facecolor='#f8f9fa',
                                             edgecolor='#e74c3c', alpha=0.9))
        term_row, term_col = self.terminal_state
        rect = patches.Rectangle((term_col - 1, self.rows - term_row), 1, 1,
                                 facecolor='#3498db', alpha=0.8,
                                 edgecolor='#2980b9', linewidth=3)
        ax.add_patch(rect)
        ax.text(term_col - 0.5, self.rows - term_row + 0.5, 'Terminal\n+10',
                ha='center', va='center', color='white',
                fontsize=16, weight='bold')
        start_row, start_col = self.start_state
        rect = patches.Rectangle((start_col - 1, self.rows - start_row), 1, 1,
                                 facecolor='#27ae60', alpha=0.8,
                                 edgecolor='#229954', linewidth=3)
        ax.add_patch(rect)
        ax.text(start_col - 0.5, self.rows - start_row + 0.5, 'Start',
                ha='center', va='center', color='white',
                fontsize=14, weight='bold')
        if state:
            row, col = state
            circle = patches.Circle((col - 0.5, self.rows - row + 0.5), 0.3,
                                    facecolor='#f1c40f', edgecolor='#f39c12',
                                    linewidth=3, alpha=0.9)
            ax.add_patch(circle)
            ax.text(col - 0.5, self.rows - row + 0.5, 'Agent',
                    fontsize=10, ha='center', va='center')
        # Plotting Part
        ax.set_xticks(np.arange(0.5, self.cols + 0.5))
        ax.set_yticks(np.arange(0.5, self.rows + 0.5))
        ax.set_xticklabels(range(1, self.cols + 1))
        ax.set_yticklabels(range(self.rows, 0, -1))
        ax.set_xlabel('Column', fontsize=14, weight='bold')
        ax.set_ylabel('Row', fontsize=14, weight='bold')
        ax.set_title('Grid World Environment - Reinforcement Learning',
                     fontsize=16, weight='bold', pad=20)

        for i in range(1, self.rows + 1):  # Add grid coordinates
            for j in range(1, self.cols + 1):
                if (i, j) not in self.obstacles and (i, j) != self.terminal_state and (i, j) != self.start_state:
                    ax.text(j - 0.5, self.rows - i + 0.5, f'({i},{j})',
                            ha='center', va='center', fontsize=9,
                            color='gray', alpha=0.7)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class QLearningAgent:
    def __init__(self, env, learning_rate=1.0, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.q_table = np.zeros(
            (env.rows + 1, env.cols + 1, len(env.actions)))  # Initialize Q-table: 5x5 grid x 4 actions

        self.episode_rewards = []  # Here, it tracks learning history
        self.episode_lengths = []

    def choose_action(self, state, training=True):
        row, col = state

        if training and np.random.random() < self.epsilon:
            valid_actions = []  # Exploration(random)
            for action in self.env.actions:
                next_state, _, _ = self.env.get_next_state(state, action)
                if next_state != state:  # Only consider moves that actually change state
                    valid_actions.append(action)

            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.choice(self.env.actions)
        else:
            return self.env.actions[np.argmax(self.q_table[row, col])]  # Exploration(best action)

    def update_q_value(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        action_idx = self.env.actions.index(action)

        current_q = self.q_table[row, col, action_idx]

        if next_state == self.env.terminal_state:  # Here, it gets max Qvalue for next state
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_row, next_col])

        new_q = current_q + self.learning_rate * (  # Q-learning update rule
                reward + self.discount_factor * next_max_q - current_q
        )

        self.q_table[row, col, action_idx] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_state_values(self):
        state_values = np.zeros((self.env.rows + 1, self.env.cols + 1))
        for row in range(1, self.env.rows + 1):
            for col in range(1, self.env.cols + 1):
                if (row, col) in self.env.obstacles:
                    state_values[row, col] = np.nan  # Mark obstacles as nan
                else:
                    state_values[row, col] = np.max(self.q_table[row, col])
        return state_values

    def get_policy(self):
        policy = {}
        for row in range(1, self.env.rows + 1):
            for col in range(1, self.env.cols + 1):
                if (row, col) not in self.env.obstacles:
                    best_action_idx = np.argmax(self.q_table[row, col])
                    policy[(row, col)] = self.env.actions[best_action_idx]
        return policy


def train_agent(env, agent, episodes=100, convergence_threshold=10, window_size=30):
    print(f"Starting training for {episodes} episodes...")
    print(f"Convergence threshold: average reward > {convergence_threshold} over {window_size} episodes")
    print(f"Learning rate: {agent.learning_rate}, Discount: {agent.discount_factor}")
    print("-" * 70)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:  # For stopping infinite loops
            action = agent.choose_action(state)
            next_state, reward, done = env.get_next_state(state, action)

            agent.update_q_value(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        agent.decay_epsilon()

        if len(agent.episode_rewards) >= window_size:  # Check for convergence
            recent_avg = np.mean(agent.episode_rewards[-window_size:])
            if recent_avg > convergence_threshold:
                print(f"CONVERGED at episode {episode + 1}!")
                print(f"   Average reward over last {window_size} episodes: {recent_avg:.2f}")
                break

        if (episode + 1) % 10 == 0 or episode == 0:
            recent_avg = np.mean(agent.episode_rewards[-10:]) if len(agent.episode_rewards) >= 10 else total_reward
            status = "***" if total_reward >= 8 else "+++" if total_reward >= 5 else "---"
            print(f"{status} Episode {episode + 1:3d} | Reward: {total_reward:4.1f} | "
                  f"Steps: {steps:3d} | ε: {agent.epsilon:.3f} | "  # Print progress
                  f"Avg: {recent_avg:.2f}")

    return agent.episode_rewards


def visualize_training_progress(rewards, learning_rate):
    plt.figure(figsize=(15, 6))

    moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')  # Create moving average with proper alignment
    moving_avg_padded = np.full(len(rewards),
                                np.nan)  # Pad the moving average with nan values to match the length of rewards
    moving_avg_padded[9:] = moving_avg

    # Here, I create dataframe for seaborn
    df = pd.DataFrame({
        'Episode': range(1, len(rewards) + 1),
        'Reward': rewards,
        'Moving_Avg_10': moving_avg_padded
    })

    plt.subplot(1, 2, 1)  # Line plot with seaborn
    sns.lineplot(data=df, x='Episode', y='Reward', alpha=0.6, label='Episode Reward', linewidth=1)
    sns.lineplot(data=df.dropna(subset=['Moving_Avg_10']), x='Episode', y='Moving_Avg_10',
                 label='Moving Average (10)', color='red', linewidth=3)

    plt.title(f'Training Progress (LR: {learning_rate})', fontsize=14, weight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)  # Distribution with seaborn
    sns.histplot(rewards, bins=20, kde=True, color='skyblue',
                 edgecolor='navy', alpha=0.7)
    plt.title('Reward Distribution', fontsize=14, weight='bold')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_state_values_seaborn(agent):
    state_values = agent.get_state_values()

    df_values = pd.DataFrame(state_values[1:, 1:],
                             # Create dataFfame for seaborn and reverse the index to flip the heatmap vertically
                             index=range(1, 6),  # Reversed index
                             columns=range(1, 6))

    mask = np.zeros_like(df_values,
                         dtype=bool)  # Create mask for obstacles and reverse the index to match the flipped heatmap
    mask_df = pd.DataFrame(mask, index=df_values.index, columns=df_values.columns)
    for obstacle in agent.env.obstacles:
        row, col = obstacle
        mask_df.loc[row, col] = True  # Adjust for 1-indexed coordinates and the flipped index

    mask = mask_df.values  # Convert back to numpy array for heatmap

    plt.figure(figsize=(12, 10))

    # Create heatmap
    cmap = sns.color_palette("viridis", as_cmap=True)
    data_min = np.nanmin(df_values.values) if not np.isnan(np.nanmin(df_values.values)) else 0
    data_max = np.nanmax(df_values.values) if not np.isnan(np.nanmax(df_values.values)) else 0

    vmin = min(data_min, 0) - abs(data_min * 0.1)  # It will extend range slightly below 0
    vmax = max(data_max, 0) + abs(data_max * 0.1)  # otherwise extend range slightly above 0

    if abs(vmax - vmin) < 1 and (
            data_min != 0 or data_max != 0):  # ensure a minimum range if all values are zero or very close
        range_mid = (vmin + vmax) / 2
        vmin = range_mid - 0.5
        vmax = range_mid + 0.5
    elif abs(vmax - vmin) < 1 and data_min == 0 and data_max == 0:
        vmin = -0.5
        vmax = 0.5

    ax = sns.heatmap(df_values, annot=True, fmt='.1f', cmap=cmap,
                     mask=mask, cbar_kws={'label': 'State Value'},
                     square=True, linewidths=2, linecolor='white',
                     annot_kws={'size': 12, 'weight': 'bold'},
                     vmin=vmin, vmax=vmax)  # Explicitly set vmin and vmax

    # Add special annotations
    start_row, start_col = agent.env.start_state
    term_row, term_col = agent.env.terminal_state

    # For Start
    ax.add_patch(plt.Rectangle((start_col - 1, start_row - 1), 1, 1,  #
                               fill=False, edgecolor='green', linewidth=4))
    ax.text(start_col - 0.5, start_row - 0.5, 'Start',
            ha='center', va='center', fontsize=18, weight='bold', color='white')

    # For TErminal
    ax.add_patch(plt.Rectangle((term_col - 1, term_row - 1), 1, 1,
                               fill=False, edgecolor='blue', linewidth=4))
    ax.text(term_col - 0.5, term_row - 0.5, 'Terminal',
            ha='center', va='center', fontsize=18, weight='bold', color='white')

    # For Special Jump
    for from_state in agent.env.special_jump:
        from_row, from_col = from_state
        ax.text(from_col - 0.5, from_row - 0.5, 'Jump\n+5',
                ha='center', va='center', fontsize=18, weight='bold', color='white')

    # For obstacles
    for obstacle in agent.env.obstacles:
        row, col = obstacle
        ax.text(col - 0.5, row - 0.5, 'X',
                ha='center', va='center', fontsize=20, weight='bold', color='red')

    plt.title('State Values Heatmap - Q-learning Results',
              fontsize=16, weight='bold', pad=20)
    plt.xlabel('Column', fontsize=12, weight='bold')
    plt.ylabel('Row', fontsize=12, weight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # Set y-axis tick labels to match the flipped rows
    ax.set_yticklabels(range(1, agent.env.rows + 1))
    plt.tight_layout()
    plt.show()


def visualize_optimal_policy(agent):
    policy = agent.get_policy()
    action_symbols = {1: 'N', 2: 'S', 3: 'E', 4: 'W'}
    action_colors = {1: '#e74c3c', 2: '#3498db', 3: '#27ae60', 4: '#f39c12'}

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create grid background
    for i in range(agent.env.rows + 1):
        ax.axhline(i, color='gray', linewidth=2, alpha=0.3)
    for j in range(agent.env.cols + 1):
        ax.axvline(j, color='gray', linewidth=2, alpha=0.3)

    # Fill obstacles
    for obstacle in agent.env.obstacles:
        row, col = obstacle
        ax.add_patch(plt.Rectangle((col - 1, 5 - row), 1, 1,
                                   facecolor='#2c3e50', alpha=0.8,
                                   edgecolor='#34495e', linewidth=3))
        ax.text(col - 0.5, 5 - row + 0.5, 'X',
                fontsize=20, ha='center', va='center', weight='bold', color='white')

    for (row, col), action in policy.items():  # Policy directions
        symbol = action_symbols[action]
        color = action_colors[action]
        ax.text(col - 0.5, 5 - row + 0.5, symbol,
                ha='center', va='center', fontsize=18,
                color=color, weight='bold',
                bbox=dict(boxstyle="circle,pad=0.3",
                          facecolor='white',
                          edgecolor=color, linewidth=2))

    # Mark special states
    start_row, start_col = agent.env.start_state
    term_row, term_col = agent.env.terminal_state

    ax.add_patch(plt.Rectangle((start_col - 1, 5 - start_row), 1, 1,
                               fill=False, edgecolor='#27ae60', linewidth=4))
    ax.text(start_col - 0.5, 5 - start_row + 0.5, 'Start',
            ha='center', va='center', fontsize=12, weight='bold', color='#27ae60')

    ax.add_patch(plt.Rectangle((term_col - 1, 5 - term_row), 1, 1,
                               fill=False, edgecolor='#3498db', linewidth=4))
    ax.text(term_col - 0.5, 5 - term_row + 0.5, 'Terminal',
            ha='center', va='center', fontsize=12, weight='bold', color='#3498db')

    # Mark special jump
    for from_state in agent.env.special_jump:
        from_row, from_col = from_state
        ax.text(from_col - 0.5, 5 - from_row + 0.5, 'Jump\n+5',
                ha='center', va='center', fontsize=10, weight='bold', color='#e74c3c')

    # Mark obstacles with X
    for obstacle in agent.env.obstacles:
        row, col = obstacle
        ax.text(col - 0.5, 5 - row + 0.5, 'X',
                ha='center', va='center', fontsize=15, weight='bold')

    ax.set_xlim(0, agent.env.cols)
    ax.set_ylim(0, agent.env.rows)
    ax.set_xticks(range(1, agent.env.cols + 1))
    ax.set_yticks(range(1, agent.env.rows + 1))
    ax.set_xticklabels(range(1, agent.env.cols + 1))
    ax.set_yticklabels(range(agent.env.rows, 0, -1))
    ax.set_xlabel('Column', fontsize=14, weight='bold')
    ax.set_ylabel('Row', fontsize=14, weight='bold')
    ax.set_title('Optimal Policy - Learned Strategy',
                 fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Initialize environment
    env = GridWorld()
    print("Ulster University")
    print("Name: Badar Ul Islam")
    print("Student ID: 20036990")
    print("Course: MSc in Computer Science and Technolgy")
    print("Module: Deep Learning and Its Application")
    print("Professors: Dr Muhammad Akram/ Dr Okunola Adebola Orogun")
    print("=" * 50)

    print("Grid World Environment Analysis:")
    print("=" * 50)
    print(f"📏 Size: {env.rows}x{env.cols}")
    print(f"📍 Start state: {env.start_state}")
    print(f"🎯 Terminal state: {env.terminal_state} (+10 reward)")
    print(f"✨ Special jump: {list(env.special_jump.keys())[0]} -> {list(env.special_jump.values())[0]} (+5 reward)")
    print(f"🚫 Obstacles: {sorted(env.obstacles)}")
    print()

    # Environment visualization
    env.render()

    learning_rates = [1.0, 0.8, 0.5, 0.3, 0.1]  # learning rates, we use

    best_agent = None
    best_avg_reward = -np.inf
    best_lr = None

    for lr in learning_rates:
        print(f"\n{'=' * 70}")
        print(f"Training with learning rate: {lr}")
        print(f"{'=' * 70}")

        # Create and train agent
        agent = QLearningAgent(env, learning_rate=lr, epsilon=0.3, discount_factor=0.9)
        rewards = train_agent(env, agent, episodes=100)

        final_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)  # Evaluate performance
        print(f"Final average reward (last 10 episodes): {final_avg:.2f}")

        if final_avg > best_avg_reward:
            best_avg_reward = final_avg
            best_agent = agent
            best_lr = lr

        visualize_training_progress(rewards, lr)  # Training progress for each learning rate

    if best_agent:  # Results with best agent
        print(f"\n{'*' * 30}")
        print(f"🏆 BEST PERFORMANCE: Learning rate = {best_lr}")
        print(f"💰 Average reward: {best_avg_reward:.2f}")
        print(f"{'*' * 30}")

        # here, visualizing results
        visualize_state_values_seaborn(best_agent)
        visualize_optimal_policy(best_agent)

        # Q-table analysis
        print("\nFinal Q-values Analysis for Key States:")
        print("=" * 60)
        print("State\t\tNorth\tSouth\tEast\tWest\tBest Action")
        print("-" * 60)

        key_states = [(2, 1), (2, 4), (4, 4), (5, 4), (5, 5)]
        action_names = {1: 'North', 2: 'South', 3: 'East', 4: 'West'}

        for state in key_states:
            row, col = state
            q_values = best_agent.q_table[row, col]
            best_action_idx = np.argmax(q_values)
            best_action = action_names[best_agent.env.actions[best_action_idx]]

            print(f"{state}\t{q_values[0]:6.2f}\t{q_values[1]:6.2f}\t"
                  f"{q_values[2]:6.2f}\t{q_values[3]:6.2f}\t{best_action}")

    return best_agent  # it will return best agent


if __name__ == "__main__":
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    sns.set_palette("husl")

    # Main program
    best_agent = main()