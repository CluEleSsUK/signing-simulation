import numpy as np
import random

class CommitteeEnvironment:
    def __init__(self, n, t, c, r, p, initial_balances):
        self.n = n  # Total number of committee members
        self.t = t  # Threshold for action to receive reward
        self.c = c  # Cost of taking action
        self.r = r  # Reward if t or more take action
        self.p = p  # Punishment if less than t take action
        self.balances = np.array(initial_balances, dtype=np.float64)  # Balances of each member

    def step(self, actions):
        num_actions = np.sum(actions)  # Count of members taking the action
        
        rewards = np.zeros(self.n, dtype=np.float64)  # Initialize rewards/punishments
        
        if num_actions >= self.t:
            rewards[:] = self.r  # Everyone gets reward
        else:
            rewards[actions == 0] = -self.p  # Non-actors get punished
        
        # Update balances: action takers lose cost c
        self.balances -= actions * self.c
        
        # Update balances with rewards (converted to float64)
        self.balances += rewards
        
        # Ensure balances don't go below a minimum threshold (e.g., 0)
        self.balances = np.maximum(self.balances, 0)
        
        return self.balances, rewards

class QLearningAgent:
    def __init__(self, idx, n, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.idx = idx  # Index of the agent
        self.n = n  # Number of committee members
        self.actions = actions  # [0, 1]: 0 = no action, 1 = take action
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Initialize Q-table
        self.action_count = 0  # Track the number of times the agent takes the action
    
    def choose_action(self, state):
        # Epsilon-greedy policy for exploration vs exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            state_key = self._state_key(state)
            if state_key not in self.q_table:
                return random.choice(self.actions)
            return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)
        
        # Initialize Q-table entries if they do not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
        
        # Track action if the agent takes it
        if action == 1:
            self.action_count += 1

    def _state_key(self, state):
        # If the state is a scalar (int, float), wrap it in a tuple
        return (state,)  # Ensures state is always a tuple

def run_simulation(n, t, c, r, p, episodes, initial_balances):
    env = CommitteeEnvironment(n, t, c, r, p, initial_balances)
    agents = [QLearningAgent(idx=i, n=n, actions=[0, 1], epsilon=0.2) for i in range(n)]  # Increased exploration
    
    for episode in range(episodes):
        # Get current states (balances) for each agent
        current_states = env.balances.copy()
        
        # Each agent chooses an action
        actions = np.array([int(agent.choose_action(current_states[agent.idx])) for agent in agents])
        
        # Take a step in the environment and observe new balances and rewards
        new_balances, rewards = env.step(actions)
        
        # Each agent learns from the results
        for i, agent in enumerate(agents):
            agent.learn(current_states[i], actions[i], rewards[i], new_balances[i])
        
        # Output some progress every 10000 episodes
        if episode % 10000 == 0:
            print(f"Episode {episode}: Balances = {env.balances}")
    
    # After the simulation, calculate and output the signing frequency
    print("\n--- Signing Frequencies ---")
    for i, agent in enumerate(agents):
        signing_frequency = (agent.action_count / episodes) * 100
        print(f"Agent {i} signed {signing_frequency:.2f}% of the time.")

# Parameters
n = 5  # Number of committee members
t = 3  # Threshold to get reward
c = 2  # Cost of taking action
r = 5  # Reward if at least t members act
p = 10  # Increased punishment
initial_balances = [10] * n  # Starting balance for all members

# Run the simulation for 100000 episodes to calculate signing frequencies
run_simulation(n=n, t=t, c=c, r=r, p=p, episodes=100000, initial_balances=initial_balances)

