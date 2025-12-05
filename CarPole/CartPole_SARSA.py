import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Hyperparameters - same as before
EPISODES = 300
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
PRINT_EVERY = 10

# SARSA Network (Q-value estimator)
class SARSA(nn.Module):
    def __init__(self, state_size, action_size):
        super(SARSA, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Returns Q(s,a) for all actions

# Epsilon-greedy action selection (shared helper)
def select_action(model, state, action_size, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()

# Train SARSA (on-policy!)
def train_sarsa(episodes):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = SARSA(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START

    print("Starting SARSA training...")

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False

        # Choose first action A from current state S using epsilon-greedy
        action = select_action(model, state, action_size, epsilon)

        while not done:
            # Take action A, observe R, S', A' (next action from policy)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Choose next action A' from S' using the same policy (on-policy!)
            next_action = select_action(model, next_state_tensor, action_size, epsilon)

            # SARSA update: Q(S,A) ← Q(S,A) + α [R + γ Q(S',A') - Q(S,A)]
            q_values = model(state)
            next_q_values = model(next_state_tensor)

            target = reward + GAMMA * next_q_values[0, next_action] * (not done)
            target_q = q_values.clone()
            target_q[0, action] = target

            # Update
            optimizer.zero_grad()
            loss = F.mse_loss(q_values, target_q)
            loss.backward()
            optimizer.step()

            # Transition
            state = next_state_tensor
            action = next_action
            total_reward += reward

        # Decay epsilon
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # Print every 10 episodes
        if (episode + 1) % PRINT_EVERY == 0 or episode == episodes - 1:
            print(f"Episode {episode + 1:4d}/{episodes} | Reward: {total_reward:3.0f} | Epsilon: {epsilon:.3f}")

    env.close()
    return model

# Play function - unchanged
def play_cartpole(model):
    env = gym.make('CartPole-v1', render_mode="human")
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0
    done = False

    print("Playing trained SARSA agent...")

    while not done:
        env.render()
        with torch.no_grad():
            action = torch.argmax(model(state)).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.FloatTensor(next_state).unsqueeze(0)
        total_reward += reward

    print(f"Game Over! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    trained_model = train_sarsa(EPISODES)
    play_cartpole(trained_model)