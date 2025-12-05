import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Hyperparameters
EPISODES = 300
GAMMA = 0.99
LEARNING_RATE = 0.001
PRINT_EVERY = 10

# A2C Network (Actor + Critic in one model)
class A2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head (policy: probabilities over actions)
        self.actor = nn.Linear(128, action_size)

        # Critic head (value function: V(s))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_probs = F.softmax(self.actor(x), dim=-1)   # π(a|s)
        state_value = self.critic(x)                      # V(s)

        return action_probs, state_value

# Train A2C
def train_a2c(episodes):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = A2C(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting A2C training...")

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        total_reward = 0

        log_probs = []
        values = []
        rewards = []
        masks = []  # 1 if not done, 0 if done

        while not done:
            action_probs, state_value = model(state)
            value = state_value.item()
            probs = action_probs.squeeze(0)

            # Sample action from policy
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Store transition
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(torch.tensor([reward]))
            masks.append(torch.tensor([1 - done]))  # 1 if still alive, 0 if done

            state = torch.FloatTensor(next_state).unsqueeze(0)
            total_reward += reward

        # Compute returns and advantages
        R = 0
        returns = []
        advantages = []
        saved_values = [v.item() for v in values]

        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Normalize advantages
        for i in range(len(rewards)):
            advantage = returns[i] - saved_values[i]
            advantages.append(advantage)

        advantages = torch.tensor(advantages)
        if advantages.std() != 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()
        returns = returns

        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss  # 0.5 is common weight for critic

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (episode + 1) % PRINT_EVERY == 0 or episode == episodes - 1:
            print(f"Episode {episode + 1:4d}/{episodes} | Reward: {total_reward:3.0f}")

    env.close()
    return model

# Play function (same as before)
def play_cartpole(model):
    env = gym.make('CartPole-v1', render_mode="human")
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0
    done = False

    print("Playing trained A2C agent...")

    while not done:
        env.render()
        with torch.no_grad():
            action_probs, _ = model(state)
            action = torch.argmax(action_probs, dim=-1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = torch.FloatTensor(next_state).unsqueeze(0)
        total_reward += reward

    print(f"Game Over! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    trained_model = train_a2c(EPISODES)
    play_cartpole(trained_model)