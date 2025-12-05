import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")  # Silence pygame warning

# Hyperparameters
EPISODES = 300
MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay
class ExperienceReplay:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
    def add(self, experience):
        self.memory.append(experience)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Train DQN - Clean output every 10 episodes
def train_dqn(episodes):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    memory = ExperienceReplay(MEMORY_SIZE)
    epsilon = EPSILON_START

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            if random.random() <= epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    action = torch.argmax(dqn(state)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            total_reward += reward
            memory.add((state, action, reward, next_state, done))
            state = next_state

            if len(memory.memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states = torch.cat([s for s, a, r, ns, d in batch])
                actions = torch.tensor([a for s, a, r, ns, d in batch])
                rewards = torch.tensor([r for s, a, r, ns, d in batch], dtype=torch.float32)
                next_states = torch.cat([ns for s, a, r, ns, d in batch])
                dones = torch.tensor([d for s, a, r, ns, d in batch], dtype=torch.float32)

                with torch.no_grad():
                    next_q = dqn(next_states).max(1)[0]
                    targets = rewards + GAMMA * next_q * (1 - dones)

                current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(current_q, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # Print only every 10 episodes or the last one
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            print(f"Episode {episode + 1:4d}/{episodes} | Reward: {total_reward:3.0f} | Epsilon: {epsilon:.3f}")

    env.close()
    return dqn

# Play function
def play_cartpole(model):
    env = gym.make('CartPole-v1', render_mode="human")
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0
    done = False
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
    print("Starting training...")
    trained_model = train_dqn(EPISODES)
    print("Training finished! Starting demo...")
    play_cartpole(trained_model)