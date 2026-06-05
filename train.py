import logging
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
import numpy as np
import random
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

MAX_MEMORY_BYTES = 1024 * 1024
BUFFER_SIZE_LIMIT = 5000

dashboard_state = {
    'episode': 0,
    'reward': 0.0,
    'epsilon': 1.0,
    'loss': 0.0,
    'memory': 0.0,
    'history_reward': [],
    'history_loss': [],
    'history_memory': [],
    'frame': b''
}

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            try:
                with open('dashboard.html', 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"404")
        
        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {
                'episode': dashboard_state['episode'],
                'reward': dashboard_state['reward'],
                'epsilon': dashboard_state['epsilon'],
                'loss': dashboard_state['loss'],
                'memory': dashboard_state['memory'],
                'history_reward': dashboard_state['history_reward'],
                'history_loss': dashboard_state['history_loss'],
                'history_memory': dashboard_state['history_memory']
            }
            self.wfile.write(json.dumps(data).encode('utf-8'))
            
        elif self.path == '/frame':
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(dashboard_state['frame'])
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_dashboard_server():
    server = HTTPServer(('0.0.0.0', 8000), DashboardHandler)
    logging.info("UI: http://localhost:8000")
    server.serve_forever()

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

    def __len__(self):
        return self.size
    
    def get_size_bytes(self):
        return (
            self.states.nbytes + 
            self.actions.nbytes + 
            self.rewards.nbytes + 
            self.next_states.nbytes + 
            self.dones.nbytes
        )

def get_model_size_bytes(model):
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    return size

def check_memory(model, buffer):
    model_size = get_model_size_bytes(model)
    buffer_size = buffer.get_size_bytes()
    total_size = model_size + buffer_size
    if total_size > MAX_MEMORY_BYTES:
        logging.error("OOM")
        raise RuntimeError("OOM")
    return total_size

def train():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    server_thread = threading.Thread(target=start_dashboard_server, daemon=True)
    server_thread.start()

    q_network = QNetwork(state_dim, action_dim)
    q_network.train()
    
    q_network.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(q_network, inplace=True)
    
    target_network = QNetwork(state_dim, action_dim)
    target_network.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(target_network, inplace=True)
    
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(BUFFER_SIZE_LIMIT, state_dim)

    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 1000
    target_update = 10
    num_episodes = 200

    global_step = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        episode_losses = []

        while not done:
            frame = env.render()
            if frame is not None:
                dashboard_state['frame'] = frame.tobytes()

            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * global_step / epsilon_decay)
            global_step += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) > batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = buffer.sample(batch_size)
                
                b_state = torch.FloatTensor(b_state)
                b_action = torch.LongTensor(b_action).unsqueeze(1)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1)
                b_next_state = torch.FloatTensor(b_next_state)
                b_done = torch.FloatTensor(b_done).unsqueeze(1)

                q_values = q_network(b_state).gather(1, b_action)
                with torch.no_grad():
                    next_q_values = target_network(b_next_state).max(1)[0].unsqueeze(1)
                    target_q_values = b_reward + (1 - b_done) * gamma * next_q_values

                loss = F.smooth_l1_loss(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_losses.append(loss.item())

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
        current_memory = check_memory(q_network, buffer)

        dashboard_state['episode'] = episode
        dashboard_state['reward'] = episode_reward
        dashboard_state['epsilon'] = epsilon
        dashboard_state['loss'] = avg_loss
        dashboard_state['memory'] = current_memory / 1024
        
        dashboard_state['history_reward'].append(episode_reward)
        dashboard_state['history_loss'].append(avg_loss)
        dashboard_state['history_memory'].append(current_memory / 1024)

        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        if episode % 10 == 0:
            logging.info(f"E:{episode} R:{episode_reward:.1f} L:{avg_loss:.3f} M:{current_memory/1024:.1f}K")

    env.close()

    q_network.eval()
    quantized_model = torch.ao.quantization.convert(q_network, inplace=False)
    
    torch.save(quantized_model.state_dict(), 'tinyrl_int8.pth')
    logging.info("Saved tinyrl_int8.pth")
    logging.info("Done.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    train()
