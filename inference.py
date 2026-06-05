import logging
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

dashboard_state = {
    "episode": 0,
    "reward": 0.0,
    "epsilon": 0.0,
    "loss": 0.0,
    "memory": 0.0,
    "history_reward": [],
    "history_loss": [],
    "history_memory": [],
    "frame": b"",
}

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            try:
                with open("dashboard.html", "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"404")

        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            data = {
                "episode": dashboard_state["episode"],
                "reward": dashboard_state["reward"],
                "epsilon": dashboard_state["epsilon"],
                "loss": dashboard_state["loss"],
                "memory": dashboard_state["memory"],
                "history_reward": dashboard_state["history_reward"],
                "history_loss": dashboard_state["history_loss"],
                "history_memory": dashboard_state["history_memory"]
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path == "/frame":
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(dashboard_state["frame"])

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_dashboard_server():
    server = HTTPServer(("0.0.0.0", 8000), DashboardHandler)
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

def run_inference():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    server_thread = threading.Thread(target=start_dashboard_server, daemon=True)
    server_thread.start()

    model = QNetwork(state_dim, action_dim)
    model.train()

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)

    try:
        quantized_model.load_state_dict(torch.load("tinyrl_int8.pth"))
    except FileNotFoundError:
        logging.error("No weights found.")
        return

    quantized_model.eval()
    logging.info("Inference start.")

    num_episodes = 200
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            frame = env.render()
            if frame is not None:
                dashboard_state["frame"] = frame.tobytes()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = quantized_model(state_tensor)
                action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

            time.sleep(0.02)

        dashboard_state["episode"] = episode
        dashboard_state["reward"] = episode_reward
        dashboard_state["history_reward"].append(episode_reward)
        logging.info(f"E:{episode} R:{episode_reward:.1f}")

    env.close()

    logging.info("Done.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    run_inference()
