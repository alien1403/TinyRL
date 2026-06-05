# TinyRL

A minimal Deep Reinforcement Learning implementation tailored for resource-constrained microcontrollers (e.g., ESP32, ARM Cortex-M0). 

This project trains a DQN agent on CartPole-v1 while strictly enforcing a 1 MB memory footprint and simulating INT8 precision.

## Key Features

- **Quantization-Aware Training (QAT)**: Uses PyTorch QAT directly within the RL training loop. The network learns a policy using simulated 8-bit precision, proving convergence without a Floating Point Unit (FPU).
- **Strict Memory Management**: Pre-allocates contiguous `np.int8` and `np.float32` arrays for the Replay Buffer. This prevents dynamic memory allocation overhead and guarantees the combined model and buffer footprint stays under 1 MB.
- **Micro-Architecture**: Neural network is constrained to a maximum of 2 hidden layers with 32 neurons each.
- **Zero-Dependency Dashboard**: Includes a custom `http.server` thread that streams real-time MJPEG video and training metrics directly to an HTML5 Canvas. No Flask, Node, or TensorBoard required.

### Training
Start the training loop and dashboard:
```bash
docker compose up --build
```
The INT8 quantized weights will be saved to `tinyrl_int8.pth` upon completion.
*(Note: Once the terminal logs that the weights are saved, press `CTRL + C` in the terminal to stop the container before starting inference.)*

### Inference
Run inference using the saved INT8 weights:
```bash
docker compose run --service-ports tinyrl python inference.py
```

### Dashboard
During either training or inference, access the real-time dashboard at:
`http://localhost:8000`

To expose the dashboard remotely, you can tunnel the port using ngrok:
```bash
ngrok http 8000
```

## License
MIT
