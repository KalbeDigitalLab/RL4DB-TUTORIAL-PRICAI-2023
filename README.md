# Reinforcement Learning for Digital Business

Welcome to the "Reinforcement Learning for Digital Business" repository! This project provides simple implementations of reinforcement learning algorithms for online advertising and inventory management in a simulated environment. Follow the steps below to get started.

## Getting Started

### Step 1: Set up the Environment

1. Create a virtual environment using Conda:
   **Note:** This environment tested with python version 3.7, so if you need to reproduce this code without problem, please use same python version.
   ```bash
   conda create --name rl4db python=3.7
   ```

2. Activate the virtual environment:
   ```bash
   conda activate rl4db
   ```

### Step 2: Install Requirements

Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

**Note:**
If you encounter the error `AttributeError: module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'`, reinstall `opencv-python-headless` using pip:
   ```bash
   pip uninstall opencv-python-headless
   pip install opencv-python-headless
   ```

## Running the Examples

Explore the provided examples for reinforcement learning in online advertising and inventory management. Each example is located in its respective directory.
