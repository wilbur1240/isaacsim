# Installing Isaac Lab with Isaac Sim via Pip in Conda

## Your Current Setup ✅
- **Isaac Sim**: 5.0.0 (installed via pip)
- **Environment**: Conda
- **OS**: Ubuntu 22.04
- **Python**: 3.11
- **Status**: Perfect! Ready to add Isaac Lab to the same conda environment

## Compatibility Confirmation

Isaac Lab 2.2 includes full compatibility with Isaac Sim 5.0 as well as backwards compatibility with Isaac Sim 4.5

For Isaac Sim 5.X, the required Python version is 3.11

Your setup with conda + pip installation is fully supported! 🎉

## Installation Overview

Since you've already installed Isaac Sim via pip in a conda environment, you'll install Isaac Lab in the **same conda environment** using the pip installation method. This keeps everything unified and avoids dependency conflicts.

## Step-by-Step Installation

### Step 1: Verify Your Conda Environment

First, make sure your conda environment with Isaac Sim is working:

```bash
# Activate your conda environment
conda activate <your_env_name>  # Replace with your actual environment name

# Verify Python version
python --version
# Should output: Python 3.11.x

# Verify Isaac Sim is installed
python -c "import isaacsim; print('Isaac Sim installed successfully')"

# Check PyTorch and CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

**Expected Output**: You should see PyTorch 2.4+ with CUDA 12.4 or higher

### Step 2: Install Isaac Lab via Pip

With your conda environment activated, install Isaac Lab:

```bash
# Make sure your conda environment is activated
conda activate <your_env_name>

# Install Isaac Lab
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab (this will also install necessary dependencies)
pip install isaaclab==2.2.0
```

**Alternative: Install from source for development**

If you want to modify Isaac Lab or stay on the latest development version:

```bash
# Clone Isaac Lab repository
cd ~/Projects  # or wherever you want to put it
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Checkout the latest stable release
git checkout v2.2.0

# Install in editable mode (recommended for development)
pip install -e .

# Install optional learning frameworks
pip install -e ".[rl_games]"    # For RL Games
pip install -e ".[rsl_rl]"      # For RSL RL
pip install -e ".[sb3]"         # For Stable Baselines3
pip install -e ".[skrl]"        # For skrl
pip install -e ".[all]"         # Or install everything
```

### Step 3: Verify Installation

Test if Isaac Lab is installed correctly:

```bash
# Test 1: Check Isaac Lab installation
python -c "import isaaclab; print(f'Isaac Lab version: {isaaclab.__version__}')"

# Test 2: List available environments
python -c "from isaaclab.envs import list_all_envs; print(list_all_envs())"

# Test 3: Run a simple script
python -m isaaclab.envs --task Isaac-Cartpole-v0 --num_envs 16
```

### Step 4: Install Additional RL Frameworks (Optional)

If you plan to do reinforcement learning:

```bash
# Install learning frameworks
pip install rl-games==1.6.1
pip install rsl-rl
pip install stable-baselines3
pip install skrl
```

### Step 5: Set Up Environment Variables

Add these to your shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
# Isaac Lab directory (only if you installed from source)
export ISAACLAB_PATH="${HOME}/Projects/IsaacLab"

# Optional: Create an alias for quick activation
alias isaaclab="conda activate <your_env_name>"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Directory Structure

### If Installed via Pip Only:
```
<conda_env>/lib/python3.11/site-packages/
├── isaacsim/              # Isaac Sim
├── isaaclab/              # Isaac Lab core
└── isaaclab_tasks/        # Example tasks
```

### If Installed from Source (Editable):
```
~/Projects/IsaacLab/
├── scripts/
│   ├── tutorials/           # Tutorial examples
│   └── tools/               # Development tools
├── source/
│   ├── extensions/
│   │   ├── isaaclab/        # Core Isaac Lab
│   │   ├── isaaclab_tasks/  # Example tasks
│   │   └── usv_simulation/  # Your custom extension!
│   └── standalone/          # Standalone scripts
└── docs/                    # Documentation
```

## Creating Your USV Extension

### Option 1: If Installed via Pip

Create a separate project directory:

```bash
# Create your project directory
mkdir -p ~/Projects/usv_simulation
cd ~/Projects/usv_simulation

# Create package structure
mkdir -p usv_simulation/{robots,tasks,utils}
touch usv_simulation/__init__.py
touch setup.py

# Install in editable mode
pip install -e .
```

**setup.py example:**
```python
from setuptools import setup, find_packages

setup(
    name="usv_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "isaaclab>=2.2.0",
    ],
)
```

### Option 2: If Installed from Source

Use the extension creation tool:

```bash
# Navigate to Isaac Lab
cd ~/Projects/IsaacLab

# Create extension
python scripts/tools/create_extension.py --name usv_simulation

# This creates:
# source/extensions/usv_simulation/
```

## Running Scripts

### Basic Usage

```bash
# Activate your conda environment
conda activate <your_env_name>

# Run a script directly
python scripts/my_script.py

# Run with Isaac Lab's task runner
python -m isaaclab.envs --task Isaac-Cartpole-v0 --num_envs 32

# Run in headless mode
python -m isaaclab.envs --task Isaac-Cartpole-v0 --num_envs 32 --headless

# Run with specific GPU
python -m isaaclab.envs --task Isaac-Cartpole-v0 --device cuda:0
```

### Common Script Patterns

```python
# In your Python scripts
import isaaclab
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Your code here...
```

## Testing Examples

Test Isaac Lab with built-in examples:

```bash
# Activate environment
conda activate <your_env_name>

# Example 1: Cartpole environment
python -m isaaclab.envs --task Isaac-Cartpole-v0 --num_envs 64

# Example 2: Humanoid locomotion
python -m isaaclab.envs --task Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 32

# Example 3: Manipulation task
python -m isaaclab.envs --task Isaac-Reach-Franka-v0 --num_envs 16

# Example 4: Run with visualization
python -m isaaclab.envs --task Isaac-Cartpole-v0 --num_envs 4 --enable_cameras
```

## Common Issues & Solutions

### Issue 1: "Module not found: isaaclab"
**Solution**: 
```bash
# Make sure conda environment is activated
conda activate <your_env_name>

# Reinstall Isaac Lab
pip install isaaclab==2.2.0
```

### Issue 2: CUDA/PyTorch version mismatch
**Solution**: 
```bash
# Check what CUDA version you have
python -c "import torch; print(torch.version.cuda)"

# If needed, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Issue 3: Import errors with Isaac Sim
**Solution**:
```bash
# Verify Isaac Sim packages are installed
pip list | grep isaacsim

# If missing, reinstall
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

### Issue 4: Performance issues
**Solution**:
```bash
# Enable tensor compilation (PyTorch 2.0+)
export TORCH_COMPILE_MODE=reduce-overhead

# Increase shared memory for parallel environments
ulimit -n 4096
```

### Issue 5: Display/GUI issues
**Solution**:
```bash
# Run in headless mode for servers
python script.py --headless

# For GUI issues on desktop
export DISPLAY=:0
```

## Development Workflow

### Typical Development Cycle

```bash
# 1. Activate environment
conda activate <your_env_name>

# 2. Navigate to your project
cd ~/Projects/usv_simulation  # or ~/Projects/IsaacLab if from source

# 3. Edit your code
code .  # Or use your preferred editor

# 4. Test your code
python scripts/test_usv.py --num_envs 16

# 5. Debug with smaller environment count
python scripts/test_usv.py --num_envs 1 --enable_cameras

# 6. Run training
python scripts/train_usv.py --task USV-Navigate-v0 --num_envs 4096 --headless
```

### Using Jupyter Notebooks

```bash
# Install Jupyter in your conda environment
conda install jupyter

# Start Jupyter
jupyter notebook

# In notebook cell:
import isaaclab
from isaaclab.envs import ManagerBasedRLEnv
# Your code...
```

## Performance Tips

### Optimize for Training

```bash
# Use large batch sizes for parallel training
python train.py --num_envs 4096 --headless

# Enable tensor compilation
export TORCH_COMPILE_MODE=reduce-overhead

# Use mixed precision training
python train.py --num_envs 2048 --headless --mixed_precision
```

### Optimize for Development

```bash
# Use fewer environments for faster iteration
python script.py --num_envs 4

# Enable live visualization
python script.py --num_envs 1 --enable_cameras

# Use CPU for debugging (slower but easier to debug)
python script.py --device cpu --num_envs 1
```

## Quick Reference Commands

```bash
# Environment management
conda activate <your_env_name>
conda deactivate

# List installed packages
pip list | grep isaac

# Update Isaac Lab
pip install --upgrade isaaclab

# Run environments
python -m isaaclab.envs --task <TaskName> --num_envs <N>

# Common flags
--num_envs 32          # Number of parallel environments
--headless             # Run without GUI
--enable_cameras       # Enable camera rendering
--device cuda:0        # Specify GPU device
--task TaskName        # Specify task to run
--seed 42              # Set random seed
```

## Resources

- **Documentation**: https://isaac-sim.github.io/IsaacLab/main/
- **GitHub**: https://github.com/isaac-sim/IsaacLab
- **Isaac Sim Docs**: https://docs.omniverse.nvidia.com/isaacsim/latest/
- **Forum**: https://forums.developer.nvidia.com/c/isaac-sim
- **Discord**: Join NVIDIA Omniverse Discord

## Summary

✅ **Your conda setup with pip-installed Isaac Sim is fully supported**

✅ **Installing Isaac Lab in the same conda environment keeps everything unified**

✅ **No need for complex linking or bundled Python - everything is managed by pip/conda**

✅ **You can now create your USV environment using standard Python workflows**

The key advantages of this approach:
- Everything in one conda environment
- Standard Python import statements work
- Easy to manage dependencies with pip
- No special wrapper scripts needed
- Full IDE integration

Ready to start building your USV simulation? Let me know if you encounter any issues!