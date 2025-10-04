# Adapting MarineGym for USV Development in Isaac Sim

## Overview

MarineGym is a recent (March 2025) high-performance RL platform built on Isaac Sim for underwater robotics. While not yet publicly released, you can adapt their concepts for USV development. This guide shows you how.

## Current Status of MarineGym

**Important**: MarineGym code appears to not be publicly available yet. The project was just published in March 2025 by researchers at Zhejiang University and Heriot-Watt University.

**Project Info**:
- Website: https://marine-gym.com/
- Paper: https://arxiv.org/abs/2503.09203
- Performance: 250,000 FPS on RTX 3060 GPU
- Features: 5 UUV models, GPU-accelerated hydrodynamics, domain randomization

## Strategy for USV Development

Since MarineGym focuses on underwater vehicles (UUVs), here's how to adapt it for surface vehicles (USVs):

### Option 1: Wait for Public Release (Recommended if timing allows)
Contact the authors at Zhejiang University to inquire about:
- Code release timeline
- Possibility of early access for research
- Collaboration opportunities
- Contact: lmw@zju.edu.cn, li_dejun@zju.edu.cn

### Option 2: Build Your Own USV System (Start Now)

Based on MarineGym's architecture, create a custom USV simulator using their approach:

## Core Components to Implement

### 1. **Hydrodynamics Plugin** (GPU-Accelerated)

MarineGym uses Fossen's equations adapted for GPU computation. For USVs, you need surface-specific dynamics:

**Key Differences from UUVs**:
- Surface wave interaction (heave, pitch, roll)
- Air-water interface effects
- Wind forces (above water)
- Reduced depth control complexity

**Implementation approach**:
```python
# Pseudo-code structure based on MarineGym's approach
class USVHydrodynamics:
    def __init__(self):
        self.water_density = 1025  # kg/m³
        self.air_density = 1.225   # kg/m³
        
    def compute_forces(self, state, control_input):
        # Added mass effects
        added_mass = self.calculate_added_mass(state)
        
        # Hydrodynamic drag
        drag = self.calculate_drag(state)
        
        # Wave forces (NEW for USV)
        wave_forces = self.calculate_wave_forces(state)
        
        # Wind forces (NEW for USV)
        wind_forces = self.calculate_wind_forces(state)
        
        # Buoyancy (surface floating)
        buoyancy = self.calculate_surface_buoyancy(state)
        
        return total_forces
```

### 2. **Vehicle Models**

**USV Types to Model**:

1. **Differential Drive USV**
   - Two thrusters, differential steering
   - Simple, highly maneuverable
   - Good for initial testing

2. **Rudder-Propeller USV**
   - Single main thruster + rudder
   - More efficient for straight-line travel
   - Common commercial configuration

3. **Airboat USV**
   - Above-water propulsion
   - Excellent for shallow waters
   - Less efficient but unique capabilities

4. **Hybrid Air-Sea Vehicle** (like MarineGym's HAUV)
   - Adapt their tiltrotor concept
   - Can transition between surface and aerial modes

### 3. **Environmental Simulation**

**Water Surface**:
- Wave generation using Gerstner waves or FFT-based methods
- Variable sea states (calm to rough)
- Current simulation

**Atmospheric Conditions**:
- Wind speed and direction
- Gusts and turbulence
- Weather effects

### 4. **Sensor Suite for USVs**

```python
# USV-specific sensors
sensors = {
    'gps': GPSSensor(),           # High accuracy on surface
    'imu': IMUSensor(),            # 6-DOF motion
    'compass': CompassSensor(),    # Heading
    'lidar': LiDARSensor(),        # Obstacle detection
    'camera': CameraSensor(),      # Visual perception
    'radar': RadarSensor(),        # Long-range detection
    'depth_sounder': DepthSensor() # Water depth below
}
```

## Practical Implementation Steps

### Step 1: Setup Isaac Sim Environment

```bash
# Install Isaac Sim (version 4.2.0 or later)
# Follow NVIDIA's official installation guide
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html

# Create conda environment
conda create -n usv_sim python=3.10
conda activate usv_sim

# Install dependencies
pip install torch torchvision
pip install numpy scipy matplotlib
```

### Step 2: Start with Basic Floating Physics

Use the `isaac_underwater` repository as a starting point:

```bash
git clone https://github.com/leonlime/isaac_underwater.git
cd isaac_underwater
```

This provides basic buoyancy and drag implementations you can modify.

### Step 3: Implement Surface-Specific Physics

**Key modifications needed**:

1. **Buoyancy for surface vessels**: Modify to keep vessel at waterline
2. **Wave interaction**: Add wave force calculations
3. **Wind forces**: Implement above-water drag and side forces
4. **Stability**: Implement metacentric height calculations

### Step 4: Create USV URDF Models

```xml
<!-- Example simplified USV URDF structure -->
<robot name="differential_usv">
  <link name="hull">
    <inertial>
      <mass value="50.0"/>
      <inertia ixx="5.0" ixy="0" ixz="0" 
               iyy="8.0" iyz="0" izz="10.0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="usv_hull.obj"/>
      </geometry>
    </visual>
  </link>
  
  <joint name="left_thruster_joint" type="continuous">
    <parent link="hull"/>
    <child link="left_thruster"/>
  </joint>
  
  <link name="left_thruster">
    <!-- Thruster properties -->
  </link>
  
  <!-- Right thruster similar -->
</robot>
```

### Step 5: Integrate with Isaac Sim

```python
# Example Isaac Sim USV setup
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

# Create world
world = World()

# Load USV model
usv_path = "/path/to/usv.usd"
add_reference_to_stage(usd_path=usv_path, prim_path="/World/USV")

# Create robot wrapper
usv = Robot(prim_path="/World/USV")

# Add custom physics
# (This is where you implement your hydrodynamics)
```

## Learning from MarineGym's Approach

### GPU Acceleration
MarineGym achieves 250,000 FPS using PyTorch tensors for physics calculations:

```python
import torch

class GPUHydrodynamics:
    def __init__(self, num_envs=1024):
        self.num_envs = num_envs
        self.device = torch.device('cuda')
        
    def compute_batch_forces(self, states, controls):
        # All calculations in batch on GPU
        forces = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Vectorized calculations here
        # This allows parallel simulation of 1000+ USVs
        
        return forces
```

### Domain Randomization
For robust policies, randomize:
- Hull mass and inertia
- Thruster characteristics
- Water density
- Wave conditions
- Wind patterns
- Sensor noise

## Alternative: Modify Existing Frameworks

If MarineGym isn't available, you can extend:

1. **Pegasus Simulator** (for aerial vehicles)
   - GitHub: https://github.com/PegasusSimulator/PegasusSimulator
   - Modify for surface operations
   - Already has good Isaac Sim integration

2. **Isaac Lab**
   - Official NVIDIA framework
   - Create custom USV environment
   - Good documentation and examples

## Resources and Next Steps

**Essential Reading**:
1. MarineGym paper: https://arxiv.org/abs/2503.09203
2. Fossen's "Handbook of Marine Craft Hydrodynamics" (for theory)
3. Isaac Sim documentation: https://docs.omniverse.nvidia.com/isaacsim/

**Community Resources**:
- NVIDIA Isaac Sim forum: https://forums.developer.nvidia.com/c/isaac-sim
- Isaac Sim Discord
- awesome-isaac-gym: https://github.com/robotlearning123/awesome-isaac-gym

**Recommended Workflow**:
1. Start with basic floating cube (isaac_underwater)
2. Add wave simulation
3. Implement differential drive USV
4. Add realistic sensors
5. Create RL training tasks
6. Scale up with GPU acceleration

## Contact the Researchers

For the most direct path, reach out to the MarineGym team:
- **Email**: lmw@zju.edu.cn, li_dejun@zju.edu.cn
- **Institution**: Zhejiang University, China
- **Ask about**: Code release, collaboration, adapting for USV

They may be interested in expanding MarineGym to include surface vehicles!

## Conclusion

While MarineGym isn't publicly available yet, you can:
1. Build a similar system using their published methodology
2. Start with basic physics implementations available in isaac_underwater
3. Contact the authors for potential collaboration
4. Contribute to the community by open-sourcing your USV implementation

The underwater robotics and surface vehicle communities would greatly benefit from a unified Isaac Sim framework covering both domains!