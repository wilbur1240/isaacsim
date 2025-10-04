# Building a USV Environment with Isaac Lab

## Yes, Isaac Lab Can Do This! ✅

Isaac Lab is **perfect** for building your USV environment. It provides all the tools you need:
- Custom robot support (you can add any vehicle)
- Physics simulation (rigid body dynamics via PhysX)
- Custom environment creation
- Sensor integration (cameras, LiDAR, GPS, IMU)
- Domain randomization capabilities
- RL training integration (optional for later)

**You don't need GPU acceleration to start** - that's just for training thousands of parallel environments. For development and testing, CPU simulation works fine!

## Architecture Overview

Isaac Lab uses a modular approach:
1. **Robot Configuration** - Define your USV as an Articulation
2. **Environment Setup** - Create the scene (water, obstacles, etc.)
3. **Physics** - Add custom forces (buoyancy, drag, waves)
4. **Sensors** - Attach cameras, GPS, IMU to your USV
5. **Tasks** (optional) - Define goals for RL training

## Step-by-Step Implementation

### Step 1: Install Isaac Lab

```bash
# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run installation (this also installs Isaac Sim if needed)
./isaaclab.sh --install

# Verify installation
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

**System Requirements**:
- Ubuntu 20.04/22.04 or Windows 10/11
- Python 3.10
- NVIDIA GPU (RTX recommended, but GTX 1060+ works)
- 32GB RAM recommended

### Step 2: Create Your Extension Project

Isaac Lab uses extensions to organize custom code:

```bash
# Create a new extension from template
cd IsaacLab
python scripts/tools/create_extension.py --name usv_simulation
```

This creates a structure like:
```
source/extensions/usv_simulation/
├── usv_simulation/
│   ├── __init__.py
│   ├── robots/          # USV robot definitions
│   ├── tasks/           # Environment definitions
│   └── utils/           # Helper functions
├── config/
└── setup.py
```

### Step 3: Define Your USV Robot

Create `usv_simulation/robots/differential_usv.py`:

```python
"""Differential drive USV robot configuration."""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

# USV Configuration
DIFFERENTIAL_USV_CFG = ArticulationCfg(
    # Path to your USV USD file (we'll create this next)
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/your/usv.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # We'll handle buoyancy separately
            max_depenetration_velocity=1.0,
        ),
    ),
    
    # Initial state of the USV
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Spawn at water surface
        rot=(1.0, 0.0, 0.0, 0.0),  # No rotation (quaternion)
        joint_pos={
            "left_thruster": 0.0,
            "right_thruster": 0.0,
        },
    ),
    
    # Actuators (thrusters)
    actuators={
        "thrusters": ImplicitActuatorCfg(
            joint_names_expr=["left_thruster", "right_thruster"],
            effort_limit=20.0,  # Max thrust in Newtons
            velocity_limit=100.0,
            stiffness=0.0,  # Direct force control
            damping=10.0,
        ),
    },
)
```

### Step 4: Create USV 3D Model (USD File)

**Option A: Use existing USD file**
If you have a USV CAD model (STEP, STL, OBJ):

```python
# Convert to USD using Isaac Sim
from omni.isaac.core.utils.stage import create_new_stage
from pxr import UsdGeom, UsdPhysics, PhysxSchema

# Load your mesh
mesh_path = "/path/to/usv_hull.obj"
# Use Isaac Sim GUI: Create -> Mesh -> Import...
# Then add physics properties
```

**Option B: Create simple box USV for testing**

```python
# Simple rectangular hull for initial testing
"""
Create this USD with Isaac Sim GUI:
1. Create > Shape > Cube (name it "hull")
2. Scale to 2m x 1m x 0.5m
3. Add Physics > Rigid Body
4. Create 2 revolute joints for thrusters
5. Save as differential_usv.usd
"""
```

### Step 5: Implement USV Physics (Custom Forces)

Create `usv_simulation/utils/surface_physics.py`:

```python
"""Surface vehicle physics for buoyancy and drag."""

import torch
from isaaclab.envs import ManagerBasedEnv

class SurfacePhysics:
    """Handles water surface physics for USVs."""
    
    def __init__(self, env: ManagerBasedEnv):
        self.env = env
        self.water_level = 0.0  # Water surface at z=0
        self.water_density = 1025.0  # kg/m³ (seawater)
        self.air_density = 1.225  # kg/m³
        
        # USV physical properties
        self.hull_volume = 0.5  # m³ (submerged volume)
        self.drag_coefficient = 0.5
        self.hull_area = 2.0  # m² (frontal area)
        
    def compute_buoyancy(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute buoyancy force to keep USV at water surface.
        
        Args:
            positions: USV positions [num_envs, 3]
            
        Returns:
            buoyancy_force: Force vectors [num_envs, 3]
        """
        num_envs = positions.shape[0]
        forces = torch.zeros((num_envs, 3), device=positions.device)
        
        # Vertical position relative to water
        z_depth = positions[:, 2] - self.water_level
        
        # Buoyancy force (upward when submerged)
        # F = ρ * V * g
        g = 9.81
        buoyancy_magnitude = self.water_density * self.hull_volume * g
        
        # Apply restoring force when displaced from surface
        forces[:, 2] = -z_depth * buoyancy_magnitude * 10.0  # Spring-like restoring
        
        return forces
    
    def compute_drag(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Compute hydrodynamic drag.
        
        Args:
            velocities: USV velocities [num_envs, 6] (linear + angular)
            
        Returns:
            drag_force: Force vectors [num_envs, 6]
        """
        num_envs = velocities.shape[0]
        
        # Linear drag: F = -0.5 * ρ * v² * Cd * A * v_hat
        linear_vel = velocities[:, :3]
        speed = torch.norm(linear_vel, dim=1, keepdim=True)
        
        drag_magnitude = 0.5 * self.water_density * speed * speed * \
                        self.drag_coefficient * self.hull_area
        
        # Direction opposite to motion
        drag_force = torch.zeros((num_envs, 6), device=velocities.device)
        drag_force[:, :3] = -drag_magnitude * (linear_vel / (speed + 1e-6))
        
        # Angular drag (damping)
        drag_force[:, 3:] = -2.0 * velocities[:, 3:]
        
        return drag_force
    
    def compute_wave_forces(self, positions: torch.Tensor, time: float) -> torch.Tensor:
        """
        Simple sinusoidal wave forces.
        
        Args:
            positions: USV positions [num_envs, 3]
            time: Current simulation time
            
        Returns:
            wave_forces: Force vectors [num_envs, 3]
        """
        num_envs = positions.shape[0]
        forces = torch.zeros((num_envs, 3), device=positions.device)
        
        # Simple sinusoidal waves
        wave_height = 0.1  # meters
        wave_frequency = 0.5  # Hz
        
        # Vertical wave force
        forces[:, 2] = wave_height * torch.sin(
            2 * 3.14159 * wave_frequency * time + positions[:, 0]
        ) * 100.0
        
        return forces
```

### Step 6: Create USV Environment

Create `usv_simulation/tasks/usv_navigation_env.py`:

```python
"""USV navigation environment."""

from __future__ import annotations
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..robots.differential_usv import DIFFERENTIAL_USV_CFG
from ..utils.surface_physics import SurfacePhysics

@configclass
class USVNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for USV navigation environment."""
    
    # Environment settings
    episode_length_s = 30.0  # 30 second episodes
    decimation = 2  # Control frequency = sim_freq / decimation
    
    # Action and observation spaces
    action_space = 2  # Left and right thruster commands
    observation_space = 8  # [x, y, yaw, vx, vy, vyaw, goal_x, goal_y]
    state_space = 0
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,  # 60 Hz simulation
        render_interval=decimation
    )
    
    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,  # Start with 4 parallel environments
        env_spacing=10.0,  # 10 meters between environments
        replicate_physics=True
    )
    
    # USV robot
    robot: ArticulationCfg = DIFFERENTIAL_USV_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class USVNavigationEnv(DirectRLEnv):
    """USV navigation environment."""
    
    cfg: USVNavigationEnvCfg
    
    def __init__(self, cfg: USVNavigationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Initialize custom physics
        self.surface_physics = SurfacePhysics(self)
        
        # Goal positions for each environment
        self.goal_positions = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        
    def _setup_scene(self):
        """Setup the scene with water plane and USV."""
        
        # Add water plane (visual only)
        from isaaclab.sim.spawners import spawn_ground_plane
        spawn_ground_plane(
            "/World/ground",
            size=(100.0, 100.0),
            color=(0.1, 0.3, 0.6),  # Blue water
        )
        
        # Add USV robot
        self.robot = self.scene["robot"]
        
        super()._setup_scene()
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply thruster forces before physics step."""
        
        # Scale actions to thruster forces
        thruster_forces = actions * 10.0  # Scale to Newtons
        
        # Apply forces to thrusters
        self.robot.set_joint_effort_target(
            thruster_forces, 
            joint_ids=[0, 1]  # Left and right thrusters
        )
    
    def _apply_action(self):
        """Apply custom physics forces."""
        
        # Get current state
        positions = self.robot.data.root_pos_w
        velocities = self.robot.data.root_vel_w
        
        # Compute physics forces
        buoyancy = self.surface_physics.compute_buoyancy(positions)
        drag = self.surface_physics.compute_drag(velocities)
        waves = self.surface_physics.compute_wave_forces(
            positions, 
            self.episode_length_buf[0].item() * self.cfg.sim.dt
        )
        
        # Apply external forces
        total_force = buoyancy + drag[:, :3] + waves
        total_torque = drag[:, 3:]
        
        self.robot.set_external_force_and_torque(
            total_force, 
            total_torque
        )
    
    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        
        # USV state
        pos = self.robot.data.root_pos_w
        vel = self.robot.data.root_lin_vel_b
        
        # Extract position and orientation
        x = pos[:, 0]
        y = pos[:, 1]
        yaw = self._compute_yaw_from_quat(self.robot.data.root_quat_w)
        
        # Goal relative to USV
        goal_rel = self.goal_positions - pos[:, :2]
        
        obs = torch.cat([
            x.unsqueeze(1),
            y.unsqueeze(1),
            yaw.unsqueeze(1),
            vel[:, :2],  # vx, vy
            vel[:, 5].unsqueeze(1),  # vyaw
            goal_rel
        ], dim=1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        
        # Distance to goal
        pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(pos - self.goal_positions, dim=1)
        
        # Reward for getting closer to goal
        reward = -distance * 0.1
        
        # Bonus for reaching goal
        reward += torch.where(distance < 0.5, 10.0, 0.0)
        
        return reward.unsqueeze(1)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Success: reached goal
        pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(pos - self.goal_positions, dim=1)
        success = distance < 0.5
        
        return success, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        
        super()._reset_idx(env_ids)
        
        # Reset USV to random position
        num_resets = len(env_ids)
        
        # Random start positions
        start_pos = torch.zeros((num_resets, 3), device=self.device)
        start_pos[:, 0] = torch.rand(num_resets, device=self.device) * 4 - 2
        start_pos[:, 1] = torch.rand(num_resets, device=self.device) * 4 - 2
        start_pos[:, 2] = 0.0  # Water surface
        
        # Set robot state
        self.robot.write_root_pose_to_sim(
            start_pos,
            torch.tensor([[1.0, 0, 0, 0]], device=self.device).repeat(num_resets, 1),
            env_ids=env_ids
        )
        
        # Reset velocities
        self.robot.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device),
            env_ids=env_ids
        )
        
        # Random goal positions
        self.goal_positions[env_ids, 0] = torch.rand(num_resets, device=self.device) * 6 - 3
        self.goal_positions[env_ids, 1] = torch.rand(num_resets, device=self.device) * 6 - 3
    
    def _compute_yaw_from_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion."""
        # quat format: [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw
```

### Step 7: Test Your Environment

Create a simple test script `test_usv_env.py`:

```python
"""Test USV environment."""

from usv_simulation.tasks.usv_navigation_env import USVNavigationEnvCfg, USVNavigationEnv

def main():
    # Create environment
    env_cfg = USVNavigationEnvCfg()
    env_cfg.scene.num_envs = 1  # Single environment for testing
    
    env = USVNavigationEnv(env_cfg)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run simulation loop
    for i in range(1000):
        # Random actions for testing
        actions = torch.rand(env.num_envs, env.cfg.action_space) * 2 - 1
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        
        if i % 100 == 0:
            print(f"Step {i}: Reward = {rewards[0].item():.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
```

Run it:
```bash
./isaaclab.sh -p test_usv_env.py
```

## Next Steps After Basic Environment Works

### 1. **Add Realistic Water Visual**
```python
# In Isaac Sim GUI, create water shader:
# Create > Material > OmniPBR
# Adjust: color, roughness, metallic for water appearance
```

### 2. **Add Sensors**
```python
from isaaclab.sensors import CameraCfg, ContactSensorCfg

# Add camera to USV
camera = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/camera",
    update_period=0.1,
    height=480,
    width=640,
)
```

### 3. **Add Obstacles**
```python
# Spawn buoys, docks, or other obstacles
from isaaclab.sim.spawners import spawn_from_usd

spawn_from_usd(
    "/World/envs/env_.*/obstacle",
    usd_path="/path/to/buoy.usd",
)
```

### 4. **Improve Wave Simulation**
Use Gerstner waves or FFT-based methods for realistic ocean surfaces.

### 5. **Add Wind Forces**
Extend `SurfacePhysics` to include wind effects on above-water surfaces.

## Key Advantages of Isaac Lab for USV

✅ **Modular Design** - Easy to add/modify components
✅ **Well Documented** - Excellent tutorials and examples  
✅ **GPU Ready** - When you need speed, it scales easily
✅ **Active Community** - NVIDIA support + Discord
✅ **RL Integration** - Built-in support for training if needed
✅ **Sensor Rich** - Easy to add cameras, LiDAR, etc.

## Troubleshooting Tips

**Issue: USV sinks or flies away**
- Check `disable_gravity=False` in robot config
- Verify buoyancy force calculation
- Add debug prints to see forces

**Issue: Simulation is slow**
- Reduce `num_envs` during development
- Increase `decimation` value
- Use simpler collision meshes

**Issue: Physics unstable**
- Decrease simulation `dt` (try 1/120)
- Check mass and inertia values
- Reduce maximum velocities

## Resources

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab/
- **Tutorials**: Focus on "Adding a New Robot" and "Environment Design"
- **Examples**: Check `scripts/tutorials/` in Isaac Lab repo
- **Discord**: Join Omniverse Discord for community help

## Summary

**Yes, Isaac Lab is perfect for your USV environment!** The workflow is:

1. ✅ Install Isaac Lab (one command)
2. ✅ Create extension project 
3. ✅ Define USV robot config
4. ✅ Implement surface physics (buoyancy, drag, waves)
5. ✅ Build environment class
6. ✅ Test and iterate

You can start simple (floating box) and gradually add complexity (realistic hull, sensors, waves). No GPU acceleration needed initially - focus on getting the physics right first!

Let me know which step you'd like me to elaborate on!