# Workflow for Setting Up a USV Training Environment in IsaacLab

This guide provides a comprehensive workflow for creating an Unmanned Surface Vehicle (USV) training environment using IsaacLab and Isaac Sim.

---

## Table of Contents
1. [Phase 1: Environment Setup and Planning](#phase-1-environment-setup-and-planning)
2. [Phase 2: Create Custom Environment](#phase-2-create-custom-environment)
3. [Phase 3: Implement Environment Components](#phase-3-implement-environment-components)
4. [Phase 4: Training Setup](#phase-4-training-setup)
5. [Phase 5: Testing and Iteration](#phase-5-testing-and-iteration)

---

## Phase 1: Environment Setup and Planning

### 1. Define Your USV Task

**Objective**: Clearly define what you want the USV to learn
- Waypoint navigation
- Obstacle avoidance
- Path following
- Docking maneuvers
- Station keeping
- Multi-USV coordination

**Observation Space**: Define sensor inputs
- GPS coordinates (position)
- IMU data (orientation, angular velocity, linear acceleration)
- Velocity (linear and angular)
- Target/waypoint information
- Cameras (RGB, depth)
- LiDAR/sonar for obstacle detection
- Distance to obstacles/boundaries

**Action Space**: Define control outputs
- Continuous: Thrust values, rudder angle
- Discrete: Forward/backward/left/right commands
- Differential thrust for twin-engine USVs

**Reward Function**: Define success metrics
- Distance to goal/waypoint
- Collision penalties
- Heading alignment
- Fuel/energy efficiency
- Time efficiency
- Smoothness of trajectory

### 2. Prepare Assets

**USV Model Requirements**:
- USD or URDF format
- Proper rigid body dynamics
- Mass and inertia properties
- Buoyancy configuration
- Collision meshes
- Visual meshes
- Articulated joints (rudder, propellers)

**Environment Assets**:
- Water surface (plane or OmniWater)
- Obstacles (buoys, rocks, other vessels)
- Navigation markers
- Docks/ports
- Terrain (shorelines, islands)

---

## Phase 2: Create Custom Environment

### 3. Set Up Directory Structure

```bash
cd /opt/IsaacLab

# Create directory for your USV task
mkdir -p source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/navigation/usv

# Create necessary files
cd source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/navigation/usv
touch __init__.py
touch usv_env_cfg.py
touch usv_env.py
```

### 4. Create Environment Configuration

**File**: `usv_env_cfg.py`

```python
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

##
# Scene Configuration
##

@configclass
class USVSceneCfg(InteractiveSceneCfg):
    """Configuration for the USV scene."""

    # Ground plane (water surface)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # USV robot
    usv = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/path/to/your/usv_model.usd",  # Update with your USV model path
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),  # Slightly above water
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Target marker
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP Settings
##

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # USV state
        base_lin_vel = ObsTerm(func=lambda env: env.scene["usv"].data.root_lin_vel_b)
        base_ang_vel = ObsTerm(func=lambda env: env.scene["usv"].data.root_ang_vel_b)
        base_position = ObsTerm(func=lambda env: env.scene["usv"].data.root_pos_w)
        base_orientation = ObsTerm(func=lambda env: env.scene["usv"].data.root_quat_w)
        
        # Target information
        target_position = ObsTerm(func=lambda env: env.scene["target"].data.root_pos_w)
        
        # Relative information
        distance_to_target = ObsTerm(func=compute_distance_to_target)
        direction_to_target = ObsTerm(func=compute_direction_to_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the environment."""

    # Positive rewards
    progress_to_target = RewTerm(
        func=reward_progress_to_target,
        weight=1.0,
    )
    
    reaching_target = RewTerm(
        func=reward_reaching_target,
        weight=10.0,
    )

    # Negative rewards
    action_rate = RewTerm(
        func=reward_action_rate,
        weight=-0.01,
    )
    
    energy_consumption = RewTerm(
        func=reward_energy_consumption,
        weight=-0.001,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the environment."""

    # Time limit
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length)
    
    # Success condition
    target_reached = DoneTerm(
        func=lambda env: compute_distance_to_target(env) < 1.0
    )


@configclass
class EventsCfg:
    """Event terms for the environment."""

    # Reset events
    reset_scene = EventTerm(
        func=lambda env, env_ids: reset_scene_to_default(env, env_ids),
        mode="reset",
    )
    
    # Randomize target position
    randomize_target = EventTerm(
        func=lambda env, env_ids: randomize_target_position(env, env_ids),
        mode="reset",
    )


##
# Environment Configuration
##

@configclass
class USVEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the USV navigation environment."""

    # Scene settings
    scene: USVSceneCfg = USVSceneCfg(num_envs=1024, env_spacing=10.0)
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Environment settings
    episode_length_s = 30.0  # Episode length in seconds
    decimation = 2  # Number of physics steps per control step
    
    def __post_init__(self):
        """Post initialization."""
        # Set up simulation parameters
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation


# Helper functions (implement these based on your needs)
def compute_distance_to_target(env):
    """Compute distance from USV to target."""
    usv_pos = env.scene["usv"].data.root_pos_w
    target_pos = env.scene["target"].data.root_pos_w
    return ((usv_pos - target_pos) ** 2).sum(dim=-1).sqrt()


def compute_direction_to_target(env):
    """Compute normalized direction vector from USV to target."""
    usv_pos = env.scene["usv"].data.root_pos_w
    target_pos = env.scene["target"].data.root_pos_w
    direction = target_pos - usv_pos
    return direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)


def reward_progress_to_target(env):
    """Reward for making progress toward target."""
    # Implement reward based on distance change
    pass


def reward_reaching_target(env):
    """Bonus reward for reaching the target."""
    distance = compute_distance_to_target(env)
    return (distance < 1.0).float()


def reward_action_rate(env):
    """Penalty for large actions (encourage smooth control)."""
    # Implement based on action magnitude
    pass


def reward_energy_consumption(env):
    """Penalty for energy usage."""
    # Implement based on thrust/power usage
    pass


def reset_scene_to_default(env, env_ids):
    """Reset the scene to default state."""
    # Implement reset logic
    pass


def randomize_target_position(env, env_ids):
    """Randomize target position for curriculum learning."""
    # Implement randomization
    pass
```

### 5. Create Environment Class

**File**: `usv_env.py`

```python
from omni.isaac.lab.envs import ManagerBasedRLEnv
from .usv_env_cfg import USVEnvCfg


class USVEnv(ManagerBasedRLEnv):
    """Environment for USV navigation tasks."""
    
    cfg: USVEnvCfg

    def __init__(self, cfg: USVEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the USV environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Additional initialization if needed
        self._setup_water_physics()
        self._setup_sensors()
    
    def _setup_water_physics(self):
        """Set up water physics (buoyancy, drag, etc.)."""
        # Implement water physics setup
        # This might involve setting up buoyancy forces, drag coefficients, etc.
        pass
    
    def _setup_sensors(self):
        """Set up any additional sensors (cameras, lidar, etc.)."""
        # Implement sensor setup
        pass
```

### 6. Register the Environment

**File**: `__init__.py`

```python
import gymnasium as gym

from .usv_env import USVEnv
from .usv_env_cfg import USVEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-USV-Navigation-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.navigation.usv:USVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": USVEnvCfg,
        "rl_games_cfg_entry_point": None,  # Add if using RL Games
    },
)
```

---

## Phase 3: Implement Environment Components

### 7. Implement Water Physics

Create a separate module for water dynamics:

**File**: `water_physics.py`

```python
import torch
from omni.isaac.lab.utils.math import quat_rotate_inverse


class WaterPhysics:
    """Implements water physics for USV simulation."""
    
    def __init__(self, water_density=1000.0, drag_coefficient=0.5):
        self.water_density = water_density
        self.drag_coefficient = drag_coefficient
    
    def compute_buoyancy_force(self, usv_state, submerged_volume):
        """Compute buoyancy force based on submerged volume."""
        gravity = 9.81
        buoyancy_force = self.water_density * gravity * submerged_volume
        return torch.tensor([0.0, 0.0, buoyancy_force])
    
    def compute_drag_force(self, velocity, cross_sectional_area):
        """Compute hydrodynamic drag force."""
        drag_force = 0.5 * self.water_density * self.drag_coefficient * cross_sectional_area * velocity.pow(2)
        return -drag_force * velocity.sign()  # Oppose motion
```

### 8. Implement Reward Functions

Update your reward functions with specific implementations:

```python
def reward_progress_to_target(env):
    """Reward for making progress toward target."""
    current_distance = compute_distance_to_target(env)
    previous_distance = env.previous_distance_to_target
    progress = previous_distance - current_distance
    env.previous_distance_to_target = current_distance
    return progress * 10.0


def reward_action_rate(env):
    """Penalty for large actions (encourage smooth control)."""
    if env.actions is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.sum(env.actions ** 2, dim=-1)


def reward_energy_consumption(env):
    """Penalty for energy usage."""
    # Assuming actions represent thrust
    return -torch.abs(env.actions).sum(dim=-1)
```

---

## Phase 4: Training Setup with Stable-Baselines3

### 9. Create Gymnasium Wrapper

IsaacLab environments are compatible with Gymnasium, but you may need a wrapper for SB3 compatibility:

**File**: `sb3_wrapper.py`

```python
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class IsaacLabVecEnvWrapper(VecEnv):
    """Wrapper to make IsaacLab environment compatible with SB3."""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        VecEnv.__init__(
            self,
            self.num_envs,
            self.observation_space,
            self.action_space,
        )
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step(self._actions)
        
        # Convert to numpy for SB3
        obs = obs.cpu().numpy()
        rewards = rewards.cpu().numpy()
        dones = dones.cpu().numpy()
        
        # SB3 expects dones to include both termination and truncation
        dones = np.logical_or(dones, truncated.cpu().numpy())
        
        return obs, rewards, dones, infos
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()
    
    def close(self):
        self.env.close()
    
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)]
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]
    
    def seed(self, seed=None):
        self.env.seed(seed)
```

### 10. Create Training Script

**File**: `train_usv_sb3.py`

```python
import argparse
import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train USV with SB3")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3"], help="RL algorithm")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli=["--headless"] if args.headless else [])
simulation_app = app_launcher.app

# Import after launching Isaac Sim
from sb3_wrapper import IsaacLabVecEnvWrapper


def create_env():
    """Create the USV environment."""
    # Create IsaacLab environment
    env = gym.make("Isaac-USV-Navigation-v0", num_envs=args.num_envs)
    
    # Wrap for SB3 compatibility
    env = IsaacLabVecEnvWrapper(env)
    
    # Optional: Add normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    return env


def main():
    """Main training function."""
    
    # Create environment
    env = create_env()
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./logs/{args.algo}_usv/",
        name_prefix="usv_model",
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"./logs/{args.algo}_usv/best_model/",
        log_path=f"./logs/{args.algo}_usv/eval/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    
    # Select algorithm
    if args.algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=f"./logs/{args.algo}_usv/tensorboard/",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif args.algo == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log=f"./logs/{args.algo}_usv/tensorboard/",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif args.algo == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log=f"./logs/{args.algo}_usv/tensorboard/",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    # Train the model
    print(f"Training {args.algo} on USV Navigation task...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    model.save(f"./logs/{args.algo}_usv/final_model")
    env.save(f"./logs/{args.algo}_usv/vec_normalize.pkl")
    
    print("Training complete!")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

### 11. Run Training

```bash
# Train with PPO (recommended for continuous control)
python train_usv_sb3.py --algo PPO --num_envs 512 --total_timesteps 5000000

# Train with SAC (good for continuous actions)
python train_usv_sb3.py --algo SAC --num_envs 256 --total_timesteps 3000000

# Train headless (faster, no visualization)
python train_usv_sb3.py --algo PPO --num_envs 1024 --total_timesteps 5000000 --headless
```

### 12. Monitor Training Progress

```bash
# Launch TensorBoard
tensorboard --logdir ./logs/PPO_usv/tensorboard/

# Open browser to http://localhost:6006
```

---

## Phase 5: Testing and Evaluation

### 13. Create Evaluation Script

**File**: `eval_usv_sb3.py`

```python
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate trained USV model")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3"])
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--render", action="store_true", help="Render the environment")
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli=[] if args.render else ["--headless"])
simulation_app = app_launcher.app

from sb3_wrapper import IsaacLabVecEnvWrapper


def evaluate_model():
    """Evaluate the trained model."""
    
    # Create environment
    env = gym.make("Isaac-USV-Navigation-v0", num_envs=1)
    env = IsaacLabVecEnvWrapper(env)
    
    # Load normalization statistics
    try:
        env = VecNormalize.load(f"{args.model_path}_vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("No normalization statistics found, using raw environment")
    
    # Load model
    if args.algo == "PPO":
        model = PPO.load(args.model_path)
    elif args.algo == "SAC":
        model = SAC.load(args.model_path)
    elif args.algo == "TD3":
        model = TD3.load(args.model_path)
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check if target was reached
                if "target_reached" in info[0] and info[0]["target_reached"]:
                    success_count += 1
                
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print statistics
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.num_episodes} episodes)")
    print("="*50)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate: {success_count / args.num_episodes * 100:.1f}%")
    print("="*50)
    
    env.close()


if __name__ == "__main__":
    evaluate_model()
    simulation_app.close()
```

### 14. Run Evaluation

```bash
# Evaluate trained model with visualization
python eval_usv_sb3.py --model_path ./logs/PPO_usv/final_model --algo PPO --num_episodes 10 --render

# Evaluate without rendering (faster)
python eval_usv_sb3.py --model_path ./logs/PPO_usv/best_model/best_model --algo PPO --num_episodes 100
```

### 15. Create Inference/Deployment Script

**File**: `deploy_usv.py`

```python
import argparse
import gymnasium as gym
from stable_baselines3 import PPO

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

app_launcher = AppLauncher(args_cli=[])
simulation_app = app_launcher.app

from sb3_wrapper import IsaacLabVecEnvWrapper


def deploy():
    """Deploy trained model for continuous operation."""
    
    env = gym.make("Isaac-USV-Navigation-v0", num_envs=1)
    env = IsaacLabVecEnvWrapper(env)
    
    model = PPO.load(args.model_path)
    
    obs = env.reset()
    
    print("USV is now operational. Press Ctrl+C to stop.")
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if done:
                obs = env.reset()
                print("Target reached! Moving to next waypoint...")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        env.close()


if __name__ == "__main__":
    deploy()
    simulation_app.close()
```

---

## Phase 6: Advanced Features

### 16. Curriculum Learning

Implement progressive difficulty:

```python
class CurriculumCallback:
    """Callback for curriculum learning."""
    
    def __init__(self, env):
        self.env = env
        self.current_level = 0
        self.levels = [
            {"max_distance": 10.0, "num_obstacles": 0},
            {"max_distance": 20.0, "num_obstacles": 2},
            {"max_distance": 50.0, "num_obstacles": 5},
            {"max_distance": 100.0, "num_obstacles": 10},
        ]
    
    def __call__(self, locals_, globals_):
        # Check if ready to advance
        if locals_["self"].num_timesteps % 100000 == 0:
            success_rate = self.evaluate_current_level()
            if success_rate > 0.8 and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                self.update_environment()
                print(f"Advanced to curriculum level {self.current_level}")
    
    def update_environment(self):
        level_config = self.levels[self.current_level]
        self.env.set_attr("max_target_distance", level_config["max_distance"])
        self.env.set_attr("num_obstacles", level_config["num_obstacles"])
```

### 17. Domain Randomization

Add domain randomization for robustness:

```python
@configclass
class RandomizationCfg:
    """Configuration for domain randomization."""
    
    # Randomize USV properties
    mass_range = (50.0, 150.0)  # kg
    drag_coefficient_range = (0.3, 0.7)
    
    # Environmental randomization
    water_current_range = (-1.0, 1.0)  # m/s
    wave_height_range = (0.0, 0.5)  # m
    wind_speed_range = (0.0, 10.0)  # m/s
    
    # Sensor noise
    gps_noise_std = 0.1  # m
    imu_noise_std = 0.01  # rad/s or m/s²
```

### 18. Multi-Task Learning

Train on multiple objectives simultaneously:

```python
# Register multiple task variants
gym.register(
    id="Isaac-USV-Waypoint-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.navigation.usv:USVEnv",
    kwargs={"env_cfg_entry_point": USVWaypointCfg},
)

gym.register(
    id="Isaac-USV-Docking-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.navigation.usv:USVEnv",
    kwargs={"env_cfg_entry_point": USVDockingCfg},
)

gym.register(
    id="Isaac-USV-ObstacleAvoidance-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.navigation.usv:USVEnv",
    kwargs={"env_cfg_entry_point": USVObstacleAvoidanceCfg},
)
```

---

## Tips and Best Practices

### Performance Optimization
- Start with fewer environments (256-512) for debugging
- Scale up to 1024+ environments for faster training
- Use headless mode for production training
- Monitor GPU memory usage

### Hyperparameter Tuning
- **PPO**: Good default for most tasks
  - Learning rate: 3e-4
  - Batch size: 64-256
  - n_steps: 2048-4096
  
- **SAC/TD3**: Better for precise control
  - Learning rate: 3e-4
  - Buffer size: 1M
  - Batch size: 256

### Debugging Tips
1. Start with a simple reward function
2. Visualize training with small num_envs first
3. Check reward scaling (should be in range [-10, 10])
4. Monitor success rate and episode length
5. Use TensorBoard for tracking metrics

### Common Issues
- **Slow training**: Increase num_envs, use headless mode
- **Unstable training**: Reduce learning rate, normalize observations
- **Poor performance**: Check reward function, increase training time
- **Memory issues**: Reduce num_envs or buffer size

---

## Resources

- **IsaacLab Documentation**: https://isaac-sim.github.io/IsaacLab/
- **Stable-Baselines3 Documentation**: https://stable-baselines3.readthedocs.io/
- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/
- **Gymnasium Documentation**: https://gymnasium.farama.org/

---

## Summary

This workflow provides a complete pipeline for:
1. ✅ Setting up IsaacLab environment
2. ✅ Creating custom USV simulation
3. ✅ Integrating with Stable-Baselines3
4. ✅ Training RL agents with PPO/SAC/TD3
5. ✅ Evaluating and deploying trained models
6. ✅ Advanced features (curriculum learning, domain randomization)

Start with the basic setup, verify it works, then progressively add complexity!