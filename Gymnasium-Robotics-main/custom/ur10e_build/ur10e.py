import numpy as np
import os
import mujoco
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# ur10e pick and place task
# the registration call below will automatically add the environment to
# Gymnasium's registry when this module is imported.  You can still
# instantiate the class directly if you prefer to skip the registry.
from gymnasium.envs.registration import register


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ur10eEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "ur10e_gripper.xml"),
        frame_skip: int = 2,
        no_early_termination: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            no_early_termination,
            **kwargs,
        )

        self.observation_space = spaces.Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float64),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        })

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            **kwargs,
        )

        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.cartesian_action_scale = 0.16  # meters per action step
        self.max_joint_delta = 1.0  # maximum joint target step per control update
        self.joint_lower_limits = self.model.jnt_range[:6, 0].copy()
        self.joint_upper_limits = self.model.jnt_range[:6, 1].copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.gripper_state = 0.0
        
        self.stage = 0

        self.elbow_z = 0.0
        self.wrist2_rot = np.eye(3).flatten()
        self.shoulder_rot = np.eye(3).flatten()
        self.object0_pos = np.zeros(3)
        self.object0_rot = np.eye(3).flatten()
        self.episode_success = False
        self.no_early_termination = no_early_termination
        self.hover_pos = None
        self.hover_offset = 0.30
        self.prev_distance = 0.0
        self.gamma = 0.99
        self._gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("left_driver_joint").id]
        self.grasp_verify_steps = 0
        self.grasp_verified = False
        self.object_grasped = False
        self.close_steps = 0

    def _current_goal_pos(self):
        object0_pos = self.data.site("object0").xpos.copy()
        if self.stage == 0:
            return object0_pos
        elif self.stage == 1:
            return object0_pos
        return object0_pos
        #return self.data.body("target").xpos.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if action.shape == (7,):
            cart_action = action[:3]
            gripper_signal = action[6]
        elif action.shape == (4,):
            cart_action = action[:3]
            gripper_signal = action[3]
        else:
            raise ValueError(f"Expected action shape (4,) or (7,), got {action.shape}")

        cart_action = np.clip(cart_action, -1.0, 1.0) * self.cartesian_action_scale
        # --- POSITION ERROR ---
        pos_error = cart_action

        # --- ORIENTATION ERROR (HARDCODED TCP DOWN) ---
        flange_rot = self.data.site("UR10E_TCP").xmat.reshape(3, 3)
        current_z = flange_rot[:, 2]
        desired_z = np.array([0.0, 0.0, -1.0])

        # force rotation toward downward Z
        rot_error = np.cross(current_z, desired_z)

        # keep your original gain style
        rot_gain = 0.5
        rot_error *= rot_gain

        # Combine into 6D task space command
        target_delta_6d = np.concatenate([pos_error, rot_error])

        # --- FULL 6D JACOBIAN (position + rotation) ---
        jacp = np.zeros((3, self.model.nv), dtype=np.float64, order="C")
        jacr = np.zeros((3, self.model.nv), dtype=np.float64, order="C")

        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.model.site("UR10E_TCP").id
        )

        # Combine into 6xN Jacobian
        jacobian = np.vstack([jacp[:, :6], jacr[:, :6]])
                
        # Damped least-squares (DLS) for numerical stability near singularities
        damping = 0.05
        jac_t = jacobian.T
        identity = np.eye(jacobian.shape[1])
        joint_delta = (
            jac_t
            @ np.linalg.inv(jacobian @ jac_t + damping**2 * np.eye(6))
            @ target_delta_6d
        )
        joint_delta = np.clip(joint_delta, -self.max_joint_delta, self.max_joint_delta)

        desired_qpos = self.data.qpos[:6].copy() + joint_delta
        desired_qpos = np.clip(desired_qpos, self.joint_lower_limits, self.joint_upper_limits)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[:6] = desired_qpos
        ctrl[6] = np.clip((gripper_signal + 1.0) * 0.5 * 255.0, 0.0, 255.0)

        prev_achieved = self.data.site("UR10E_TCP").xpos.copy()
        #prev_desired = self._current_goal_pos()
        


        distance = np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.object0_pos)


        self.do_simulation(ctrl, self.frame_skip)
        self.gripper_state = float(self.data.qpos[self._gripper_qpos_idx]) / 0.9  # normalize to [0, 1], assuming 0.9 is fully closed

        observation = self._get_obs()

        # Cache sim state to self.* BEFORE compute_reward so it can read them
        self.elbow_z      = self.data.body("UR10E_forearm_link").xpos[2]
        self.wrist2_rot   = self.data.body("UR10E_wrist_2_link").xmat.copy().flatten()
        self.shoulder_rot = self.data.body("UR10E_shoulder_link").xmat.copy().flatten()
        self.object0_pos  = self.data.site("object0").xpos.copy()
        self.object0_rot  = self.data.site("object0").xmat.copy().flatten()
        
        # Compute z_dot for orientation reward
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z = flange_rot[:, 2]
        z_dot = float(np.dot(flange_z, np.array([0, 0, -1])))

        prev_desired = self.data.site("object0").xpos.copy()
        self.prev_distance = float(np.linalg.norm(prev_achieved - prev_desired))
        # Build info BEFORE compute_reward so it can be passed in
        info = {
            "elbow_z":       self.elbow_z,
            "wrist2_rot":    self.wrist2_rot,
            "shoulder_rot":  self.shoulder_rot,
            "object0_pos":   self.object0_pos,
            "object0_rot":   self.object0_rot,
            "hover_pos":     self.hover_pos,
            "gripper_state": self.gripper_state,   
            "stage":         self.stage,
            "z_dot":         z_dot,
            "prev_distance": self.prev_distance,
            "is_success":    False,
            "grasp_verify_steps": self.grasp_verify_steps,
            "grasp_verified": self.grasp_verified,
            "object_grasped": self.object_grasped,
            "close_steps": self.close_steps,
        }

        
        reward, reward_components = self._compute_reward_components(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )

        info["reward_components"] = reward_components

        ag_pos    = observation["achieved_goal"][:3]
        dg_pos    = observation["desired_goal"][:3]
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z   = flange_rot[:, 2]
        target_down = np.array([0, 0, -1])
        z_dot = np.sum(flange_z * target_down, axis=-1)
        distance   = np.linalg.norm(ag_pos - dg_pos)

        terminated = False
        gripper_open = self.gripper_state < 0.3
        #print(distance, gripper_open)
        gripper_close = self.gripper_state > 0.45
        if self.stage == 0:
            if distance < 0.04 and gripper_open:
                self.stage = 1   
                """terminated = True
                self.episode_success = True
                if self.no_early_termination:
                    terminated = False"""
        elif self.stage == 1:
            if gripper_close and self.object0_pos[2] > 0.05:
                terminated = True
                self.episode_success = True
                if self.no_early_termination:
                    terminated = False

        elif self.stage == 2:
            if distance < 0.02 and z_dot > 0.96 and gripper_close and self.object0_pos[2] > 0.2:
                terminated = True
                self.episode_success = True
                if self.no_early_termination:
                    terminated = False
        else:
            print("Invalid stage:", self.stage)
            "so far the 2 stages are somewhat successful but at this point I also need to add stage 2 which will lift the box up towards the target"

        # Now that compute_reward has run, update is_success in info
        info["is_success"] = bool(self.episode_success)

        truncated = False
        if self.render_mode == "human":
            target_geom_id = self.model.geom("target_geom").id
            if self.stage == 0:
                self.model.geom_rgba[target_geom_id] = [1, 0, 0, 1]  # red = hover stage
            elif self.stage == 1:
                self.model.geom_rgba[target_geom_id] = [1, 0.5, 0, 1]    # orange = reach stage
            elif self.stage == 2:   
                self.model.geom_rgba[target_geom_id] = [0, 1, 0, 1]      # green = lift stage
            if self.episode_success:
                self.model.geom_rgba[target_geom_id] = [0, 0, 1, 1]  # blue = success 
            self.render()


        return observation, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # This is the entry point for Stable Baselines3 HER
        total, _ = self._compute_reward_components(achieved_goal, desired_goal, info)
        return total
    
    def _compute_reward_components(self, achieved_goal, desired_goal, info):
        # 1. Unpack goals (supports both single vectors and batches)
        ag_pos = achieved_goal[..., :3]
        dg_pos = desired_goal[..., :3]

        # 2. Extract state from info (Handle single dict or list of dicts from HER)
        if isinstance(info, (list, np.ndarray)):
            # Batch mode (HER Replay)
            # Use .get() with defaults to avoid KeyErrors during edge cases
            z_dot = np.array([i.get("z_dot", 0.0) for i in info])
            obj_pos = np.array([i.get("object0_pos", [0,0,0]) for i in info])
            gripper_state = np.array([i.get("gripper_state", 0.0) for i in info])
            stages = np.array([i.get("stage", 0) for i in info])
        else:
            # Single mode (Live step)
            z_dot = info.get("z_dot", 0.0)
            obj_pos = info.get("object0_pos", np.zeros(3))
            gripper_state = info.get("gripper_state", 0.0)
            stages = info.get("stage", 0)

        # 3. Calculate distance-based reward
        dist_total = np.linalg.norm(ag_pos - dg_pos, axis=-1)
        base_reward = -dist_total

        # 4. Gripper Rewards (penalties for being in the wrong state)
        # We use np.minimum/maximum to ensure compatibility with both scalars and arrays
        gripper_reward_open  = np.minimum(0.0, 0.2 - gripper_state) * 0.5
        gripper_reward_close = np.minimum(0.0, gripper_state - 0.7) * 0.5

        # 5. Calculate Final Reward using NumPy masking
        # This replaces the "if stages == 0" logic which causes the ValueError
        if isinstance(stages, np.ndarray):
            reward = np.zeros_like(dist_total, dtype=np.float64)
            
            # Mask for Stage 0: Reach for box while keeping gripper open
            mask0 = (stages == 0)
            reward[mask0] = base_reward[mask0] + gripper_reward_open[mask0]
            
            # Mask for Stage 1: Close gripper and lift
            mask1 = (stages == 1)
            # Scaled bonus down to 2.0 to prevent Q-value explosion
            lift_bonus = np.where(obj_pos[..., 2] > 0.05, 2.0, 0.0)
            reward[mask1] = base_reward[mask1] + gripper_reward_close[mask1] + lift_bonus[mask1]
            
            # Mask for Stage 2: Move to target while holding box
            mask2 = (stages == 2)
            lift_bonus_s2 = np.where(obj_pos[..., 2] > 0.05, 1.0, 0.0)
            reward[mask2] = base_reward[mask2] + lift_bonus_s2[mask2]
        else:
            # Scalar logic for the live environment step
            if stages == 0:
                reward = base_reward + gripper_reward_open
            elif stages == 1:
                object_height = max(0.0, obj_pos[2] - 0.025)
                lift_reward = object_height * 80.0
                reward = base_reward + gripper_reward_close + lift_reward
                #reward = base_reward + gripper_reward_close
                if obj_pos[2] > 0.05:
                    reward += 2.0
                    
            else:
                reward = base_reward
                if obj_pos[2] > 0.05:
                    reward += 1.0

        # 6. Logging components (must return scalars for the info dict)
        orientation_reward = (np.mean(z_dot) * 0.5 - 0.5)
        components = {
            "distance_reward": float(np.mean(base_reward)),
            "orientation_reward": float(orientation_reward),
            "total_reward": float(np.mean(reward)),
            "avg_stage": float(np.mean(stages))
        }

        return reward, components
    #since gym 0.26, the step function should return (obs, reward, terminated, truncated, info) instead of (obs, reward, done, info) to distinguish between episode termination and truncation due to time limits or other factors.
    #I should use terminated to indicate if the episode ended due to success or failure, and truncated to indicate if it ended due to time limits or other factors. In this case, I will set terminated to False for now and rely on the TimeLimit wrapper to handle episode truncation after a certain number of steps.

    
    def reset_model(self):#resetira model na začetne pogoje
        self.stage = 0
        self.prev_distance_xy = None
        self.prev_distance_z = None
        self.grasp_verify_steps = 0
        self.grasp_verified = False
        self.object_grasped = False
        self.close_steps = 0

        key_id = self.model.key("UR10E_home").id
        #qpos = self.init_qpos#initial position and velocity of the robot arm
        #qvel = self.init_qvel
        qpos = self.model.key_qpos[key_id].copy()
        qvel = self.model.key_qvel[key_id].copy()
        ctrl = self.model.key_ctrl[key_id].copy()

        self.gripper_state = 0.0
        self.episode_success = False

        object0_id = self.model.body("object0").id
        object0_jnt_adr = self.model.body_jntadr[object0_id]

        #distance of the target from origin
        x = self.np_random.uniform(low=-0.4, high=0.4, size=1)
        y = self.np_random.uniform(low=0.5, high=0.8, size=1)
        while True:
            xy = self.np_random.uniform(low=-0.8, high=0.8, size=2)#distance of x y z from origin
            #z = self.np_random.uniform(low=0.2, high=0.7, size=1)
            z = np.array([0.3])
            self.goal = np.concatenate([x, y, z])
            
            #donut shaped range
            dist = np.linalg.norm(self.goal[:2])
            break
        
        yaw  = self.np_random.uniform(0, 2 * np.pi)
        half = yaw / 2.0
        goal_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz

        while True:
            x2 = self.np_random.uniform(low=-0.2, high=0.4, size=1)
            y2 = self.np_random.uniform(low=0.5, high=0.8, size=1)
            #z2  = self.np_random.uniform(low=0.025, high=0.025, size=1)
            z2 = np.array([0.025])  # fixed height for the object, can be adjusted as needed
            object_pos = np.concatenate([x, y, z2])
            break
        yaw2  = self.np_random.uniform(0, 2 * np.pi)
        half2 = yaw2 / 2.0
        object_quat = np.array([np.cos(half2), 0.0, 0.0, np.sin(half2)])
        
        #random_rot = random_rotation_matrix(self.np_random)  # see helper below
        #self.goal_rot = random_rot.flatten()
        #self.data.body("target").xpos[:] = self.goal#da ne skace vec tocka
        self.model.body("target").pos[:] = self.goal
        self.model.body("target").quat[:] = goal_quat

        object0_qpos_start = object0_jnt_adr
        qpos[object0_qpos_start:object0_qpos_start+3] = object_pos#the stuff to make the cube get rotated and moved is so garbage
        qpos[object0_qpos_start+3:object0_qpos_start+7] = object_quat#I know theres a better way but I forgor
        # Zero out object0 velocities (6 DOF: linear + angular)
        qvel[object0_qpos_start:object0_qpos_start+6] = 0

        self.hover_pos = object_pos + np.array([0.0, 0.0, self.hover_offset])  # hover above the object


        self.set_state(qpos, qvel)
        self.data.ctrl[:] = ctrl
        self.prev_distance = float(np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.object0_pos))

        # Initialize prev_dist to the actual initial distance after setting the goal
        return self._get_obs()

    def _get_obs(self):#dobi obzervacijo da pol lahko reagira na okolje 
        position = self.data.qpos[:6].flatten()#joint angles
        velocity = self.data.qvel[:6].flatten()#joint velocities

        flange_pos = self.data.site("UR10E_TCP").xpos.copy()
        flange_rot = self.data.site("UR10E_TCP").xmat.copy()
        flange_z_axis = flange_rot.reshape(3, 3)[:, 2]

        target_pos = self.data.body("target").xpos

        object0_pos = self.data.site("object0").xpos.copy()
        if self.stage == 0:
            # hover point: object XY, object Z + 0.15
            #desired_pos = self.hover_pos
            desired_pos = object0_pos
        elif self.stage == 1:
            # stage 1: reach the box itself
            desired_pos = object0_pos
        else:
            # stage 2: lift the box up to the target
            desired_pos = object0_pos
        stage = np.array([self.stage], dtype=np.float64)

        return {
            "observation": np.concatenate([
                np.cos(position),
                np.sin(position),
                velocity,
                flange_z_axis,
                object0_pos,
                self.data.site("object0").xmat.copy().flatten(),
                stage
            ]).astype(np.float64),
            "achieved_goal": flange_pos.astype(np.float64),
            "desired_goal": desired_pos.astype(np.float64),
        }

# register custom environment automatically on import
# you can call `gym.make("UR10E-v0")` once this module is imported
register(
    id="UR10E-v0",
    entry_point="ur10e:ur10eEnv",
    max_episode_steps=500,
) 