import abc
import torch
import torch.nn.functional as F
import numpy as np
from safety_gymnasium import register
import gymnasium.spaces as spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium.utils.ezpickle import EzPickle


class SafeEnv(abc.ABC):
    @abc.abstractmethod
    def is_state_safe(self, states: torch.Tensor):
        pass

    @abc.abstractmethod
    def barrier_fn(self, states: torch.Tensor):
        pass

    def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor):
        pass

    def trans_fn(self, states: torch.Tensor, actions: torch.Tensor):
        pass

    def done_fn(self, states, actions, next_states):
        pass


def nonneg_barrier(x):
    return F.softplus(-3 * x)


# def interval_barrier(x, lb, rb, eps=1e-6):
#     x = (x - lb) / (rb - lb)
#     b = -((x + eps) * (1 - x + eps) / (0.5 + eps)**2).log()
#     b_min, b_max = 0, -np.log(4 * eps)
#     grad = 1. / eps - 1
#     out = grad * torch.max(-x, x - 1)
#     return torch.where(torch.as_tensor((0 < x) & (x < 1)), b, b_max + out)


def interval_barrier(x, lb, rb, eps=1e-2, grad=None):
    x = (x - lb) / (rb - lb) * 2 - 1
    b = -((1 + x + eps) * (1 - x + eps) / (1 + eps)**2).log()
    b_min, b_max = 0, -np.log(eps * (2 + eps) / (1 + eps)**2)
    if grad is None:
        grad = 2. / eps / (2 + eps)
    out = grad * (abs(x) - 1)
    return torch.where(torch.as_tensor((-1 < x) & (x < 1)), b / b_max, 1 + out)


class SafeInvertedPendulumEnv(InvertedPendulumEnv, SafeEnv):
    episode_unsafe = False

    def __init__(self, threshold=0.2, task='upright', random_reset=False, violation_penalty=10):
        self.threshold = threshold
        self.task = task
        self.random_reset = random_reset
        self.violation_penalty = violation_penalty
        super().__init__()
        EzPickle.__init__(self, threshold=threshold, task=task, random_reset=random_reset)  # deepcopy calls `get_state`

    def reset_model(self):
        if self.random_reset:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)
        else:
            self.set_state(self.init_qpos, self.init_qvel)
        self.episode_unsafe = False
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)

        next_state, _, terminated, truncated, info = super().step(a)
        # reward = (next_state[0]**2 + next_state[1]**2)  # + a[0]**2 * 0.01
        # reward = next_state[1]**2  # + a[0]**2 * 0.01

        if self.task == 'upright':
            reward = -next_state[1]**2
        elif self.task == 'swing':
            reward = next_state[1]**2
        elif self.task == 'move':
            reward = next_state[0]**2
        else:
            assert 0
        
        if abs(next_state[..., 1]) > self.threshold or abs(next_state[..., 0]) > 0.9:
            # breakpoint()
            self.episode_unsafe = True
            reward -= self.violation_penalty
        info['episode.unsafe'] = self.episode_unsafe
        return next_state, reward, float(self.episode_unsafe), self.episode_unsafe, truncated, info

    def is_state_safe(self, states):
        # return states[..., 1].abs() <= self.threshold
        return self.barrier_fn(states) <= 1.0

    def barrier_fn(self, states):
        return interval_barrier(states[..., 1], -self.threshold, self.threshold).maximum(interval_barrier(states[..., 0], -0.9, 0.9))

    def reward_fn(self, states, actions, next_states):
        return -(next_states[..., 0]**2 + next_states[..., 1]**2) - actions[..., 0]**2 * 0.01


class SafeInvertedPendulumSwingEnv(SafeInvertedPendulumEnv):
    def __init__(self, threshold=1.5, task='swing', random_reset=False, violation_penalty=10):
        super().__init__(threshold=threshold, task=task)


class SafeInvertedPendulumMoveEnv(SafeInvertedPendulumEnv):
    def __init__(self, threshold=0.2, task='move', random_reset=False, violation_penalty=10):
        super().__init__(threshold=threshold, task=task)

register(id='SafeInvertedPendulum-v2', entry_point=SafeInvertedPendulumEnv, max_episode_steps=1000)
register(id='SafeInvertedPendulumSwing-v2', entry_point=SafeInvertedPendulumSwingEnv, max_episode_steps=1000)
register(id='SafeInvertedPendulumMove-v2', entry_point=SafeInvertedPendulumMoveEnv, max_episode_steps=1000)

class SafeClassicPendulum(PendulumEnv, SafeEnv):
    def __init__(self, threshold=1.5, init_state=(0.3, -0.9), goal_state=(0, 0), max_torque=2.0, obs_type='state', task='upright', **kwargs):
        self.init_state = np.array(init_state, dtype=np.float32)
        self.goal_state = goal_state
        self.threshold = threshold
        self.obs_type = obs_type
        self.task = task
        super().__init__(**kwargs)

        if obs_type == 'state':
            high = np.array([np.pi / 2, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        elif obs_type == 'observation':
            high = np.array([1, 1, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:
            assert 0

        self.max_torque = max_torque
        self.action_space = spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        th, thdot = self.state
        if self.obs_type == 'state':
            return np.array([angle_normalize(th), thdot], dtype=np.float32)
        else:
            return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self):
        self.state = self.init_state
        self.last_u = None
        self.episode_unsafe = False
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        # costs = (angle_normalize(th) - self.goal_state[0]) ** 2 + \
        #     0.1 * (thdot - self.goal_state[1]) ** 2  # + 0.001 * (u ** 2)
        costs = (angle_normalize(th) - self.goal_state[0]) ** 2

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot], np.float32)
        if abs(newth) > self.threshold:
            # costs = 1000
            self.episode_unsafe = True
            done = True
        else:
            done = False
        return self._get_obs(), -costs, float(self.episode_unsafe), done, False, {'episode.unsafe': self.episode_unsafe}

    def reward_fn(self, states, actions, next_states):
        th, thdot = self.parse_state(states)
        max_torque = self.max_torque

        actions = actions.clamp(-1, 1)[..., 0] * max_torque
        goal_th, goal_thdot = self.goal_state
        costs = (th - goal_th) ** 2 + .1 * (thdot - goal_thdot) ** 2 + .001 * actions ** 2
        costs = torch.where(self.is_state_safe(next_states), costs, torch.tensor(1000., device=costs.device))

        return -costs

    def trans_fn(self, states: torch.Tensor, u: torch.Tensor):
        th, thdot = self.parse_state(states)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = u.clamp(-1, 1)[..., 0] * self.max_torque

        newthdot = thdot + (-3 * self.g / (2 * l) * (th + np.pi).sin() + 3. / (m * l ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = newthdot.clamp(-self.max_speed, self.max_speed)

        dims = list(range(1, states.ndim)) + [0]
        if self.obs_type == 'state':
            return torch.stack([newth, newthdot]).permute(dims)
        return torch.stack([newth.cos(), newth.sin(), newthdot]).permute(dims)

    def parse_state(self, states):
        if self.obs_type == 'state':
            thdot = states[..., 1]
            th = states[..., 0]
        else:
            thdot = states[..., 2]
            th = torch.atan2(states[..., 1], states[..., 0])
        return th, thdot

    def is_state_safe(self, states: torch.Tensor):
        th, thdot = self.parse_state(states)
        return th.abs() <= self.threshold

    def barrier_fn(self, states: torch.Tensor):
        th, thdot = self.parse_state(states)
        b1 = interval_barrier(th, -self.threshold, self.threshold)
        return b1

class SafeClassicPendulumUpright(SafeClassicPendulum):
    def __init__(self, threshold=1.5, init_state=(0.3, -0.9), goal_state=(0, 0), max_torque=2, obs_type='state', task='upright', **kwargs):
        super().__init__(threshold, init_state, goal_state, max_torque, obs_type, task, **kwargs)

class SafeClassicPendulumTilt(SafeClassicPendulum):
    def __init__(self, threshold=1.5, init_state=(0.3, -0.9), goal_state=(-0.41151684, 0), max_torque=2, obs_type='state', task='upright', **kwargs):
        super().__init__(threshold, init_state, goal_state, max_torque, obs_type, task, **kwargs)


register(id='MyPendulum-v0', entry_point=SafeClassicPendulum, max_episode_steps=200)
register(id='MyPendulumUpright-v0', entry_point=SafeClassicPendulumUpright, max_episode_steps=200)
register(id='MyPendulumTilt-v0', entry_point=SafeClassicPendulumTilt, max_episode_steps=200)