"""
wrapper定义文件
"""
from typing import Union, List, Tuple, Callable
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper,FinalEvalRewardEnv
import gym
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import torch
from ding.torch_utils import to_ndarray
import os
import warnings
import copy
import subprocess
import shutil



# 粘性动作wrapper
class StickyActionWrapper(gym.ActionWrapper):
    """
    Overview:
       A certain possibility to select the last action
    Interface:
        ``__init__``, ``action``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``p_sticky``: possibility to select the last action
    """

    def __init__(self, env: gym.Env, p_sticky: float=0.25):
        super().__init__(env)
        self.p_sticky = p_sticky
        self.last_action = 0

    def action(self, action):
        if np.random.random() < self.p_sticky:
            return_action = self.last_action
        else:
            return_action = action
        self.last_action = action
        return return_action


# 稀疏奖励wrapper
class SparseRewardWrapper(gym.Wrapper):
    """
    Overview:
       Only death and pass sparse reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        dead = True if reward == -15 else False
        reward = 0
        if info['flag_get']:
            reward = 15
        if dead:
            reward = -15
        return obs, reward, done, info


# 硬币奖励wrapper
class CoinRewardWrapper(gym.Wrapper):
    """
    Overview:
        add coin reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_coins = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += (info['coins'] - self.num_coins) * 10
        self.num_coins = info['coins']
        return obs, reward, done, info

# 连续进度奖励wrapper
class ContinuousDriveRewardWrapper(gym.Wrapper):
    """
    Overview:
        Dense reward shaping: 鼓励向右推进、吃金币，轻度惩罚耗时，死亡额外惩罚。
    Interface:
        ``__init__``, ``reset``, ``step``
    """

    def __init__(
        self,
        env: gym.Env,
        progress_coef: float = 0.05,   # x_pos 每前进 1 像素给的奖励
        coin_coef: float = 1.0,        # 每增加 1 枚金币的奖励
        time_penalty: float = 0.01,    # 每步耗时的惩罚（基于游戏倒计时，time 递减）
        alive_bonus: float = 0.001,    # 存活的小额奖励，鼓励持续探索，但远小于 time_penalty，避免原地站桩
        death_penalty: float = 5.0,    # 死亡时额外惩罚
        stagnation_step_limit: int = 20,   # 允许在原地或小范围震荡的最大步数
        stagnation_penalty: float = 0.5,   # 进入“卡住”状态后的额外每步惩罚
        stagnation_x_threshold: float = 1.0,  # 认为“前进了一点点”的最小增量，用于避免在小范围来回抖动时重置计数
    ):
        super().__init__(env)
        self.progress_coef = progress_coef
        self.coin_coef = coin_coef
        self.time_penalty = time_penalty
        self.alive_bonus = alive_bonus
        self.death_penalty = death_penalty
        self.stagnation_step_limit = stagnation_step_limit
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_x_threshold = stagnation_x_threshold
        self.last_x = 0
        self.last_coins = 0
        self.last_time = 400  # mario 环境默认倒计时初值，运行时会在 reset 时更新
        # 记录到目前为止到达过的最远 x，用于识别“在某个位置附近长时间徘徊”
        self.best_x = 0
        self.steps_since_best = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        info = getattr(self.env, "unwrapped", self.env)
        # 从 info 读取初始 x、time、coins
        self.last_x = 0
        self.last_coins = 0
        self.last_time = getattr(info, "time", 400)
        self.best_x = 0
        self.steps_since_best = 0
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # 进度：只奖励向右的增量
        cur_x = info.get("x_pos", self.last_x)
        delta_x = max(0, cur_x - self.last_x)

        # 使用“历史最远 x_pos”来判断是否长期没有实质前进：
        # - 只有当当前位置比历史最远位置大于一个阈值时，才认为真正取得了进展，并重置计数；
        # - 在两个墙之间来回左右移动时，cur_x 基本在一个小区间内震荡，不会超过 best_x+threshold，
        #   因此 steps_since_best 会持续累加，最终触发惩罚。
        if cur_x > self.best_x + self.stagnation_x_threshold:
            self.best_x = cur_x
            self.steps_since_best = 0
        else:
            self.steps_since_best += 1

        # 金币增量
        cur_coins = info.get("coins", self.last_coins)
        delta_coins = max(0, cur_coins - self.last_coins)

        # 时间惩罚（环境 time 是递减的）
        cur_time = info.get("time", self.last_time)
        delta_time = max(0, self.last_time - cur_time)

        shaped = (
            base_reward
            + self.progress_coef * delta_x
            + self.coin_coef * delta_coins
            + self.alive_bonus
            - self.time_penalty * delta_time
        )

        # 对“卡住”状态添加额外惩罚：
        # - 在允许的无前进步数内，仅靠 time_penalty 就是略微负收益
        # - 一旦超过 stagnation_step_limit，每一步都会叠加额外惩罚，迫使智能体离开局部最优。
        if self.steps_since_best >= self.stagnation_step_limit and not info.get("flag_get", False):
            shaped -= self.stagnation_penalty

        # 死亡额外惩罚
        if done and not info.get("flag_get", False):
            shaped -= self.death_penalty

        # 更新缓存
        self.last_x = cur_x
        self.last_coins = cur_coins
        self.last_time = cur_time

        return obs, shaped, done, info

# CAM相关，不需要了解
def _convert_to_h264(src_path: str):
    if shutil.which('ffmpeg') is None:
        warnings.warn("ffmpeg not found; skip h264 conversion")
        return
    base, ext = os.path.splitext(src_path)
    dst_path = f"{base}_h264.mp4"
    cmd = [
        'ffmpeg', '-y', '-i', src_path,
        '-c:v', 'libx264', '-crf', '18', dst_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        warnings.warn(f"ffmpeg convert failed for {src_path}: {e}")


def dump_arr2video(arr, video_folder):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 6
    size = (256, 240)
    cam_path = video_folder + '/cam_pure.mp4'
    obs_path = video_folder + '/obs_pure.mp4'
    merged_path = video_folder + '/merged.mp4'
    out = cv2.VideoWriter(cam_path, fourcc, fps, size)
    out1 = cv2.VideoWriter(obs_path, fourcc, fps, size)
    out2 = cv2.VideoWriter(merged_path, fourcc, fps, size)
    for frame, obs in arr:
        frame = (255 * frame).astype('uint8').squeeze(0)
        frame_c = cv2.resize(cv2.applyColorMap(frame, cv2.COLORMAP_JET), size)
        out.write(frame_c)

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out1.write(obs)

        merged_frame = cv2.addWeighted(obs, 0.6, frame_c, 0.4, 0)
        out2.write(merged_frame)

    out.release(); out1.release(); out2.release()
    _convert_to_h264(cam_path)
    _convert_to_h264(obs_path)
    _convert_to_h264(merged_path)


def get_cam(img, model):
    target_layers = [model.encoder.main[0]]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam


def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordCAM(gym.Wrapper):

    def __init__(
        self,
        env,
        cam_model,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super(RecordCAM, self).__init__(env)
        self._env = env
        self.cam_model = cam_model

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum([x is not None for x in [episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = []

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            warnings.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        observations = super(RecordCAM, self).reset(**kwargs)
        if not self.recording:
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = []

        self.recorded_frames = 0
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        time_step = super(RecordCAM, self).step(action)
        observations, rewards, dones, infos = time_step

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.append(
                (get_cam(observations, model=self.cam_model), copy.deepcopy(self.env.render(mode='rgb_array')))
            )
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > 10000:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones or infos['time'] < 250:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return time_step

    def close_video_recorder(self) -> None:
        if self.recorded_frames > 0:
            dump_arr2video(self.video_recorder, self.video_folder)
        self.video_recorder = []
        self.recording = False
        self.recorded_frames = 0

    def seed(self, seed: int) -> None:
        self._env.seed(seed)