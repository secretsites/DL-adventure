"""
智能体训练入口，包含训练逻辑
"""
import swanlab
from easydict import EasyDict
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv, StickyActionWrapper , SparseRewardWrapper , CoinRewardWrapper, ContinuousDriveRewardWrapper
from policy import DQNPolicy
from model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from mario_dqn_config import mario_dqn_config
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from functools import partial
import os
import gym_super_mario_bros


# 动作相关配置
action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


# mario环境
def wrapped_mario_env(version=0, action=7, obs=1, reward_mode="dense"):
    wrappers = [
        lambda env: MaxAndSkipWrapper(env, skip=4),
        lambda env: WarpFrameWrapper(env, size=84),
        lambda env: ScaledFloatFrameWrapper(env),
        lambda env: FrameStackWrapper(env, n_frames=obs),
    ]

    if reward_mode == "dense":
        wrappers.append(
            lambda env: ContinuousDriveRewardWrapper(
                env,
                progress_coef=0.05,
                coin_coef=1.0,
                time_penalty=0.01,
                alive_bonus=0.001,
                death_penalty=5.0,
                stagnation_step_limit=10,
                stagnation_penalty=0.75,
                stagnation_x_threshold=1.0,
            )
        )
    elif reward_mode == "sparse":
        wrappers.append(lambda env: SparseRewardWrapper(env))
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    wrappers.append(lambda env: CoinRewardWrapper(env))
    wrappers.append(lambda env: FinalEvalRewardEnv(env))

    return DingEnvWrapper(
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={'env_wrapper': wrappers}
    )


def main(cfg, args, seed=0, max_env_step=int(3e6)):
    # Easydict类实例，包含一些配置
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        seed=seed,
        save_cfg=True,
    )
    # 在 compile_config 之后再次覆盖 ckpt/迭代相关可选参数，避免被默认值回写
    if getattr(args, "save_ckpt_after_iter", None) is not None:
        if 'learner' not in cfg.policy.learn:
            cfg.policy.learn.learner = EasyDict()
        if 'hook' not in cfg.policy.learn.learner:
            cfg.policy.learn.learner.hook = EasyDict()
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = args.save_ckpt_after_iter
    if getattr(args, "train_iterations", None) is not None:
        if 'learner' not in cfg.policy.learn:
            cfg.policy.learn.learner = EasyDict()
        cfg.policy.learn.learner.train_iterations = args.train_iterations
    # 收集经验的环境数量以及用于评估的环境数量
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # 收集经验的环境，使用并行环境管理器
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs, reward_mode=args.reward_mode) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    # 评估性能的环境，使用并行环境管理器
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs, reward_mode=args.reward_mode) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # 为mario环境设置种子
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    # 为torch、numpy、random等package设置种子
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # 采用DQN模型
    model = DQN(**cfg.policy.model)
    # 采用DQN策略
    policy = DQNPolicy(cfg.policy, model=model)
    
    # 初始化 SwanLab
    swanlab.init(
        experiment_name=cfg.exp_name,
        project="Mario-DQN-Experiment",
        config=cfg,
    )
    swanlab.sync_tensorboardX()
    # 设置学习、经验收集、评估、经验回放等强化学习常用配置
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # 设置epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # 训练以及评估
    while True:
        # 根据当前训练迭代数决定是否进行评估
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # 更新epsilon greedy信息
        eps = epsilon_greedy(collector.envstep)
        # 经验收集器从环境中收集经验
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        # 将收集的经验放入replay buffer
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # 采样经验进行训练
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
        if collector.envstep >= max_env_step:
            break
    
    swanlab.finish()


if __name__ == "__main__":
    from copy import deepcopy
    import argparse
    parser = argparse.ArgumentParser()
    # 种子
    parser.add_argument("--seed", "-s", type=int, default=0)
    # 游戏版本，v0 v1 v2 v3 四种选择
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    # 动作集合种类，包含[["right"], ["right", "A"]]、SIMPLE_MOVEMENT、COMPLEX_MOVEMENT，分别对应2、7、12个动作
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,12])
    # 观测空间叠帧数目，不叠帧或叠四帧
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])
    parser.add_argument("--reward_mode", type=str, default="dense", choices=["dense", "sparse"], help="dense 使用连续 shaping，sparse 仅通关/死亡奖励")
    parser.add_argument("--dueling_aggregator", type=str, default="mean", choices=["mean", "max", "lse"], help="对决网络的聚合方式，便于探索不同归一化策略")
    # 最大环境步数
    parser.add_argument("--max_env_step", type=int, default=int(3e6))
    # 运行时可覆盖的可选超参
    parser.add_argument("--collector_env_num", type=int, default=None)
    parser.add_argument("--evaluator_env_num", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--update_per_collect", type=int, default=None)
    parser.add_argument("--target_update_freq", type=int, default=None)
    parser.add_argument("--discount_factor", type=float, default=None)
    parser.add_argument("--nstep", type=int, default=None)
    parser.add_argument("--eps_start", type=float, default=None)
    parser.add_argument("--eps_end", type=float, default=None)
    parser.add_argument("--eps_decay", type=int, default=None)
    parser.add_argument("--replay_buffer_size", type=int, default=None)
    parser.add_argument("--save_ckpt_after_iter", type=int, default=None)
    parser.add_argument("--train_iterations", type=int, default=None)
    args = parser.parse_args()
    # 覆盖配置（仅当传入时）
    if args.collector_env_num is not None:
        mario_dqn_config.env.collector_env_num = args.collector_env_num
    if args.evaluator_env_num is not None:
        mario_dqn_config.env.evaluator_env_num = args.evaluator_env_num
    if args.learning_rate is not None:
        mario_dqn_config.policy.learn.learning_rate = args.learning_rate
    if args.batch_size is not None:
        mario_dqn_config.policy.learn.batch_size = args.batch_size
    if args.update_per_collect is not None:
        mario_dqn_config.policy.learn.update_per_collect = args.update_per_collect
    if args.target_update_freq is not None:
        mario_dqn_config.policy.learn.target_update_freq = args.target_update_freq
    if args.discount_factor is not None:
        mario_dqn_config.policy.discount_factor = args.discount_factor
    if args.nstep is not None:
        mario_dqn_config.policy.nstep = args.nstep
    if args.eps_start is not None:
        mario_dqn_config.policy.other.eps.start = args.eps_start
    if args.eps_end is not None:
        mario_dqn_config.policy.other.eps.end = args.eps_end
    if args.eps_decay is not None:
        mario_dqn_config.policy.other.eps.decay = args.eps_decay
    if args.replay_buffer_size is not None:
        mario_dqn_config.policy.other.replay_buffer.replay_buffer_size = args.replay_buffer_size
    # 对决网络聚合方式
    mario_dqn_config.policy.model.dueling_aggregator = args.dueling_aggregator
    # 训练相关 hook 的可选覆盖
    if args.save_ckpt_after_iter is not None:
        # 更密集或更稀疏地保存 ckpt
        if 'learner' not in mario_dqn_config.policy.learn:
            mario_dqn_config.policy.learn.learner = EasyDict()
        if 'hook' not in mario_dqn_config.policy.learn.learner:
            mario_dqn_config.policy.learn.learner.hook = EasyDict()
        mario_dqn_config.policy.learn.learner.hook.save_ckpt_after_iter = args.save_ckpt_after_iter
    if args.train_iterations is not None:
        if 'learner' not in mario_dqn_config.policy.learn:
            mario_dqn_config.policy.learn.learner = EasyDict()
        mario_dqn_config.policy.learn.learner.train_iterations = args.train_iterations
    mario_dqn_config.exp_name = 'exp/v'+str(args.version)+'_'+str(args.action)+'a_'+str(args.obs)+'f_seed'+str(args.seed)
    mario_dqn_config.policy.model.obs_shape=[args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape=args.action
    main(deepcopy(mario_dqn_config), args, seed=args.seed, max_env_step=args.max_env_step)