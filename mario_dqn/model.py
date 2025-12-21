"""
神经网络模型定义
"""
import copy
import math
from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze
from ding.model.common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead


class DQN(nn.Module):

    mode = ['compute_q', 'compute_q_logit']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
        dueling_aggregator: str = "mean",
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the DQN (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`dueling`): Whether choose ``DuelingHead`` or ``DiscreteHead(default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(DQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        # 自定义的 dueling 变体仅处理单一离散动作空间；多头场景沿用默认实现。
        if dueling and not multi_head:
            self.head = _DuelingQHead(
                input_dim=head_hidden_size,
                action_dim=action_shape,
                hidden_dim=head_hidden_size,
                activation=activation,
                aggregator=dueling_aggregator,
            )
        elif multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )


    def forward(self, x: torch.Tensor, mode: str='compute_q_logit') -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)


    def compute_q(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            DQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        Examples:
            >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


    def compute_q_logit(self, x: torch.Tensor) -> Dict:
        x = self.encoder(x)
        x = self.head(x)
        return x['logit']


class _DuelingQHead(nn.Module):
    """
    Dueling head with configurable aggregator (mean / max / log-sum-exp baseline).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        # 独立的 V 与 A 支路；使用 deepcopy 确保激活层不共享状态。
        act_v = copy.deepcopy(activation) if activation is not None else None
        act_a = copy.deepcopy(activation) if activation is not None else None

        value_layers = [nn.Linear(input_dim, hidden_dim)]
        if act_v is not None:
            value_layers.append(act_v)
        value_layers.append(nn.Linear(hidden_dim, 1))
        self.value = nn.Sequential(*value_layers)

        adv_layers = [nn.Linear(input_dim, hidden_dim)]
        if act_a is not None:
            adv_layers.append(act_a)
        adv_layers.append(nn.Linear(hidden_dim, action_dim))
        self.advantage = nn.Sequential(*adv_layers)

        self.aggregator = aggregator
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = self.value(x)
        advantage = self.advantage(x)

        if self.aggregator == "mean":
            centered_adv = advantage - advantage.mean(dim=1, keepdim=True)
        elif self.aggregator == "max":
            centered_adv = advantage - advantage.max(dim=1, keepdim=True)[0]
        elif self.aggregator == "lse":
            # log-sum-exp baseline is a smooth approximation to max; subtract log(action_dim) to keep scale comparable.
            baseline = torch.logsumexp(advantage, dim=1, keepdim=True) - math.log(float(self.action_dim))
            centered_adv = advantage - baseline
        else:
            raise ValueError(f"Unsupported dueling_aggregator: {self.aggregator}")

        logit = value + centered_adv
        return {"logit": logit, "action": logit.argmax(dim=1)}