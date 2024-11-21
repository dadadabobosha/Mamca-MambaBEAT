import torch
import torch.nn as nn
from functools import partial

# 导入必要的 Mamba 配置和工具
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import GenerationMixin
from .selective_SSM import MixerModel, _init_weights


# 定义去噪模块
class denosing_unit(nn.Module):
    def __init__(self, block, layers, in_channel=1, out_channel=16):
        super(denosing_unit, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layers = nn.Sequential(
            block(in_channel, out_channel),
            *[block(out_channel, out_channel) for _ in range(layers - 1)],
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 定义 MAMCA 模型
class MAMCA(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        length=1000,
        num_claasses=2,
        device=None,
        dtype=None,
        fused_add_norm=False,
    ):
        super().__init__()
        self.config = config
        self.fused_add_norm = False  # 添加 fused_add_norm 属性

        # CHANGE: Stronger regularization
        self.dropout = nn.Dropout(0.5)
        self.spec_augment = nn.Sequential(
            nn.Dropout(0.1), nn.Dropout2d(0.1)  # Channel dropout
        )

        # Rest of initialization remains the same
        d_model = config.d_model
        n_layer = config.n_layer
        ssm_cfg = config.ssm_cfg
        # rms_norm = config.rms_norm
        rms_norm = False  # 强制禁用 RMSNorm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        # Backbone
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            rms_norm=False,
            initializer_cfg=None,
            # fused_add_norm=fused_add_norm,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        # Denosing unit
        self.denosing = denosing_unit(
            BasicBlock, 2, in_channel=1, out_channel=config.d_model
        )

        # CHANGE: Two-stage classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(int(self.config.d_model * length), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_claasses),
        )

    # CHANGE new forward
    def forward(self, hidden_states, inference_params=None):
        if self.training:
            hidden_states = self.spec_augment(hidden_states)

        hidden_states = self.denosing(hidden_states)
        hidden_states = self.backbone(hidden_states, inference_params=inference_params)
        hidden_states = hidden_states.view(hidden_states.size(0), -1)
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)


# 初始化模型函数
def get_model(input_length, num_classes, device="cuda"):
    config = MambaConfig()
    # CHANGE: Adjusted model configuration
    config.d_model = 16  # 模型维度
    config.n_layer = 1  # 层数
    config.ssm_cfg = {  # SSM 参数
        "d_state": 16,  # changed to 16
        "d_conv": 4,
        "expand": 2,
    }

    # 实例化 MAMCA 模型
    model = MAMCA(
        config=config,
        length=input_length,
        num_claasses=num_classes,
        device=device,
        fused_add_norm=False,
    )
    return model.to(device)


# 初始化权重函数
def _init_weights(module, n_layer):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
