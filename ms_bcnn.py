import os
#修改好版本
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os, random, pickle, json, re
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.set_num_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Utility =========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def linear_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """线性插值重采样"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape
    if T == target_len:
        return arr
    src_x = np.linspace(0.0, 1.0, num=T, endpoint=True, dtype=np.float32)
    tgt_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True, dtype=np.float32)
    out = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(tgt_x, src_x, arr[:, d])
    return out


def decimate_repeat_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    间隔采样 + 重复采样
    - 下采样（T > target_len）：间隔采样
    - 上采样（T < target_len）：重复采样
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape

    if T == target_len:
        return arr
    elif T > target_len:
        # 下采样：间隔采样
        idx = np.round(np.linspace(0, T - 1, target_len)).astype(int)
        return arr[idx]
    else:
        # 上采样：重复采样（edge repeat）
        out = np.zeros((target_len, D), dtype=np.float32)
        if T == 1:
            # 只有1个点：重复整个序列
            out[:] = arr[0]
        else:
            # 分段重复
            segment_len = target_len // T
            remainder = target_len % T

            start_idx = 0
            for i in range(T):
                # 计算当前段的长度
                current_segment_len = segment_len + (1 if i < remainder else 0)
                end_idx = start_idx + current_segment_len
                # 重复当前点
                out[start_idx:end_idx] = arr[i]
                start_idx = end_idx
        return out


def hybrid_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    重复 + 线性插值混合重采样（优化版）
    - 下采样：先取L个原点，再做轻微平滑（3点移动平均）
    - 上采样：根据点数选择策略
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape

    if T == target_len:
        return arr
    elif T > target_len:
        # 下采样：先等距取L个原点，再轻微平滑
        idx = np.round(np.linspace(0, T - 1, target_len)).astype(int)
        out = arr[idx]

        # 轻微平滑（3点移动平均）
        if target_len >= 3:
            smoothed = np.zeros_like(out)
            # 中间点
            for i in range(1, target_len - 1):
                smoothed[i] = np.mean(out[i - 1:i + 2], axis=0)
            # 边界点（复制边界）
            smoothed[0] = out[0]
            smoothed[-1] = out[-1]
            return smoothed
        else:
            return out
    else:
        # 上采样：根据原始点数选择策略
        if T == 1:
            # 1个点：重复
            out = np.zeros((target_len, D), dtype=np.float32)
            out[:] = arr[0]
            return out
        elif T == 2:
            # 2个点：线性插值
            return linear_resample(arr, target_len)
        else:
            # 3个点以上：线性插值
            return linear_resample(arr, target_len)


def pool_resample(arr: np.ndarray, target_len: int, pool_type: str = 'avg') -> np.ndarray:
    """
    池化方式重采样（简化版）
    - 下采样：池化（平均/最大/中位数）
    - 上采样：重复采样
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape

    if T == target_len:
        return arr
    elif T > target_len:
        # 下采样：池化
        out = np.zeros((target_len, D), dtype=np.float32)

        # 计算段边界（保证每段至少有一个点）
        edges = np.linspace(0, T, target_len + 1).astype(int)
        # 确保边界不重复
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1

        for i in range(target_len):
            start, end = edges[i], edges[i + 1]
            # 简化的边界处理：如果段为空，直接取前一段的最后一个点
            if start >= end:
                if i > 0:
                    segment = arr[edges[i - 1]:edges[i]]
                else:
                    segment = arr[edges[1]:edges[2]]  # 第一段为空时取第二段
            else:
                segment = arr[start:end]

            if pool_type == 'avg':
                out[i] = np.mean(segment, axis=0)
            elif pool_type == 'max':
                out[i] = np.max(segment, axis=0)
            elif pool_type == 'median':
                out[i] = np.median(segment, axis=0)
            elif pool_type == 'weighted':
                # 三角加权池化
                seg_len = len(segment)
                if seg_len > 1:
                    weights = 1 - np.abs(np.linspace(-1, 1, seg_len))
                    weights = weights / np.sum(weights)
                    out[i] = np.sum(segment * weights[:, np.newaxis], axis=0)
                else:
                    out[i] = segment[0]

        return out
    else:
        # 上采样：重复采样
        return decimate_repeat_resample(arr, target_len)


def conv_blurpool_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    卷积重采样（抗混叠）
    - 下采样：先做低通卷积（抗混叠），再抽取到目标长度
    - 上采样：先插值到目标长度，再做小卷积平滑去"齿状"
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape

    if T == target_len:
        return arr
    elif T > target_len:
        # 下采样：低通卷积 + 抽取
        # 定义模糊核（BlurPool核：[1,4,6,4,1]/16）
        blur_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

        # 对每个特征维度进行卷积
        out_conv = np.zeros_like(arr)
        for d in range(D):
            # 使用反射填充边界
            padded = np.pad(arr[:, d], (2, 2), mode='reflect')
            # 应用卷积核
            conv_result = np.convolve(padded, blur_kernel, mode='valid')
            out_conv[:, d] = conv_result

        # 等距抽取目标长度
        idx = np.round(np.linspace(0, T - 1, target_len)).astype(int)
        return out_conv[idx]
    else:
        # 上采样：线性插值 + 小卷积平滑
        # 先线性插值到目标长度
        upsampled = linear_resample(arr, target_len)

        # 小卷积核平滑（3点移动平均）
        if target_len >= 3:
            smoothed = np.zeros_like(upsampled)
            # 中间点
            for i in range(1, target_len - 1):
                smoothed[i] = np.mean(upsampled[i - 1:i + 2], axis=0)
            # 边界点（复制边界）
            smoothed[0] = upsampled[0]
            smoothed[-1] = upsampled[-1]
            return smoothed
        else:
            return upsampled


def fir_lowpass_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    FIR低通重采样（窗化sinc核 + 抽取）
    - 下采样：设计窗化sinc低通核，卷积后抽取
    - 上采样：先插值，再用小FIR核平滑
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape

    if T == target_len:
        return arr
    elif T > target_len:
        # 下采样：FIR低通滤波 + 抽取

        # 计算抽取因子
        decimation_factor = T / target_len

        # 设计FIR低通滤波器（窗化sinc核）
        # 截止频率 ≈ 新采样率的一半
        cutoff_freq = 0.5 / decimation_factor

        # 滤波器长度（奇数）
        filter_length = min(31, T)
        if filter_length % 2 == 0:
            filter_length += 1

        # 生成sinc核
        n = np.arange(filter_length) - (filter_length - 1) / 2
        sinc_filter = np.sinc(2 * cutoff_freq * n)

        # 加汉宁窗
        window = np.hanning(filter_length)
        fir_kernel = sinc_filter * window

        # 归一化
        fir_kernel = fir_kernel / np.sum(fir_kernel)

        # 对每个特征维度进行卷积
        pad_len = filter_length // 2
        out_conv = np.zeros_like(arr)
        for d in range(D):
            padded = np.pad(arr[:, d], (pad_len, pad_len), mode='reflect')
            conv_result = np.convolve(padded, fir_kernel, mode='valid')
            out_conv[:, d] = conv_result

        # 等距抽取目标长度
        idx = np.round(np.linspace(0, T - 1, target_len)).astype(int)
        return out_conv[idx]
    else:
        # 上采样：线性插值 + FIR平滑
        # 先线性插值到目标长度
        upsampled = linear_resample(arr, target_len)

        # 设计小FIR核进行平滑
        filter_length = min(5, target_len)
        if filter_length % 2 == 0:
            filter_length += 1

        # 简单移动平均核
        fir_kernel = np.ones(filter_length) / filter_length

        # 对每个特征维度进行卷积
        pad_len = filter_length // 2
        out_conv = np.zeros_like(upsampled)
        for d in range(D):
            padded = np.pad(upsampled[:, d], (pad_len, pad_len), mode='reflect')
            conv_result = np.convolve(padded, fir_kernel, mode='valid')
            out_conv[:, d] = conv_result

        return out_conv


def pad_feature_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    T, D = arr.shape
    if D == target_dim:
        return arr
    if D > target_dim:
        return arr[:, :target_dim]
    pad = np.zeros((T, target_dim - D), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def train_val_test_split(data: list, train=0.7, val=0.15, test=0.15, seed=42, mode="random"):
    N = len(data)
    n_tr = int(N * train)
    n_va = int(N * val)
    n_te = N - n_tr - n_va

    # 确保每个集合至少有一个样本
    if n_tr == 0 or n_va == 0 or n_te == 0:
        n_tr = max(1, n_tr)
        n_va = max(1, n_va)
        n_te = max(1, n_te)
        # 重新调整总数
        total = n_tr + n_va + n_te
        if total > N:
            # 按比例缩减
            scale = N / total
            n_tr = int(n_tr * scale)
            n_va = int(n_va * scale)
            n_te = N - n_tr - n_va

    if mode == "random":
        idx = np.arange(N)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        train = [data[i] for i in tr_idx]
        val = [data[i] for i in va_idx]
        test = [data[i] for i in te_idx]
    elif mode == "sequential":
        train = data[:n_tr]
        val = data[n_tr:n_tr + n_va]
        test = data[n_tr + n_va:]
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return train, val, test


# ========= 多分位数处理工具函数 =========
def process_multi_quantile_output(yhat, y_true, cfg, is_training=True):
    """
    处理多分位数输出的工具函数
    Args:
        yhat: 模型输出
        y_true: 真实标签
        cfg: 配置对象
        is_training: 是否为训练阶段（影响是否需要扩展y_true）
    Returns:
        yhat_processed: 处理后的预测值（如果是多分位数，只取中位数）
        y_true_processed: 处理后的真实值（如果是多分位数且训练阶段，会扩展）
    """
    if cfg.loss_function == "multi_quantile":
        if is_training:
            # 训练阶段：扩展y_true以匹配多分位数输出的形状
            y_true_expanded = y_true.repeat(1, len(cfg.multi_quantiles))
        else:
            y_true_expanded = y_true

        # 提取中位数预测
        median_idx = cfg.multi_quantiles.index(0.5) if 0.5 in cfg.multi_quantiles else len(cfg.multi_quantiles) // 2
        yhat_median = yhat[:, median_idx * cfg.Lf:(median_idx + 1) * cfg.Lf]

        return yhat_median, y_true_expanded
    else:
        return yhat, y_true


def get_loss_inputs(yhat, y_true, cfg):
    """
    获取损失函数输入的工具函数
    Args:
        yhat: 模型输出
        y_true: 真实标签
        cfg: 配置对象
    Returns:
        loss_yhat: 用于损失计算的预测值
        loss_y_true: 用于损失计算的真实值
    """
    if cfg.loss_function == "multi_quantile":
        # 对于多分位数损失，直接使用完整输出和扩展的真实值
        y_true_expanded = y_true.repeat(1, len(cfg.multi_quantiles))
        return yhat, y_true_expanded
    else:
        return yhat, y_true


# ========= 新增损失函数 =========
class LogCoshLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true + self.eps)))


class QuantileLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        err = y_true - y_pred
        return torch.mean(torch.max((self.q - 1) * err, self.q * err))


class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        # y_pred shape: (batch, Lf * len(quantiles))
        # y_true shape: (batch, Lf)
        assert y_pred.shape[1] == y_true.shape[1] * len(self.quantiles)

        total_loss = 0
        for i, q in enumerate(self.quantiles):
            start_idx = i * y_true.shape[1]
            end_idx = (i + 1) * y_true.shape[1]
            q_pred = y_pred[:, start_idx:end_idx]
            err = y_true - q_pred
            q_loss = torch.mean(torch.max((q - 1) * err, q * err))
            total_loss += q_loss

        return total_loss / len(self.quantiles)


class SMAPELoss(nn.Module):
    """对称平均绝对百分比误差"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.mean(2.0 * torch.abs(y_pred - y_true) /
                          (torch.abs(y_pred) + torch.abs(y_true) + self.eps)) * 100.0


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, y_pred, y_true):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(y_pred, y_true)
        return total_loss


class AdaptiveCombinedLoss(nn.Module):
    """自适应组合损失，基于各损失函数的梯度大小调整权重"""

    def __init__(self, losses, initial_weights=None, adapt_rate=0.1):
        super().__init__()
        self.losses = losses
        self.adapt_rate = adapt_rate
        self.weights = nn.Parameter(torch.tensor(
            initial_weights if initial_weights is not None else [1.0] * len(losses),
            dtype=torch.float32
        ))

    def forward(self, y_pred, y_true):
        total_loss = 0
        loss_values = []

        # 计算各损失值
        for loss in self.losses:
            loss_val = loss(y_pred, y_true)
            loss_values.append(loss_val)
            total_loss += self.weights[len(loss_values) - 1] * loss_val

        # 自适应调整权重（简化版本）
        if self.training:
            with torch.no_grad():
                # 基于损失值的相对大小调整权重
                loss_tensor = torch.stack(loss_values)
                normalized_loss = loss_tensor / (loss_tensor.sum() + 1e-8)
                target_weights = 1.0 / (normalized_loss + 1e-8)
                target_weights = target_weights / target_weights.sum()

                # 平滑更新权重
                self.weights.data = (1 - self.adapt_rate) * self.weights + self.adapt_rate * target_weights

        return total_loss


# ========= 学习率调度器 =========
class WarmupCosineScheduler:
    """Warmup + Cosine衰减调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine衰减阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


# ========= 多头注意力融合层 =========
class MultiHeadAttentionFusion(nn.Module):
    """多头注意力融合层"""

    def __init__(self, feature_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim必须能被num_heads整除"

        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features_a, features_b):
        # 拼接特征
        batch_size = features_a.size(0)
        features = torch.stack([features_a, features_b], dim=1)  # (B, 2, D)

        # 线性变换
        Q = self.q_linear(features).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(features).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(features).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 2, self.feature_dim)

        # 输出变换
        output = self.out_linear(attn_output.sum(dim=1))  # 合并两个分支

        return output


@dataclass
class Config:
    # ===== 基础参数 =====
    seed: int = 42

    # ===== 数据参数 =====
    train_shuffle: bool = True
    Lp: int = 24
    Lf: int = 7
    batch_size: int = 64
    split_mode: str = "random"
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ===== 重采样方法选择 =====
    resample_method: str = "linear"  # "linear", "decimate_repeat", "hybrid", "pool", "conv_blurpool", "fir_lowpass"
    pool_type: str = "avg"  # "avg", "max", "median", "weighted" (当resample_method="pool"时使用)

    # ===== 训练参数 =====
    lr: float = 5e-4
    max_epochs: int = 500
    patience: int = 50
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ===== 模型选择 =====
    model_type: str = "ms_bcnn"  # "ms_bcnn"、"dual_cnn"、"simplified_cnn"

    # ===== 归一化层选择 =====
    norm_type: str = "batch_norm"  # "batch_norm", "layer_norm", "instance_norm"

    # ===== SimplifiedCNN 参数 =====
    simplified_conv1_filters: int = 16
    simplified_conv2_filters: int = 32
    simplified_conv3_filters: int = 64
    simplified_conv1_kernel: int = 3
    simplified_conv2_kernel: int = 3
    simplified_conv3_kernel: int = 5
    simplified_fc1_units: int = 128
    simplified_fc2_units: int = 64
    simplified_fc3_units: int = 32
    simplified_pool_kernel: int = 2
    simplified_activation: str = "relu"  # "relu" 或 "tanh"

    # ===== DualMultiScaleCNN 参数 =====
    dual_branch_filters: int = 16
    dual_branch_a_kernels: tuple = (3, 5, 7)
    dual_branch_b_kernels: tuple = (7, 9, 11)
    dual_fusion_units: int = 64
    dual_fc1_units: int = 64
    dual_fc2_units: int = 32
    dual_dropout: float = 0.3
    dual_activation: str = "relu"  # "relu" 或 "tanh"


    # ===== MS--BCNN 参数（本文方法） =====
    ms_c0: int = 16
    ms_channels: tuple = (16, 16, 64)
    ms_short_kernels: tuple = (3, 5, 7)
    ms_long_kernels: tuple = (9, 11, 13)
    ms_pool_kernel: int = 2
    ms_dropout: float = 0.1
    ms_activation: str = "gelu"  # "relu", "tanh", "gelu", "silu"
    ms_fc_units: tuple = (256, 16, 8)
    ms_head_dropout: float = 0.1

    # ===== 融合方式选择 =====
    fusion_method: str = "concat"  # "concat", "add", "weighted_sum", "attention", "gate", "cross_connection", "multi_head_attention"
    fusion_attention_dim: int = 32
    fusion_gate_dim: int = 32
    fusion_weighted_sum_init_a: float = 0.5  # 加权求和融合的初始权重A
    fusion_weighted_sum_init_b: float = 0.5  # 加权求和融合的初始权重B
    fusion_cross_connection_ratio: float = 0.5  # 交叉连接融合的比例
    fusion_multi_head_attention_heads: int = 4  # 多头注意力的头数
    fusion_final_fusion: str = "concat"  # 融合后的最终处理方式："concat" 拼接 / "add" 相加
    # ===== Cross-connection 专用参数 =====
    cross_output_dim: int = 32  # A->B 投影输出通道数
    cross_mode: str = "concat"  # "add", "concat", "gated"

    # ===== 损失函数选择 =====
    loss_function: str = "smooth_l1"  # "mse", "mae", "smooth_l1", "mape", "huber", "logcosh", "quantile", "multi_quantile", "combined", "smape", "adaptive_combined"
    smooth_l1_beta: float = 1.0
    mape_eps: float = 1e-8
    smape_eps: float = 1e-8
    huber_delta: float = 1.0
    logcosh_eps: float = 1e-12  #LogCosh损失参数（当 loss_function="logcosh" 时）  防止数值不稳定的小数值
    quantile_q: float = 0.5
    multi_quantiles: tuple = (0.1, 0.5, 0.9)

    # ===== 组合损失函数参数 =====
    combined_loss_weights: tuple = (0.7, 0.3)  # 用于combined损失
    combined_loss_types: tuple = ("mse", "mae")  # 用于combined损失
    adaptive_combined_adapt_rate: float = 0.1  # 自适应组合损失的学习率

    # ===== 学习率调度 =====
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine_warmup"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_mode: str = "min"
    warmup_epochs: int = 10  # warmup阶段轮数
    min_lr: float = 1e-6  # 最小学习率

    # ===== 兼容字段 =====
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2

    dim_feedforward: int = 256
    dropout: float = 0.5
    memory_tokens_short: int = 64
    memory_tokens_long: int = 16
    ar_lookback: int = 12


class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + self.eps))) * 100.0


class MinMaxScaler:
    def __init__(self, minv: np.ndarray, maxv: np.ndarray):
        minv = np.asarray(minv, dtype=np.float32)
        maxv = np.asarray(maxv, dtype=np.float32)
        # 过滤异常值，使用分位数代替极值
        minv_clean = np.nanpercentile(minv, 5) if np.isfinite(np.nanpercentile(minv, 5)) else 0.0
        maxv_clean = np.nanpercentile(maxv, 95) if np.isfinite(np.nanpercentile(maxv, 95)) else 1.0
        minv = np.where(np.isfinite(minv), minv, minv_clean)
        maxv = np.where(np.isfinite(maxv), maxv, maxv_clean)
        rng = maxv - minv
        rng[rng < 1e-6] = 1.0
        self.minv, self.rng = minv.astype(np.float32), rng.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = 2.0 * (X - self.minv) / self.rng - 1.0
        Z = np.clip(Z, -5.0, 5.0)
        return Z.astype(np.float32)


class LabelStd:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def norm(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def denorm(self, yhat):
        """同时支持numpy数组和torch张量"""
        if isinstance(yhat, torch.Tensor):
            return (yhat * torch.tensor(self.std, device=yhat.device) +
                    torch.tensor(self.mean, device=yhat.device)).to(yhat.dtype)
        else:
            return (yhat * self.std + self.mean).astype(np.float32)


class FurnaceDataset(Dataset):
    def __init__(self, data: list, Lp: int, Lf: int, H_dim: int, M_dim: int, L_dim: int,
                 scaler: MinMaxScaler = None, ystd: LabelStd = None, cfg: Config = None):
        self.data = data
        self.Lp = Lp
        self.Lf = Lf
        self.H_dim = H_dim
        self.M_dim = M_dim
        self.L_dim = L_dim
        self.in_dim = H_dim + M_dim + L_dim + 6  # +6 时间嵌入维度
        self.scaler = scaler
        self.ystd = ystd
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        def safe_to_numpy(arr):
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr

        H = safe_to_numpy(r["features_H"])
        M = safe_to_numpy(r["features_M"])
        L = safe_to_numpy(r["features_L"])
        y = np.array(r["labels"], dtype=np.float32).reshape(-1)

        # 根据配置选择重采样方法
        if self.cfg.resample_method == "linear":
            H_rs = pad_feature_dim(linear_resample(H, self.Lp), self.H_dim)
            M_rs = pad_feature_dim(linear_resample(M, self.Lp), self.M_dim)
            L_rs = pad_feature_dim(linear_resample(L, self.Lp), self.L_dim)
        elif self.cfg.resample_method == "decimate_repeat":
            H_rs = pad_feature_dim(decimate_repeat_resample(H, self.Lp), self.H_dim)
            M_rs = pad_feature_dim(decimate_repeat_resample(M, self.Lp), self.M_dim)
            L_rs = pad_feature_dim(decimate_repeat_resample(L, self.Lp), self.L_dim)
        elif self.cfg.resample_method == "hybrid":
            H_rs = pad_feature_dim(hybrid_resample(H, self.Lp), self.H_dim)
            M_rs = pad_feature_dim(hybrid_resample(M, self.Lp), self.M_dim)
            L_rs = pad_feature_dim(hybrid_resample(L, self.Lp), self.L_dim)
        elif self.cfg.resample_method == "pool":
            H_rs = pad_feature_dim(pool_resample(H, self.Lp, self.cfg.pool_type), self.H_dim)
            M_rs = pad_feature_dim(pool_resample(M, self.Lp, self.cfg.pool_type), self.M_dim)
            L_rs = pad_feature_dim(pool_resample(L, self.Lp, self.cfg.pool_type), self.L_dim)
        elif self.cfg.resample_method == "conv_blurpool":
            H_rs = pad_feature_dim(conv_blurpool_resample(H, self.Lp), self.H_dim)
            M_rs = pad_feature_dim(conv_blurpool_resample(M, self.Lp), self.M_dim)
            L_rs = pad_feature_dim(conv_blurpool_resample(L, self.Lp), self.L_dim)
        elif self.cfg.resample_method == "fir_lowpass":
            H_rs = pad_feature_dim(fir_lowpass_resample(H, self.Lp), self.H_dim)
            M_rs = pad_feature_dim(fir_lowpass_resample(M, self.Lp), self.M_dim)
            L_rs = pad_feature_dim(fir_lowpass_resample(L, self.Lp), self.L_dim)
        else:
            raise ValueError(f"未知的重采样方法: {self.cfg.resample_method}")

        X = np.concatenate([H_rs, M_rs, L_rs], axis=1).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # ===== 时间嵌入部分 =====
        if "sample_info" in r:
            info_str = r["sample_info"]
            try:
                # 提取月份
                month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,
                             "Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
                for k,v in month_map.items():
                    if k in info_str:
                        month = v
                        break
                else:
                    month = 6  # 默认6月

                # 提取日期数字
                day_match = re.search(r'(\d+)', info_str)
                day = int(day_match.group(1)) if day_match else 1

                # 提取Sheet号
                sheet_match = re.search(r'Sheet(\d+)', info_str)
                sheet_num = int(sheet_match.group(1)) if sheet_match else 0
                hour = sheet_num * 2
                hour = min(hour, 23)

                # 周期性时间嵌入
                time_vec = np.array([
                    np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                    np.sin(2*np.pi*day/31), np.cos(2*np.pi*day/31),
                    np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
                ], dtype=np.float32)

            except Exception as e:
                print(f"[Warn] 时间解析失败: {info_str}, {e}")
                time_vec = np.zeros(6, dtype=np.float32)
        else:
            time_vec = np.zeros(6, dtype=np.float32)

        # 将时间特征广播到每个时间步
        time_expand = np.tile(time_vec, (self.Lp, 1))
        # 拼接到输入特征
        X = np.concatenate([X, time_expand], axis=1)

        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        y_norm = self.ystd.norm(y) if self.ystd is not None else y

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)


# ========= 归一化层工厂函数 =========
def create_norm_layer(norm_type, num_features, sequence_length=None):
    """创建归一化层"""
    if norm_type == "batch_norm":
        return nn.BatchNorm1d(num_features)
    elif norm_type == "layer_norm":
        # 对于LayerNorm，我们使用GroupNorm(1, num_features)来替代，更稳定
        return nn.GroupNorm(1, num_features)
    elif norm_type == "instance_norm":
        return nn.InstanceNorm1d(num_features)
    else:
        raise ValueError(f"不支持的归一化类型: {norm_type}")


class SimplifiedEvolvedCNN(nn.Module):
    """简化版进化CNN（训练更稳）"""

    def __init__(self, in_dim: int, Lp: int, Lf: int, cfg: Config):
        super().__init__()

        # 激活函数选择
        if cfg.simplified_activation == "tanh":
            activation = nn.Tanh()
        else:
            activation = nn.ReLU()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_dim, cfg.simplified_conv1_filters, kernel_size=cfg.simplified_conv1_kernel,
                      padding=cfg.simplified_conv1_kernel // 2),
            activation,
            create_norm_layer(cfg.norm_type, cfg.simplified_conv1_filters),
            nn.AvgPool1d(kernel_size=cfg.simplified_pool_kernel),
            nn.Conv1d(cfg.simplified_conv1_filters, cfg.simplified_conv2_filters,
                      kernel_size=cfg.simplified_conv2_kernel, padding=cfg.simplified_conv2_kernel // 2),
            activation,
            create_norm_layer(cfg.norm_type, cfg.simplified_conv2_filters),
            nn.Conv1d(cfg.simplified_conv2_filters, cfg.simplified_conv3_filters,
                      kernel_size=cfg.simplified_conv3_kernel, padding=cfg.simplified_conv3_kernel // 2),
            activation,
            create_norm_layer(cfg.norm_type, cfg.simplified_conv3_filters),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.simplified_conv3_filters, cfg.simplified_fc1_units),
            activation,
            nn.Linear(cfg.simplified_fc1_units, cfg.simplified_fc2_units),
            activation,
            nn.Linear(cfg.simplified_fc2_units, cfg.simplified_fc3_units),
            activation,
        )
        self.output = nn.Linear(cfg.simplified_fc3_units, Lf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return self.output(x)


class DualMultiScaleCNN(nn.Module):
    """
    改良版 Dual CNN
    - 每条分支内部用多尺度卷积并联，提取不同感受野特征
    - 使用 BatchNorm/LayerNorm/InstanceNorm 稳定训练
    - 使用 Dropout 增强正则化
    - 支持多种融合方式
    """

    def __init__(self, in_dim: int, Lp: int, Lf: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.Lf = Lf

        # 激活函数选择
        if cfg.dual_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # ========== 分支A（局部特征为主，小卷积核） ==========
        self.branch_a = nn.ModuleList([
            nn.Conv1d(in_dim, cfg.dual_branch_filters, kernel_size=k, padding=k // 2)
            for k in cfg.dual_branch_a_kernels
        ])
        # 使用GroupNorm替代LayerNorm，更稳定
        self.bn_a = create_norm_layer(cfg.norm_type, len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters)
        self.pool_a = nn.AdaptiveAvgPool1d(1)

        # ========== 分支B（全局特征为主，更大卷积核） ==========
        self.branch_b = nn.ModuleList([
            nn.Conv1d(in_dim, cfg.dual_branch_filters, kernel_size=k, padding=k // 2)
            for k in cfg.dual_branch_b_kernels
        ])
        self.bn_b = create_norm_layer(cfg.norm_type, len(cfg.dual_branch_b_kernels) * cfg.dual_branch_filters)
        self.pool_b = nn.AdaptiveAvgPool1d(1)
        # 如果 cross_mode = add，则强制 cross_output_dim 等于 B 分支通道数
        if cfg.fusion_method == "cross_connection" and cfg.cross_mode == "add":
            expected_dim = len(cfg.dual_branch_b_kernels) * cfg.dual_branch_filters
            if cfg.cross_output_dim != expected_dim:
                print(f"[提示] add 模式下 cross_output_dim 自动调整为 {expected_dim}")
                cfg.cross_output_dim = expected_dim

        # ========== 融合层 ==========
        total_filters = len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters + len(
            cfg.dual_branch_b_kernels) * cfg.dual_branch_filters

        if cfg.fusion_method == "concat":
            # 原始concat方式
            self.fuse = nn.Linear(total_filters, cfg.dual_fusion_units)
        elif cfg.fusion_method == "add":
            # 简单相加融合
            self.fuse = nn.Linear(len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters, cfg.dual_fusion_units)
        elif cfg.fusion_method == "weighted_sum":
            # 加权求和融合
            self.weight_a = nn.Parameter(torch.tensor(cfg.fusion_weighted_sum_init_a))
            self.weight_b = nn.Parameter(torch.tensor(cfg.fusion_weighted_sum_init_b))
            # 使用concat或add方式
            if cfg.fusion_final_fusion == "concat":
                self.fuse = nn.Linear(total_filters, cfg.dual_fusion_units)
            else:  # add
                self.fuse = nn.Linear(len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters, cfg.dual_fusion_units)
        elif cfg.fusion_method == "attention":
            # 注意力融合
            self.attention = nn.Sequential(
                nn.Linear(total_filters, cfg.fusion_attention_dim),
                self.activation,
                nn.Linear(cfg.fusion_attention_dim, 2),
                nn.Softmax(dim=1)
            )
            # 使用concat或add方式
            if cfg.fusion_final_fusion == "concat":
                self.fuse = nn.Linear(total_filters, cfg.dual_fusion_units)
            else:  # add
                self.fuse = nn.Linear(len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters, cfg.dual_fusion_units)
        elif cfg.fusion_method == "gate":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(total_filters, cfg.fusion_gate_dim),
                self.activation,
                nn.Linear(cfg.fusion_gate_dim, 1),
                nn.Sigmoid()
            )
            # 使用concat或add方式
            if cfg.fusion_final_fusion == "concat":
                self.fuse = nn.Linear(total_filters, cfg.dual_fusion_units)
            else:  # add
                self.fuse = nn.Linear(len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters, cfg.dual_fusion_units)
        elif cfg.fusion_method == "cross_connection":
            # 交叉连接 - 在卷积层之间添加连接
            self.cross_mode = cfg.cross_mode
            self.cross_conv = nn.Conv1d(
                cfg.dual_branch_filters * len(cfg.dual_branch_a_kernels),
                cfg.cross_output_dim,
                kernel_size=1
            )
            if self.cross_mode == "gated":
                self.gate_conv = nn.Conv1d(
                    cfg.dual_branch_filters * len(cfg.dual_branch_a_kernels),
                    cfg.cross_output_dim,
                    kernel_size=1
                )
            # 注意 concat 会改变维度
            if self.cross_mode == "concat":
                fusion_in_dim = len(cfg.dual_branch_a_kernels) * cfg.dual_branch_filters + cfg.cross_output_dim
            else:
                fusion_in_dim = len(cfg.dual_branch_b_kernels) * cfg.dual_branch_filters  # 使用B分支的维度
            self.fuse = nn.Linear(fusion_in_dim, cfg.dual_fusion_units)
        elif cfg.fusion_method == "multi_head_attention":
            # 多头注意力融合
            self.multi_head_attn = MultiHeadAttentionFusion(
                feature_dim=cfg.dual_branch_filters * len(cfg.dual_branch_a_kernels),
                num_heads=cfg.fusion_multi_head_attention_heads,
                dropout=cfg.dual_dropout
            )
            self.fuse = nn.Linear(cfg.dual_branch_filters * len(cfg.dual_branch_a_kernels), cfg.dual_fusion_units)
        else:
            raise ValueError(f"未知的融合方式: {cfg.fusion_method}")

        # ========== 全连接层 ==========
        self.fc_layers = nn.Sequential(
            nn.Linear(cfg.dual_fusion_units, cfg.dual_fc1_units),
            self.activation,
            nn.Dropout(cfg.dual_dropout),
            nn.Linear(cfg.dual_fc1_units, cfg.dual_fc2_units),
            self.activation
        )

        # ========== 输出层 ==========
        # 对于多分位数损失，输出维度需要调整
        if cfg.loss_function == "multi_quantile":
            self.output = nn.Linear(cfg.dual_fc2_units, Lf * len(cfg.multi_quantiles))
        else:
            self.output = nn.Linear(cfg.dual_fc2_units, Lf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, in_dim, Lp)

        # 分支A
        out_a_list = [self.activation(conv(x)) for conv in self.branch_a]
        out_a = torch.cat(out_a_list, dim=1)  # (B, 48, Lp)

        # 使用GroupNorm，不需要转置
        out_a = self.bn_a(out_a)
        out_a_pooled = self.pool_a(out_a).squeeze(-1)  # (B, 48)

        # 分支B
        out_b_list = [self.activation(conv(x)) for conv in self.branch_b]
        out_b = torch.cat(out_b_list, dim=1)  # (B, 48, Lp)
        out_b = self.bn_b(out_b)
        out_b_pooled = self.pool_b(out_b).squeeze(-1)  # (B, 48)

        # 融合策略
        if self.cfg.fusion_method == "concat":
            # 原始concat方式
            out = torch.cat([out_a_pooled, out_b_pooled], dim=1)  # (B, 96)
            out = self.fuse(out)

        elif self.cfg.fusion_method == "add":
            # 简单相加融合
            out = out_a_pooled + out_b_pooled
            out = self.fuse(out)

        elif self.cfg.fusion_method == "weighted_sum":
            # 加权求和融合
            weights = torch.softmax(torch.stack([self.weight_a, self.weight_b]), dim=0)
            weighted_out = weights[0] * out_a_pooled + weights[1] * out_b_pooled

            if self.cfg.fusion_final_fusion == "concat":
                # 使用concat方式
                out = torch.cat([out_a_pooled, out_b_pooled], dim=1)
                out = self.fuse(out)
            else:  # add
                out = self.fuse(weighted_out)

        elif self.cfg.fusion_method == "attention":
            # 注意力融合
            combined = torch.cat([out_a_pooled, out_b_pooled], dim=1)
            attn_weights = self.attention(combined)  # (B, 2)
            attn_out = attn_weights[:, 0:1] * out_a_pooled + attn_weights[:, 1:2] * out_b_pooled

            if self.cfg.fusion_final_fusion == "concat":
                # 使用concat方式
                out = torch.cat([out_a_pooled, out_b_pooled], dim=1)
                out = self.fuse(out)
            else:  # add
                out = self.fuse(attn_out)

        elif self.cfg.fusion_method == "gate":
            # 门控融合
            combined = torch.cat([out_a_pooled, out_b_pooled], dim=1)
            gate = self.gate(combined)  # (B, 1)
            gate_out = gate * out_a_pooled + (1 - gate) * out_b_pooled

            if self.cfg.fusion_final_fusion == "concat":
                # 使用concat方式
                out = torch.cat([out_a_pooled, out_b_pooled], dim=1)
                out = self.fuse(out)
            else:  # add
                out = self.fuse(gate_out)

        elif self.cfg.fusion_method == "cross_connection":
            cross_feat = self.cross_conv(out_a)

            if self.cross_mode == "add":
                if cross_feat.shape[1] != out_b.shape[1]:
                    raise ValueError(
                        f"add 模式要求 cross_output_dim == {out_b.shape[1]}, 但得到了 {cross_feat.shape[1]}"
                    )
                out_b_enhanced = out_b + cross_feat

            elif self.cross_mode == "concat":
                out_b_enhanced = torch.cat([out_b, cross_feat], dim=1)

            elif self.cross_mode == "gated":
                gate = torch.sigmoid(self.gate_conv(out_a))
                out_b_enhanced = gate * cross_feat + (1 - gate) * out_b

            else:
                raise ValueError(f"未知的 cross_mode: {self.cross_mode}")

            out_b_enhanced_pooled = self.pool_b(out_b_enhanced).squeeze(-1)

            if self.cross_mode == "concat":
                out = torch.cat([out_a_pooled, out_b_enhanced_pooled], dim=1)
            else:  # add 或 gated
                out = out_b_enhanced_pooled  # 使用增强后的B分支结果

            out = self.fuse(out)

        elif self.cfg.fusion_method == "multi_head_attention":
            # 多头注意力融合
            out = self.multi_head_attn(out_a_pooled, out_b_pooled)
            out = self.fuse(out)

        # 全连接
        out = self.fc_layers(out)
        return self.output(out)


# ===== Metrics (含每标签) =====
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """按论文 Appendix B.5 计算指标。

    记测试集（或评估集）有 N 个样本、K 个目标：
    - 每目标：MSE_k, MAE_k, MAPE_k
    - 总体：NMSE = (1/K)\sum_k MSE_k/(\sigma_k^2+\varepsilon),
            NMAE = (1/K)\sum_k MAE_k/(\sigma_k+\varepsilon),
      其中 \sigma_k 为评估集上第 k 个目标的真值标准差。
    """
    eps = 1e-8

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # 每目标误差
    mse_per = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per = np.mean(np.abs(y_true - y_pred), axis=0)
    # MAPE_k = 100/N * sum_n |(y - yhat)/(y + eps)|
    mape_per = np.mean(np.abs((y_true - y_pred) / (y_true + eps)), axis=0) * 100.0

    # 归一化尺度：sigma_k（评估集真值标准差）
    sigma = np.std(y_true, axis=0, ddof=0)

    nmse = float(np.mean(mse_per / (sigma ** 2 + eps)))
    nmae = float(np.mean(mae_per / (sigma + eps)))
    mape_mean = float(np.mean(mape_per))

    return {
        "nmse": nmse,
        "nmae": nmae,
        "mape": mape_mean,  # 这里返回 MAPEmean，与论文中的 macro-average 一致
        "mse_per_label": mse_per.astype(float).tolist(),
        "mae_per_label": mae_per.astype(float).tolist(),
        "mape_per_label": mape_per.astype(float).tolist(),
        "sigma_per_label": sigma.astype(float).tolist(),
    }


# ========== 训练辅助函数 ==========
def prepare_datasets(cfg, dataset):
    """准备数据集"""
    # 统计维度
    H_dims, M_dims, L_dims, y_lens = [], [], [], []
    for r in dataset:
        H = np.array(r["features_H"], dtype=np.float32)
        H_dims.append(H.shape[1] if H.ndim == 2 else 1)
        M = np.array(r["features_M"], dtype=np.float32)
        M_dims.append(M.shape[1] if M.ndim == 2 else 1)
        L = np.array(r["features_L"], dtype=np.float32)
        L_dims.append(L.shape[1] if L.ndim == 2 else 1)
        y = np.array(r["labels"], dtype=np.float32)
        y_lens.append(len(y))

    H_dim = int(np.max(H_dims))
    M_dim = int(np.max(M_dims))
    L_dim = int(np.max(L_dims))

    if cfg.Lf != int(np.median(y_lens)):
        cfg.Lf = int(np.median(y_lens))

    print(f"数据维度 - H: {H_dim}, M: {M_dim}, L: {L_dim}, 输出: {cfg.Lf}")
    print(f"重采样方法: {cfg.resample_method}" + (
        f" (池化类型: {cfg.pool_type})" if cfg.resample_method == "pool" else ""))

    # 划分数据集
    tr, va, te = train_val_test_split(
        dataset,
        train=cfg.train_ratio,
        val=cfg.val_ratio,
        test=cfg.test_ratio,
        seed=cfg.split_seed,
        mode=cfg.split_mode
    )
    print(f"数据集大小 - 训练: {len(tr)}, 验证: {len(va)}, 测试: {len(te)}")

    # 构建 scaler / ystd（基于训练集）
    feats, ys = [], []
    for r in tr:
        H = np.array(r["features_H"], dtype=np.float32)
        # 使用配置的重采样方法
        if cfg.resample_method == "linear":
            H_rs = pad_feature_dim(linear_resample(H, cfg.Lp), H_dim)
        elif cfg.resample_method == "decimate_repeat":
            H_rs = pad_feature_dim(decimate_repeat_resample(H, cfg.Lp), H_dim)
        elif cfg.resample_method == "hybrid":
            H_rs = pad_feature_dim(hybrid_resample(H, cfg.Lp), H_dim)
        elif cfg.resample_method == "pool":
            H_rs = pad_feature_dim(pool_resample(H, cfg.Lp, cfg.pool_type), H_dim)
        elif cfg.resample_method == "conv_blurpool":
            H_rs = pad_feature_dim(conv_blurpool_resample(H, cfg.Lp), H_dim)
        elif cfg.resample_method == "fir_lowpass":
            H_rs = pad_feature_dim(fir_lowpass_resample(H, cfg.Lp), H_dim)

        M = np.array(r["features_M"], dtype=np.float32)
        if cfg.resample_method == "linear":
            M_rs = pad_feature_dim(linear_resample(M, cfg.Lp), M_dim)
        elif cfg.resample_method == "decimate_repeat":
            M_rs = pad_feature_dim(decimate_repeat_resample(M, cfg.Lp), M_dim)
        elif cfg.resample_method == "hybrid":
            M_rs = pad_feature_dim(hybrid_resample(M, cfg.Lp), M_dim)
        elif cfg.resample_method == "pool":
            M_rs = pad_feature_dim(pool_resample(M, cfg.Lp, cfg.pool_type), M_dim)
        elif cfg.resample_method == "conv_blurpool":
            M_rs = pad_feature_dim(conv_blurpool_resample(M, cfg.Lp), M_dim)
        elif cfg.resample_method == "fir_lowpass":
            M_rs = pad_feature_dim(fir_lowpass_resample(M, cfg.Lp), M_dim)

        L = np.array(r["features_L"], dtype=np.float32)
        if cfg.resample_method == "linear":
            L_rs = pad_feature_dim(linear_resample(L, cfg.Lp), L_dim)
        elif cfg.resample_method == "decimate_repeat":
            L_rs = pad_feature_dim(decimate_repeat_resample(L, cfg.Lp), L_dim)
        elif cfg.resample_method == "hybrid":
            L_rs = pad_feature_dim(hybrid_resample(L, cfg.Lp), L_dim)
        elif cfg.resample_method == "pool":
            L_rs = pad_feature_dim(pool_resample(L, cfg.Lp, cfg.pool_type), L_dim)
        elif cfg.resample_method == "conv_blurpool":
            L_rs = pad_feature_dim(conv_blurpool_resample(L, cfg.Lp), L_dim)
        elif cfg.resample_method == "fir_lowpass":
            L_rs = pad_feature_dim(fir_lowpass_resample(L, cfg.Lp), L_dim)

        X = np.concatenate([H_rs, M_rs, L_rs], axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        feats.append(X)
        ys.append(np.array(r["labels"], dtype=np.float32).reshape(-1))
    feats = np.stack(feats, axis=0)
    minv = np.nanmin(feats, axis=(0, 1))
    maxv = np.nanmax(feats, axis=(0, 1))
    scaler = MinMaxScaler(minv, maxv)

    ys = np.stack(ys, axis=0)
    y_mean = np.nanmean(ys, axis=0)
    y_std = np.nanstd(ys, axis=0)
    ystd = LabelStd(y_mean, y_std)

    # 数据集/加载器
    train_ds = FurnaceDataset(tr, cfg.Lp, cfg.Lf, H_dim, M_dim, L_dim, scaler, ystd, cfg)
    val_ds = FurnaceDataset(va, cfg.Lp, cfg.Lf, H_dim, M_dim, L_dim, scaler, ystd, cfg)
    test_ds = FurnaceDataset(te, cfg.Lp, cfg.Lf, H_dim, M_dim, L_dim, scaler, ystd, cfg)

    return train_ds, val_ds, test_ds, scaler, ystd, H_dim, M_dim, L_dim




# ========= MS--BCNN（本文方法） =========
class _MSBCNNBlock(nn.Module):
    """单层多尺度双分支卷积块：短核分支与长核分支并行，分别建模局部与长程模式。"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_short: int,
        k_long: int,
        norm_type: str,
        activation: nn.Module,
        dropout: float,
    ):
        super().__init__()
        pad_s = k_short // 2
        pad_l = k_long // 2

        self.conv_s = nn.Conv1d(in_ch, out_ch, kernel_size=k_short, padding=pad_s, bias=True)
        self.conv_l = nn.Conv1d(in_ch, out_ch, kernel_size=k_long, padding=pad_l, bias=True)

        self.norm_s = create_norm_layer(norm_type, out_ch)
        self.norm_l = create_norm_layer(norm_type, out_ch)

        self.act = activation
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, hs: torch.Tensor, hl: torch.Tensor):
        hs = self.drop(self.act(self.norm_s(self.conv_s(hs))))
        hl = self.drop(self.act(self.norm_l(self.conv_l(hl))))
        return hs, hl


class MSBCNN(nn.Module):
    """
    MS--BCNN：可配置的多尺度卷积预测骨干（与论文保持一致）。
    - 输入：对齐后的多源序列拼接特征 X ∈ R^{B×Lp×Din}
    - 主干：三层双分支卷积（短核与长核），中间可配置池化
    - 融合：FuseConv，可选 concat、add、weighted_sum、gate、attention、cross_connection
    - 预测头：MLP 输出多标签目标（支持 multi-quantile 输出）
    """
    def __init__(self, in_dim: int, Lp: int, cfg: Config):
        super().__init__()
        self.in_dim = int(in_dim)
        self.Lp = int(Lp)
        self.Lf = int(cfg.Lf)
        self.cfg = cfg

        # ===== 配置解析（默认与论文 Appendix 的设置一致，可在 Config 中覆盖）=====
        c0 = int(getattr(cfg, "ms_c0", 16))
        chs = tuple(getattr(cfg, "ms_channels", (16, 16, 64)))
        ks  = tuple(getattr(cfg, "ms_short_kernels", (3, 5, 7)))
        kl  = tuple(getattr(cfg, "ms_long_kernels", (9, 11, 13)))
        pool_k = int(getattr(cfg, "ms_pool_kernel", 2))
        drop = float(getattr(cfg, "ms_dropout", 0.1))
        act_name = str(getattr(cfg, "ms_activation", "gelu")).lower()
        head_units = tuple(getattr(cfg, "ms_fc_units", (256, 16, 8)))
        head_drop = float(getattr(cfg, "ms_head_dropout", 0.1))

        if len(chs) != 3 or len(ks) != 3 or len(kl) != 3:
            raise ValueError("MS--BCNN 需要 3 层配置：ms_channels、ms_short_kernels、ms_long_kernels 的长度均应为 3。")

        # ===== 激活函数 =====
        if act_name == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_name == "tanh":
            activation = nn.Tanh()
        elif act_name == "gelu":
            activation = nn.GELU()
        elif act_name == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {act_name}")

        # ===== 输入投影到 C0（等价于逐时刻线性层）=====
        self.proj = nn.Conv1d(self.in_dim, c0, kernel_size=1, bias=True)
        self.proj_norm = create_norm_layer(cfg.norm_type, c0)
        self.proj_act = activation
        self.proj_drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

        # ===== 三层双分支卷积 =====
        self.block1 = _MSBCNNBlock(c0, chs[0], ks[0], kl[0], cfg.norm_type, activation, drop)
        self.block2 = _MSBCNNBlock(chs[0], chs[1], ks[1], kl[1], cfg.norm_type, activation, drop)
        self.block3 = _MSBCNNBlock(chs[1], chs[2], ks[2], kl[2], cfg.norm_type, activation, drop)

        self.pool = nn.AvgPool1d(kernel_size=pool_k, stride=pool_k) if pool_k and pool_k > 1 else nn.Identity()
        self.adapool = nn.AdaptiveAvgPool1d(1)

        # ===== 融合（FuseConv）=====
        self.fusion_method = str(getattr(cfg, "fusion_method", "gate")).lower()

        Cf = int(chs[2])
        self.weight_mode = str(getattr(cfg, "fusion_final_fusion", "add")).lower()   # 对应 x23
        self.cross_mode = str(getattr(cfg, "cross_mode", "add")).lower()            # 对应 x24

        # 仅在对应融合方式下创建参数，避免“未使用模块”抬高参数量统计
        self.gate_conv = None
        self.weight_fc = None
        self.attn_q = self.attn_k = self.attn_v = self.attn_out = None
        self.cross_s2l = self.cross_l2s = self.cross_gate = None
        cross_out = int(getattr(cfg, "cross_output_dim", Cf))

        if self.fusion_method == "gate":
            self.gate_conv = nn.Conv1d(2 * Cf, Cf, kernel_size=1, bias=True)

        elif self.fusion_method == "weighted_sum":
            self.weight_fc = nn.Linear(2 * Cf, 1, bias=True)

        elif self.fusion_method == "attention":
            attn_dim = int(getattr(cfg, "fusion_attention_dim", Cf))
            self.attn_q = nn.Linear(Cf, attn_dim, bias=True)
            self.attn_k = nn.Linear(Cf, attn_dim, bias=True)
            self.attn_v = nn.Linear(Cf, attn_dim, bias=True)
            self.attn_out = nn.Linear(attn_dim, Cf, bias=True)

        elif self.fusion_method == "cross_connection":
            self.cross_s2l = nn.Conv1d(Cf, cross_out, kernel_size=1, bias=True)
            self.cross_l2s = nn.Conv1d(Cf, cross_out, kernel_size=1, bias=True)
            if self.cross_mode == "gated":
                self.cross_gate = nn.Conv1d(2 * cross_out, cross_out, kernel_size=1, bias=True)

        fused_C = self._infer_fused_channels(Cf, cross_out)
        self.fused_C = int(fused_C)

        # ===== 预测头（MLP）=====
        h1, h2, h3 = (int(head_units[0]), int(head_units[1]), int(head_units[2]))
        self.head = nn.Sequential(
            nn.Linear(self.fused_C, h1),
            activation,
            nn.Dropout(head_drop) if head_drop and head_drop > 0 else nn.Identity(),
            nn.Linear(h1, h2),
            activation,
            nn.Dropout(head_drop) if head_drop and head_drop > 0 else nn.Identity(),
            nn.Linear(h2, h3),
            activation,
            nn.Dropout(head_drop) if head_drop and head_drop > 0 else nn.Identity(),
        )

        if getattr(cfg, "loss_function", "") == "multi_quantile":
            out_dim = self.Lf * len(getattr(cfg, "multi_quantiles", (0.1, 0.5, 0.9)))
        else:
            out_dim = self.Lf
        self.out = nn.Linear(h3, int(out_dim))

    def _infer_fused_channels(self, Cf: int, cross_out: int) -> int:
        fm = self.fusion_method
        if fm == "concat":
            return 2 * Cf
        if fm in {"add", "gate", "attention"}:
            return Cf
        if fm == "weighted_sum":
            return 2 * Cf if self.weight_mode == "concat" else Cf
        if fm == "cross_connection":
            return 2 * cross_out if self.cross_mode == "concat" else cross_out
        return Cf

    def _fuse(self, hs: torch.Tensor, hl: torch.Tensor) -> torch.Tensor:
        fm = self.fusion_method

        if fm == "concat":
            return torch.cat([hs, hl], dim=1)

        if fm == "add":
            return hs + hl

        if fm == "weighted_sum":
            if self.weight_fc is None:
                raise RuntimeError("weighted_sum 融合缺少 weight_fc。请检查 fusion_method 设置。")
            z = torch.cat([hs, hl], dim=1).mean(dim=2)                # (B, 2Cf)
            alpha = torch.sigmoid(self.weight_fc(z)).view(-1, 1, 1)   # (B, 1, 1)
            if self.weight_mode == "concat":
                return torch.cat([alpha * hs, (1.0 - alpha) * hl], dim=1)
            return alpha * hs + (1.0 - alpha) * hl

        if fm == "gate":
            if self.gate_conv is None:
                raise RuntimeError("gate 融合缺少 gate_conv。请检查 fusion_method 设置。")
            g = torch.sigmoid(self.gate_conv(torch.cat([hs, hl], dim=1)))
            return g * hs + (1.0 - g) * hl

        if fm == "attention":
            if self.attn_q is None:
                raise RuntimeError("attention 融合缺少注意力参数。请检查 fusion_method 设置。")
            hs_t = hs.transpose(1, 2)  # (B, L, C)
            hl_t = hl.transpose(1, 2)
            Q = self.attn_q(hs_t)
            K = self.attn_k(hl_t)
            V = self.attn_v(hl_t)
            attn = torch.softmax((Q @ K.transpose(1, 2)) / math.sqrt(Q.size(-1)), dim=-1)
            out = attn @ V
            out = self.attn_out(out)   # (B, L, C)
            return out.transpose(1, 2)

        if fm == "cross_connection":
            if self.cross_s2l is None or self.cross_l2s is None:
                raise RuntimeError("cross_connection 融合缺少互映射参数。请检查 fusion_method 设置。")
            hs_hat = self.cross_l2s(hl)  # (B, cross_out, L)
            hl_hat = self.cross_s2l(hs)

            if self.cross_mode == "concat":
                return torch.cat([hs_hat, hl_hat], dim=1)

            if self.cross_mode == "gated":
                if self.cross_gate is None:
                    raise RuntimeError("cross_connection gated 模式缺少 cross_gate。")
                g = torch.sigmoid(self.cross_gate(torch.cat([hs_hat, hl_hat], dim=1)))
                return g * hs_hat + (1.0 - g) * hl_hat

            return hs_hat + hl_hat

        return hs + hl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, Din, Lp)

        z = self.proj_drop(self.proj_act(self.proj_norm(self.proj(x))))

        hs, hl = z, z
        hs, hl = self.block1(hs, hl)
        hs, hl = self.pool(hs), self.pool(hl)

        hs, hl = self.block2(hs, hl)
        hs, hl = self.pool(hs), self.pool(hl)

        hs, hl = self.block3(hs, hl)

        h = self._fuse(hs, hl)
        h = self.adapool(h).squeeze(-1)

        h = self.head(h)
        out = self.out(h)
        return out


def build_model(cfg, in_dim):
    """构建模型"""
    if cfg.model_type == "ms_bcnn":
        print(f"使用 MS--BCNN 架构，融合方式: {cfg.fusion_method}，归一化: {cfg.norm_type}")
        model = MSBCNN(in_dim, cfg.Lp, cfg).to(DEVICE)
    elif cfg.model_type == "dual_cnn":
        print(f"使用 双流多尺度CNN 架构，融合方式: {cfg.fusion_method}，归一化: {cfg.norm_type}")
        model = DualMultiScaleCNN(in_dim, cfg.Lp, cfg.Lf, cfg).to(DEVICE)
    elif cfg.model_type == "simplified_cnn":
        print(f"使用简化版进化CNN架构，归一化: {cfg.norm_type}")
        model = SimplifiedEvolvedCNN(in_dim, cfg.Lp, cfg.Lf, cfg).to(DEVICE)
    else:
        raise ValueError(f"未知的 model_type: {cfg.model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    return model, total_params


def create_loss_function(cfg):
    """创建损失函数"""
    if cfg.loss_function == "mse":
        crit = nn.MSELoss()
        print("使用损失函数: MSE")
    elif cfg.loss_function == "mae":
        crit = nn.L1Loss()
        print("使用损失函数: MAE")
    elif cfg.loss_function == "smooth_l1":
        crit = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta)
        print("使用损失函数: SmoothL1")
    elif cfg.loss_function == "mape":
        crit = MAPELoss(eps=cfg.mape_eps)
        print("使用损失函数: MAPE")
    elif cfg.loss_function == "smape":
        crit = SMAPELoss(eps=cfg.smape_eps)
        print("使用损失函数: SMAPE")
    elif cfg.loss_function == "huber":
        crit = nn.HuberLoss(delta=cfg.huber_delta)
        print("使用损失函数: Huber")
    elif cfg.loss_function == "logcosh":
        crit = LogCoshLoss(eps=cfg.logcosh_eps)
        print("使用损失函数: LogCosh")
    elif cfg.loss_function == "quantile":
        crit = QuantileLoss(q=cfg.quantile_q)
        print(f"使用损失函数: Quantile (q={cfg.quantile_q})")
    elif cfg.loss_function == "multi_quantile":
        crit = MultiQuantileLoss(quantiles=list(cfg.multi_quantiles))
        print(f"使用损失函数: MultiQuantile {cfg.multi_quantiles}")
    elif cfg.loss_function == "combined":
        # 构建组合损失
        losses = []
        for loss_type in cfg.combined_loss_types:
            if loss_type == "mse":
                losses.append(nn.MSELoss())
            elif loss_type == "mae":
                losses.append(nn.L1Loss())
            elif loss_type == "smooth_l1":
                losses.append(nn.SmoothL1Loss(beta=cfg.smooth_l1_beta))
            elif loss_type == "mape":
                losses.append(MAPELoss(eps=cfg.mape_eps))
            elif loss_type == "smape":
                losses.append(SMAPELoss(eps=cfg.smape_eps))
            elif loss_type == "huber":
                losses.append(nn.HuberLoss(delta=cfg.huber_delta))
            elif loss_type == "logcosh":
                losses.append(LogCoshLoss(eps=cfg.logcosh_eps))

        crit = CombinedLoss(losses, weights=list(cfg.combined_loss_weights))
        print(f"使用组合损失函数: {cfg.combined_loss_types} with weights {cfg.combined_loss_weights}")
    elif cfg.loss_function == "adaptive_combined":
        # 构建自适应组合损失
        losses = []
        for loss_type in cfg.combined_loss_types:
            if loss_type == "mse":
                losses.append(nn.MSELoss())
            elif loss_type == "mae":
                losses.append(nn.L1Loss())
            elif loss_type == "smooth_l1":
                losses.append(nn.SmoothL1Loss(beta=cfg.smooth_l1_beta))
            elif loss_type == "mape":
                losses.append(MAPELoss(eps=cfg.mape_eps))
            elif loss_type == "smape":
                losses.append(SMAPELoss(eps=cfg.smape_eps))
            elif loss_type == "huber":
                losses.append(nn.HuberLoss(delta=cfg.huber_delta))
            elif loss_type == "logcosh":
                losses.append(LogCoshLoss(eps=cfg.logcosh_eps))

        crit = AdaptiveCombinedLoss(losses,
                                    initial_weights=list(cfg.combined_loss_weights),
                                    adapt_rate=cfg.adaptive_combined_adapt_rate)
        # print(f"使用自适应组合损失函数: {cfg.combined_loss_types} with initial weights {cfg.combined_loss_weights}")
    else:
        raise ValueError(f"未知的损失函数: {cfg.loss_function}")

    return crit


def create_scheduler(cfg, optimizer, train_loader):
    """创建学习率调度器"""
    if not cfg.use_scheduler:
        return None

    if cfg.scheduler_type == "reduce_on_plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler_mode,
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience
        )
        print("使用学习率调度: ReduceLROnPlateau")
    elif cfg.scheduler_type == "cosine_warmup":
        sch = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.max_epochs,
            base_lr=cfg.lr,
            min_lr=cfg.min_lr
        )
        print(f"使用学习率调度: CosineWarmup (warmup_epochs={cfg.warmup_epochs})")
    else:
        raise ValueError(f"未知的调度器类型: {cfg.scheduler_type}")

    return sch


def train_one_epoch(model, train_loader, crit, opt, cfg, epoch, ystd):
    """训练一个epoch"""
    model.train()
    tr_preds, tr_tgts, tr_losses = [], [], []

    for X, y_norm in train_loader:
        X = X.to(DEVICE)
        y_norm = y_norm.to(DEVICE)
        opt.zero_grad()

        # 普通训练（删除混合精度相关代码）
        yhat_norm = model(X)
        loss_yhat, loss_y_true = get_loss_inputs(yhat_norm, y_norm, cfg)
        loss = crit(loss_yhat, loss_y_true)

        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        tr_losses.append(loss.item())

        # 使用工具函数处理预测输出（用于指标计算）
        yhat_for_metrics, _ = process_multi_quantile_output(yhat_norm, y_norm, cfg, is_training=False)
        tr_preds.append(yhat_for_metrics.detach().cpu().numpy())
        tr_tgts.append(y_norm.cpu().numpy())

    tr_loss_mean = float(np.mean(tr_losses)) if tr_losses else float('nan')

    tr_preds = np.vstack(tr_preds) if tr_preds else np.zeros((0, cfg.Lf), dtype=np.float32)
    tr_tgts = np.vstack(tr_tgts) if tr_tgts else np.zeros((0, cfg.Lf), dtype=np.float32)
    tr_preds_den = ystd.denorm(tr_preds)
    tr_tgts_den = ystd.denorm(tr_tgts)
    tr_metrics = calculate_metrics(tr_tgts_den, tr_preds_den)

    return tr_loss_mean, tr_metrics


def evaluate(model, data_loader, crit, cfg, ystd, is_validation=True):
    """评估模型"""
    model.eval()
    preds, tgts, losses = [], [], []

    with torch.no_grad():
        for X, y_norm in data_loader:
            X = X.to(DEVICE)
            y_norm = y_norm.to(DEVICE)
            yhat_norm = model(X)

            # 使用工具函数处理损失计算输入
            loss_yhat, loss_y_true = get_loss_inputs(yhat_norm, y_norm, cfg)
            l = crit(loss_yhat, loss_y_true)

            if not (torch.isnan(l) or torch.isinf(l)):
                losses.append(l.item())

            # 使用工具函数处理预测输出（用于指标计算）
            yhat_for_metrics, _ = process_multi_quantile_output(yhat_norm, y_norm, cfg, is_training=False)
            preds.append(yhat_for_metrics.cpu().numpy())
            tgts.append(y_norm.cpu().numpy())

    loss_mean = float(np.mean(losses)) if losses else float('inf')

    preds = np.vstack(preds) if preds else np.zeros((0, cfg.Lf), dtype=np.float32)
    tgts = np.vstack(tgts) if tgts else np.zeros((0, cfg.Lf), dtype=np.float32)
    preds_den = ystd.denorm(preds)
    tgts_den = ystd.denorm(tgts)
    metrics = calculate_metrics(tgts_den, preds_den)

    return loss_mean, metrics

import gc
import torch

# 清理内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ========== 主训练函数 ==========
def run_evolved_cnn_training(
        pkl_path: str = "./all_data.pkl",
        out_dir: str = "./evolved_cnn_out",
        cfg: Config = Config(),
):
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"数据文件不存在: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, list) and isinstance(data[0], dict)):
        raise TypeError(f"Expected list of dict, got {type(data)}")

    # 准备数据集
    train_ds, val_ds, test_ds, scaler, ystd, H_dim, M_dim, L_dim = prepare_datasets(cfg, data)
    in_dim = train_ds.in_dim

    # 保存标准化参数
    np.save(os.path.join(out_dir, "label_mean.npy"), ystd.mean)
    np.save(os.path.join(out_dir, "label_std.npy"), ystd.std)
    np.save(os.path.join(out_dir, "feat_min.npy"), scaler.minv)
    np.save(os.path.join(out_dir, "feat_max.npy"), scaler.rng + scaler.minv)

    # 构建模型
    model, total_params = build_model(cfg, in_dim)

    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 损失函数
    crit = create_loss_function(cfg)

    # 学习率调度器
    sch = create_scheduler(cfg, opt, train_ds)

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=cfg.train_shuffle)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 训练循环
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0

    train_losses_norm = []
    val_losses_norm = []

    print("\n开始训练 MS--BCNN 模型...")
    # 说明：NormTrain/NormVal 为训练/验证损失（用于优化与早停）；
    #      Train/Val NMSE、NMAE、MAPE 为按论文 Appendix B.5 计算的评估指标。
    print("=" * 148)
    print(f"{'Epoch':^6} | {'NormTrain':^10} | {'NormVal':^10} | "
          f"{'Train NMSE':^10} | {'Train NMAE':^10} | {'Train MAPE':^10} | "
          f"{'Val NMSE':^10} | {'Val NMAE':^10} | {'Val MAPE':^10} | {'LR':^10}")
    print("=" * 148)

    for epoch in range(1, cfg.max_epochs + 1):
        # 训练
        tr_loss_mean, tr_metrics = train_one_epoch(
            model, train_loader, crit, opt, cfg, epoch, ystd
        )

        # 验证
        va_loss_mean, va_metrics = evaluate(model, val_loader, crit, cfg, ystd, is_validation=True)

        # 学习率调度
        current_lr = opt.param_groups[0]['lr']
        if sch is not None:
            if cfg.scheduler_type == "reduce_on_plateau":
                sch.step(va_loss_mean)
            elif cfg.scheduler_type == "cosine_warmup":
                sch.step()
                current_lr = sch.get_lr()

        # 打印一行综合信息
        print(f"{epoch:^6} | {tr_loss_mean:^10.4f} | {va_loss_mean:^10.4f} | "
              f"{tr_metrics['nmse']:^10.4f} | {tr_metrics['nmae']:^10.4f} | {tr_metrics['mape']:^10.4f} | "
              f"{va_metrics['nmse']:^10.4f} | {va_metrics['nmae']:^10.4f} | {va_metrics['mape']:^10.4f} | "
              f"{current_lr:^10.2e}")

        # 记录
        train_losses_norm.append(tr_loss_mean)
        val_losses_norm.append(va_loss_mean)

        # 早停：基于验证损失
        if va_loss_mean < best_val - 1e-6:
            best_val = va_loss_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_imp = 0
            torch.save(
                {"model": best_state, "cfg": cfg.__dict__, "in_dim": in_dim, "epoch": epoch, "val_loss": best_val},
                os.path.join(out_dir, "best_evolved_cnn.pt"),
            )
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                print(f"\n早停于第 {epoch} 轮")
                break

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, crit, cfg, ystd, is_validation=False)
    # === 获取测试集预测与真实值（用于保存） ===
    model.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for X, y_norm in test_loader:
            X = X.to(DEVICE)
            y_norm = y_norm.to(DEVICE)
            yhat_norm = model(X)
            yhat_for_metrics, _ = process_multi_quantile_output(yhat_norm, y_norm, cfg, is_training=False)
            test_preds.append(yhat_for_metrics.cpu().numpy())
            test_tgts.append(y_norm.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_tgts = np.vstack(test_tgts)

    # 反归一化
    test_preds_den = ystd.denorm(test_preds)
    test_tgts_den = ystd.denorm(test_tgts)

    # 测试集输出
    print("\n" + "=" * 60)
    print("在测试集上进行最终评估...")
    print("=" * 60)

    print("\n测试集整体指标:")
    print("=" * 50)
    print(f"NMSE:  {test_metrics['nmse']:.6f}")
    print(f"NMAE:  {test_metrics['nmae']:.6f}")
    print(f"MAPE:  {test_metrics['mape']:.6f}%")

    print("\n测试集每个标签的指标:")
    print("=" * 80)
    print(f"{'Label':^6} | {'MSE':^12} | {'MAPE':^12} | {'MAE':^12}")
    print("=" * 80)
    for i in range(min(5, cfg.Lf)):
        mse_label = test_metrics['mse_per_label'][i]
        mape_label = test_metrics['mape_per_label'][i]
        mae_label = test_metrics['mae_per_label'][i]
        print(f"{i + 1:^6} | {mse_label:^12.6f} | {mape_label:^12.6f} | {mae_label:^12.6f}")
    print("=" * 80)
    print(f"{'模型参数量':<15}: {total_params}")
    print(f"{'最佳验证损失':<15}: {best_val:.6f} (第 {best_epoch} 轮)")

    # 保存结果
    results = {
        "y_true": None,  # 实际值需要在评估时收集
        "y_pred": None,  # 预测值需要在评估时收集
        "metrics": test_metrics,
        "in_dim": in_dim,
        "cfg": cfg.__dict__,
        "train_losses_norm": train_losses_norm,
        "val_losses_norm": val_losses_norm,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "model_parameters": total_params,
    }

    out_file = os.path.join(out_dir, "evolved_cnn_results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    report = {
        "model_parameters": total_params,
        "training_epochs": len(train_losses_norm),
        "best_epoch": best_epoch,
        "final_metrics": test_metrics,
        "architecture_info": {
            "input_dim": in_dim,
            "sequence_length": cfg.Lp,
            "output_dim": cfg.Lf,
            "model_type": cfg.model_type,
            "fusion_method": cfg.fusion_method,
            "loss_function": cfg.loss_function,
            "norm_type": cfg.norm_type,
            "resample_method": cfg.resample_method,
            "pool_type": cfg.pool_type if cfg.resample_method == "pool" else None,
            "based_on_evolution": True,
        },
    }
    with open(os.path.join(out_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n训练完成! 结果保存到:", out_dir)

    # === 绘制损失曲线 ===
    if not getattr(cfg, "suppress_plots", False):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(train_losses_norm, label='Train Loss', linewidth=2)
            plt.plot(val_losses_norm, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training & Validation Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            loss_curve_path = os.path.join(out_dir, "loss_curve.png")
            plt.savefig(loss_curve_path, dpi=300)
            plt.close()
            print(f"✅ 训练/验证损失曲线已保存至: {loss_curve_path}")
        except Exception as e:
            print("⚠️ 无法绘制损失曲线:", e)

    # === 绘制测试集预测与真实值对比图 ===
    if not getattr(cfg, "suppress_plots", False):
        try:
            Yp = np.array(test_metrics.get("test_predictions", test_preds_den))
            Yt = np.array(test_metrics.get("test_true", test_tgts_den))

            fig, axes = plt.subplots(cfg.Lf, 1, figsize=(12, 2.5 * cfg.Lf), sharex=True)

            for i in range(cfg.Lf):
                axes[i].plot(Yt[:, i], label="True", color="blue", linewidth=1)
                axes[i].plot(Yp[:, i], label="Predicted", color="red", linestyle="--", linewidth=1)
                axes[i].set_ylabel(f"Label {i + 1}")
                axes[i].legend(loc="upper right")

            axes[-1].set_xlabel("Sample Index")
            plt.suptitle("Test Set: True vs Predicted", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            save_path = os.path.join(out_dir, "test_predictions.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"✅ 测试集预测–真实值对比图已保存至: {save_path}")
        except Exception as e:
            print("⚠️ 无法绘制预测对比图:", e)

    return test_metrics, model


if __name__ == "__main__":
    cfg = Config(
        # ===== 0 数据参数 =====
        seed=42,  # 随机种子，确保结果可重现
        train_shuffle=True,  # 训练集是否打乱
        split_mode="sequential",  # 数据划分方式："random"随机划分 / "sequential"顺序划分
        split_seed=42,  # 数据划分随机种子
        train_ratio=0.59,  # 训练集比例
        val_ratio=0.067,  # 验证集比例
        test_ratio=0.343,  # 测试集比例
        Lf=7,  # 输出序列长度
        max_epochs=500,  # 最大训练轮数
        patience=20,  # 早停耐心值
        grad_clip=1.0,  # 梯度裁剪阈值
        multi_quantiles=(0.1, 0.5, 0.9),  # 多个分位数值，（当 loss_function="multi_quantile" 时）
        # ===== 0 模型选择 =====
        model_type="ms_bcnn",  # "ms_bcnn"、"dual_cnn"、"simplified_cnn"

        # ===== 1 数据参数 =====
        Lp=12,  # 输入序列长度
        batch_size=32,  # 批次大小
        # ===== 1 重采样方法选择 =====
        resample_method="hybrid",  # 重采样方法：
        # "linear" - 线性插值（默认）
        # "decimate_repeat" - 间隔采样+重复采样
        # "hybrid" - 线性插值+重复混合
        # "pool" - 池化方式
        # "conv_blurpool" - 卷积重采样（抗混叠）
        # "fir_lowpass" - FIR低通重采样
        # === 2 池化类型（当resample_method="pool"时使用）
        pool_type="avg",
        # "avg" - 平均池化
        # "max" - 最大池化
        # "median" - 中位数池化
        # "weighted" - 加权池化
        # ===== 1 训练参数 =====
        lr=1e-3,  # 学习率
        weight_decay=5.39e-5,  # 权重衰减（L2正则化）
        # ===== 1 归一化层选择 =====
        norm_type="layer_norm",  # "batch_norm"批量, "layer_norm"层, "instance_norm"样本
        # ====== 1 损失函数选择 =====
        loss_function="adaptive_combined",  # 损失函数类型：
        # "mse" - 均方误差
        # "mae" - 平均绝对误差
        # "smooth_l1" - Smooth L1损失
        # "mape" - 平均绝对百分比误差
        # "huber" - Huber损失
        # "logcosh" - LogCosh损失
        # "quantile" - 分位数损失
        # "multi_quantile" - 多分位数损失，置信区间
        # "combined" - 组合损失
        # "smape" - 对称平均绝对百分比误差
        # "adaptive_combined" - 自适应组合损失
        # === 2 SmoothL1损失参数（当 loss_function="smooth_l1" 时）
        smooth_l1_beta=1.05,  # SmoothL1的beta参数
        # === 2 Huber损失参数（当 loss_function="huber" 时）
        huber_delta=1.05,  # Huber损失的delta参数
        # === 2 分位数损失参数（当 loss_function="quantile" 时）
        quantile_q=0.5,  # 分位数，偏向低预测"乐观值"，偏向高预测"悲观值"
        # === 2 组合损失函数参数（当 loss_function="combined"或"adaptive_combined" 时使用的参数）
        combined_loss_types=("mae", "logcosh"),  # 组合的损失函数类型
        # = 1. 基础常见组合（稳定、通用）
            #("mse", "mae") → 最常见，平衡整体误差与鲁棒性
            #("mse", "huber") → MSE + Huber，对异常值更稳健
            #("mae", "huber") → 两个鲁棒损失结合，适合噪声较大的数据
        # = 2. 与相对误差结合（适合比例/预测任务）
            #("mae", "mape") → 绝对误差 + 相对误差，更关注预测比例
            #("mse", "smape") → 对称相对误差，适合金融 / 时序预测
        # = 3. 与分位数损失结合（适合不确定性建模）
            #("mae", "quantile") → 平均趋势 + 分布尾部敏感
            #("huber", "quantile") → 鲁棒性 + 分位数估计
            #("mae", "multi_quantile") → 预测区间，更关注上下分位点
        # = 4. 平滑损失组合（数值稳定）
            #("mse", "smooth_l1") → 平滑版
            #("mae", "logcosh") → 类似MAE，但更平滑，对小误差更敏感
        # === 2 自适应组合损失函数参数（当 loss_function= "adaptive_combined" 时使用的参数）
        combined_loss_weights=(0.7, 0.3),  # 各损失函数的权重
        adaptive_combined_adapt_rate=0.001,  # 自适应组合损失的学习率（0.001，0.01，0.1，0.2，0.5）
        # ====== 1 学习率调度器 =====
        use_scheduler=True,  # 是否使用学习率调度器
        # === 2 学习率调度器选择（当use_scheduler=True）
        scheduler_type="cosine_warmup",  # 调度器类型："reduce_on_plateau" / "cosine_warmup"
        # == 3 reduce on plateau参数（当 scheduler_type="reduce_on_plateau" 时）
        scheduler_factor=0.35,  # 学习率衰减因子
        scheduler_patience=3,  # 耐心值
        scheduler_mode="min",  # 监控模式："min" / "max"
        # == 3 cosine warmup参数（当 scheduler_type="cosine_warmup" 时）
        warmup_epochs=10,  # warmup阶段轮数

        # ===== 1 SimplifiedCNN 参数 =====
        simplified_conv1_filters=16,  # 第一卷积层滤波器数量
        simplified_conv2_filters=16,  # 第二卷积层滤波器数量
        simplified_conv3_filters=64,  # 第三卷积层滤波器数量
        simplified_conv1_kernel=3,  # 第一卷积层核大小
        simplified_conv2_kernel=5,  # 第二卷积层核大小
        simplified_conv3_kernel=7,  # 第三卷积层核大小
        simplified_fc1_units=256,  # 第一个全连接层单元数
        simplified_fc2_units=16,  # 第二个全连接层单元数
        simplified_fc3_units=8,  # 第三个全连接层单元数
        simplified_pool_kernel=2,  # 池化层核大小
        simplified_activation="gelu",  # 激活函数："relu" / "tanh"/gelu


        # ===== 1 MS--BCNN 参数（当 model_type="ms_bcnn" 时生效） =====
        ms_c0=16,  # 输入投影通道数 C0
        ms_channels=(16, 16, 64),  # 三层卷积通道数 (C1, C2, C3)
        ms_short_kernels=(3, 5, 7),  # 三层短核大小
        ms_long_kernels=(9, 11, 13),  # 三层长核大小
        ms_pool_kernel=2,  # 层间池化核大小
        ms_dropout=0.1,  # 主干 dropout
        ms_activation="gelu",  # 激活函数
        ms_fc_units=(256, 16, 8),  # 预测头三层 MLP 宽度
        ms_head_dropout=0.1,  # 预测头 dropout

        # ===== 1 DualMultiScaleCNN 参数（当 model_type="dual_cnn" 时生效） =====
        dual_branch_filters=8,  # 每个分支的滤波器数量
        dual_branch_a_kernels=(1, 3, 5),  # 分支A的卷积核大小（局部特征）
        dual_branch_b_kernels=(7, 9, 11),  # 分支B的卷积核大小（全局特征）
        dual_fusion_units=32,  # 融合层单元数
        dual_fc1_units=32,  # 第一个全连接层单元数
        dual_fc2_units=16,  # 第二个全连接层单元数
        dual_dropout=0.1,  # Dropout比率
        dual_activation="relu",  # 激活函数："relu" / "tanh"
        fusion_method="gate" ,  # 融合方式
            # "concat" - 简单拼接
            # "add" - 简单相加
            # "cross_connection" - 交叉映射连接
            # "weighted_sum" - 加权求和
            # "gate" - 门控机制
            # "attention" - 注意力机制
            # "multi_head_attention" - 多头注意力
        # === 2 交叉连接融合参数（当 fusion_method="cross_connection" 时）
        fusion_cross_connection_ratio=0.5,  # 交叉连接比例
        cross_mode="add",  # 交叉连接模式："add" / "concat" / "gated"
        # == 3 A->B投影输出通道数（cross_mode="concat"或"gated"）
        cross_output_dim=16,  # A->B投影输出通道数 add不需要
        # === 2 加权求和融合参数（当 fusion_method="weighted_sum" 时）
        fusion_weighted_sum_init_a=0.5,  # 分支A初始权重
        fusion_weighted_sum_init_b=0.5,  # 分支B初始权重
        fusion_final_fusion="concat",  # 最终融合方式："concat" 拼接 / "add" 相加
        # === 2 门控融合参数（当 fusion_method="gate" 时）
        fusion_gate_dim=64,  # 门控层维度
        # === 2 注意力融合参数（当 fusion_method="attention" 时）
        fusion_attention_dim=16,  # 注意力层维度
        # === 2 多头注意力融合参数（当 fusion_method="multi_head_attention" 时）
        fusion_multi_head_attention_heads=4,  # 注意力头数
    )

    here = os.path.dirname(__file__)

    # 兼容多种常见数据命名
    candidate_paths = [
        os.path.join(here, "all_data.pkl"),
        "all_data_sanitized_share.pkl",
        "../all_data.pkl",
        "all_data.pkl",
    ]
    pkl_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            pkl_path = p
            break

    if pkl_path is None:
        print("错误: 未找到数据文件(all_data_clean.pkl / all_data.pkl)")
        raise SystemExit(1)

    try:
        metrics_result, trained_model = run_evolved_cnn_training(
            pkl_path=pkl_path,
            out_dir=os.path.join(here, "evolved_cnn_results"),
            cfg=cfg,
        )

        print("\n" + "=" * 60)
        print("MS--BCNN 模型训练完成!")
        print("=" * 60)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()