"""
Soft Prompt Transition (SPT) Scheduler

提供多种调度策略，用于计算 prompt 切换时的混合系数 α。
α 从 0 渐变到 1，表示从旧 prompt 过渡到新 prompt。
"""

import math
from abc import ABC, abstractmethod
from typing import Optional


class TransitionScheduler(ABC):
    """过渡调度器基类"""

    def __init__(self, window_frames: int = 9, delay_frames: int = 3):
        """
        Args:
            window_frames: 过渡窗口帧数 (不含 delay)
            delay_frames: 延迟启动帧数，在此期间 α=0
        """
        self.window_frames = window_frames
        self.delay_frames = delay_frames
        self.total_frames = delay_frames + window_frames

    @abstractmethod
    def _compute_alpha(self, t: float) -> float:
        """
        计算归一化时间 t ∈ [0, 1] 对应的 α 值

        Args:
            t: 归一化时间，0 表示过渡开始，1 表示过渡结束
        Returns:
            α 值，范围 [0, 1]
        """
        pass

    def get_alpha(self, frames_since_switch: int) -> Optional[float]:
        """
        获取当前帧的混合系数 α

        Args:
            frames_since_switch: 切换后已生成的帧数
                - 交互推理当前为按 chunk 采样：该值通常按 chunk 长度递增，
                  因此 α 也按 chunk 更新（而非逐帧）
        Returns:
            α 值 (0~1)，None 表示过渡已完成
        """
        if frames_since_switch >= self.total_frames:
            return None  # 过渡完成

        if frames_since_switch < self.delay_frames:
            return 0.0  # delay 阶段，完全使用旧 prompt

        # 计算归一化时间
        t = (frames_since_switch - self.delay_frames) / self.window_frames
        t = min(t, 1.0)  # 确保不超过 1

        return self._compute_alpha(t)

    def is_complete(self, frames_since_switch: int) -> bool:
        """判断过渡是否完成"""
        return frames_since_switch >= self.total_frames


class LinearScheduler(TransitionScheduler):
    """
    线性调度器

    α = t
    特点：匀速变化，实现简单
    """

    def _compute_alpha(self, t: float) -> float:
        return t


class CosineScheduler(TransitionScheduler):
    """
    余弦调度器 (推荐)

    α = 0.5 × (1 - cos(π × t))
    特点：开始慢、中间快、结束慢，过渡更自然
    """

    def _compute_alpha(self, t: float) -> float:
        return 0.5 * (1 - math.cos(math.pi * t))


class SigmoidScheduler(TransitionScheduler):
    """
    Sigmoid 调度器

    α = sigmoid(k × (t - 0.5)) 归一化到 [0, 1]
    特点：可通过 steepness 参数调节陡峭度
    """

    def __init__(self, window_frames: int = 9, delay_frames: int = 3, steepness: float = 6.0):
        """
        Args:
            steepness: 陡峭度参数，越大过渡越陡峭
        """
        super().__init__(window_frames, delay_frames)
        self.steepness = steepness
        # 预计算归一化因子
        self._low = self._sigmoid(-0.5 * steepness)
        self._high = self._sigmoid(0.5 * steepness)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _compute_alpha(self, t: float) -> float:
        raw = self._sigmoid(self.steepness * (t - 0.5))
        # 归一化到 [0, 1]
        return (raw - self._low) / (self._high - self._low)


class StepScheduler(TransitionScheduler):
    """
    阶梯调度器

    按 chunk 为单位跳变，每个 chunk 内 α 保持不变
    特点：以 chunk 为粒度控制
    """

    def __init__(self, window_frames: int = 9, delay_frames: int = 3, frames_per_chunk: int = 3):
        super().__init__(window_frames, delay_frames)
        self.frames_per_chunk = frames_per_chunk
        self.num_steps = max(1, window_frames // frames_per_chunk)

    def _compute_alpha(self, t: float) -> float:
        step = int(t * self.num_steps)
        step = min(step, self.num_steps - 1)
        return (step + 1) / self.num_steps


class AdaptiveScheduler:
    """
    自适应调度器：根据语义距离动态调整过渡窗口大小

    原理：
    - 语义距离小（如"走路→跑步"）→ 短窗口，避免混合期过长导致运动模糊
    - 语义距离大（如"公园→雪山"）→ 长窗口，避免过渡太快产生伪影

    用法：
    1. __init__ 时创建，此时内部 scheduler 使用默认 window
    2. 每次 prompt 切换时调用 update_for_switch(distance) 动态创建新 scheduler
    """

    def __init__(self,
                 base_scheduler_type: str = "cosine",
                 min_window: int = 3,
                 max_window: int = 15,
                 delay_frames: int = 3,
                 **scheduler_kwargs):
        """
        Args:
            base_scheduler_type: 底层调度曲线类型 (linear, cosine, sigmoid, step)
            min_window: 最短过渡窗口 (帧)
            max_window: 最长过渡窗口 (帧)
            delay_frames: 延迟启动帧数
            **scheduler_kwargs: 传递给底层 scheduler 的额外参数
        """
        self.base_scheduler_type = base_scheduler_type
        self.min_window = min_window
        self.max_window = max_window
        self.delay_frames = delay_frames
        self.scheduler_kwargs = scheduler_kwargs

        # 初始化一个默认 scheduler（使用 max_window 作为保守默认值）
        self._active_scheduler: TransitionScheduler = create_scheduler(
            scheduler_type=base_scheduler_type,
            window_frames=max_window,
            delay_frames=delay_frames,
            **scheduler_kwargs
        )

    def update_for_switch(self, semantic_distance: float):
        """
        prompt 切换时调用，根据语义距离创建对应窗口的 scheduler

        Args:
            semantic_distance: cosine distance (0=相同, 1=完全不同)
        """
        t = max(0.0, min(1.0, semantic_distance))
        window = int(self.min_window + t * (self.max_window - self.min_window))
        # 对齐到 chunk 边界 (3帧一个chunk)，最小 3 帧
        window = max(3, (window // 3) * 3)

        self._active_scheduler = create_scheduler(
            scheduler_type=self.base_scheduler_type,
            window_frames=window,
            delay_frames=self.delay_frames,
            **self.scheduler_kwargs
        )

        print(f"[SPT-Adaptive] semantic_distance={t:.3f}, window_frames={window}")

    def get_alpha(self, frames_since_switch: int) -> Optional[float]:
        """代理到内部 scheduler"""
        return self._active_scheduler.get_alpha(frames_since_switch)

    def is_complete(self, frames_since_switch: int) -> bool:
        """代理到内部 scheduler"""
        return self._active_scheduler.is_complete(frames_since_switch)

    @property
    def total_frames(self) -> int:
        """代理到内部 scheduler"""
        return self._active_scheduler.total_frames

    @property
    def window_frames(self) -> int:
        """代理到内部 scheduler"""
        return self._active_scheduler.window_frames


def create_scheduler(
    scheduler_type: str = "cosine",
    window_frames: int = 9,
    delay_frames: int = 3,
    **kwargs
) -> TransitionScheduler:
    """
    工厂函数：根据类型创建调度器

    Args:
        scheduler_type: 调度器类型 (linear, cosine, sigmoid, step)
        window_frames: 过渡窗口帧数
        delay_frames: 延迟启动帧数
        **kwargs: 特定调度器的额外参数

    Returns:
        TransitionScheduler 实例
    """
    schedulers = {
        "linear": LinearScheduler,
        "cosine": CosineScheduler,
        "sigmoid": SigmoidScheduler,
        "step": StepScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                         f"Available: {list(schedulers.keys())}")

    cls = schedulers[scheduler_type]

    if scheduler_type == "sigmoid":
        steepness = kwargs.get("steepness", 6.0)
        return cls(window_frames, delay_frames, steepness=steepness)
    elif scheduler_type == "step":
        frames_per_chunk = kwargs.get("frames_per_chunk", 3)
        return cls(window_frames, delay_frames, frames_per_chunk=frames_per_chunk)
    else:
        return cls(window_frames, delay_frames)
