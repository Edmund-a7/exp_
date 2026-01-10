"""
IAM - Intelligent Agent Memory for Video Generation

基于实体追踪的流式视频生成框架

模块:
- llm_agent: LLM Agent 实体提取和 ID 管理
- memory_bank: Memory Bank 帧管理和选择

使用方法:
    from iam import LLMAgent, MemoryBank, EntityStruct
"""

from .llm_agent import (
    EntityStruct,
    LLMWrapper,
    EntityStructExtractor,
    GlobalIDManager,
    LLMAgent
)

from .memory_bank import (
    FrameInfo,
    MemoryBank
)

__all__ = [
    # llm_agent
    "EntityStruct",
    "LLMWrapper",
    "EntityStructExtractor",
    "GlobalIDManager",
    "LLMAgent",
    # memory_bank
    "FrameInfo",
    "MemoryBank",
]

__version__ = "1.0.0"
