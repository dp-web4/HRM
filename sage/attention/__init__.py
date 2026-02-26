"""
SAGE Continuous Attention System

This module implements Tier 0 (always-on) attention orchestration
following Nova's design from HRM_IRP_Raising_Continuous_Attention_Suggestions.md
"""

# v1 kernel (basic foundation)
from .kernel import AttentionKernel, ExperienceBuffer, SleepTrigger

# v2 kernel (with SNARC salience integration)
from .kernel_v2 import AttentionKernelV2
from .experience_salience import ExperienceSalienceScorer

# v3 kernel (with LLM runtime - Tier 1 integration)
from .kernel_v3 import AttentionKernelV3

# Core components
from .state import AttentionState, StateTransition
from .atp_budget import ATPBudget, PluginTrust
from .kernel_logging import TickLogger, ActionLogger, ContextLogger
from .plugin_router import PluginRouter

__all__ = [
    # v1 kernel
    'AttentionKernel',
    'ExperienceBuffer',
    'SleepTrigger',
    # v2 kernel (with SNARC salience)
    'AttentionKernelV2',
    'ExperienceSalienceScorer',
    # v3 kernel (with LLM runtime - recommended)
    'AttentionKernelV3',
    # Core components
    'AttentionState',
    'StateTransition',
    'ATPBudget',
    'PluginTrust',
    'TickLogger',
    'ActionLogger',
    'ContextLogger',
    'PluginRouter',
]
