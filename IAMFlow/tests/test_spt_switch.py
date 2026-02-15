import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _install_module(name: str, attrs: dict) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_agent_pipeline_class():
    _install_package("pipeline")
    _install_package("utils")
    _install_package("iam")

    _install_module(
        "pipeline.interactive_causal_inference",
        {"InteractiveCausalInferencePipeline": type("StubBase", (), {})},
    )
    _install_module(
        "utils.wan_wrapper",
        {
            "WanDiffusionWrapper": object,
            "WanTextEncoder": object,
            "WanVAEWrapper": object,
        },
    )
    _install_module(
        "utils.memory",
        {
            "gpu": None,
            "get_cuda_free_memory_gb": lambda *_: 0,
            "move_model_to_device_with_memory_preservation": lambda *_, **__: None,
        },
    )
    _install_module("utils.debug_option", {"DEBUG": False})
    _install_module("utils.profiling", {"compute_pure_diffusion_time": lambda **_: (0.0, 0.0)})
    _install_module("iam.llm_agent", {"LLMAgent": object, "EntityStruct": object, "SceneStruct": object})
    _install_module("iam.memory_bank", {"MemoryBank": object})

    module_path = os.path.join(ROOT_DIR, "pipeline", "agent_causal_inference.py")
    spec = importlib.util.spec_from_file_location(
        "pipeline.agent_causal_inference",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline.agent_causal_inference"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AgentCausalInferencePipeline


class DummyGenerator:
    def __init__(self) -> None:
        self.model = SimpleNamespace(local_attn_size=None)

    def __call__(self, noisy_image_or_video=None, **_kwargs):
        return None, torch.zeros_like(noisy_image_or_video)


class DummyVAE:
    def decode_to_pixel(self, x, use_cache=False):
        return x


class DummyScheduler:
    def add_noise(self, x, *_args, **_kwargs):
        return x


def test_agent_pipeline_uses_soft_switch_when_spt_enabled():
    AgentCausalInferencePipeline = _load_agent_pipeline_class()
    pipeline = AgentCausalInferencePipeline.__new__(AgentCausalInferencePipeline)

    called_soft = []
    called_recache = []

    pipeline.spt_enabled = True
    pipeline._soft_switch = lambda: called_soft.append(True)
    pipeline._recache_after_switch = lambda *_args, **_kwargs: called_recache.append(True)
    pipeline._reset_agent_state = lambda: None
    pipeline._process_prompt_start = lambda *_args, **_kwargs: None
    pipeline._process_chunk_eviction = lambda *_args, **_kwargs: None
    pipeline._inject_iam_memory_to_bank = lambda *_args, **_kwargs: None
    pipeline._initialize_kv_cache = lambda *_args, **_kwargs: None
    pipeline._initialize_crossattn_cache = lambda *_args, **_kwargs: None
    pipeline._initialize_kv_bank = lambda *_args, **_kwargs: None
    pipeline._set_all_modules_max_attention_size = lambda *_args, **_kwargs: None
    pipeline.clear_kv_cache = lambda: None
    pipeline._get_current_transition_alpha = lambda: None
    pipeline._update_transition_state = lambda *_args, **_kwargs: None

    pipeline.text_encoder = lambda text_prompts: {"prompt_embeds": torch.zeros((1, 1, 1))}
    pipeline.generator = DummyGenerator()
    pipeline.vae = DummyVAE()
    pipeline.scheduler = DummyScheduler()
    pipeline.args = SimpleNamespace(
        context_noise=0,
        model_kwargs=SimpleNamespace(local_attn_size=-1),
    )

    pipeline.num_frame_per_block = 3
    pipeline.frame_seq_length = 1
    pipeline.bank_size = 1
    pipeline.local_attn_size = -1
    pipeline.denoising_step_list = [1]
    pipeline.record_interval = 1
    pipeline.kv_cache1 = []
    pipeline.kv_bank1 = []
    pipeline.crossattn_cache = []
    pipeline.prev_crossattn_cache = None
    pipeline.current_entities = []
    pipeline.current_chunk_id = 0
    pipeline._iam_bank_length = 0

    noise = torch.zeros((1, 3, 1, 1, 1))

    pipeline.inference(
        noise,
        text_prompts_list=[["a"], ["b"]],
        switch_frame_indices=[0],
        save_mapping=False,
    )

    assert called_soft == [True]
    assert called_recache == []
