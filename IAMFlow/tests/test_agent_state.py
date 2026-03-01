import importlib.util
import os
import sys
import types

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
    _install_module("utils.hsa_schedule", {"compute_prompt_sparse_policy": lambda **_: (False, None)})
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


class DummyMemoryBank:
    def __init__(self) -> None:
        self.cleared = False
        self.global_registry = {"dummy": 1}

    def clear(self) -> None:
        self.cleared = True
        self.global_registry = {}


class DummyIdManager:
    def __init__(self) -> None:
        self._next_id = 99


class DummyLLMAgent:
    def __init__(self) -> None:
        self.id_manager = DummyIdManager()


class _StubEntity:
    def __init__(self, entity, attrs, global_id):
        self.entity = entity
        self.attrs = attrs
        self.global_id = global_id


class DummyLLMAgentSceneOnly(DummyLLMAgent):
    def process_prompt(self, prompt, prompt_id, global_registry):
        # Return one entity so the retrieval path is exercised
        e = _StubEntity("forest spirit", ["snowy"], 1)
        return [e], {}, ["snowy forest", "pine trees"]


class DummyMemoryBankPromptStart:
    def __init__(self) -> None:
        self.global_registry = {}
        self.id_memory = []
        self.retrieve_calls = []
        self.register_calls = []

    @property
    def frame_active_memory(self):
        return list(dict.fromkeys(self.id_memory))

    def register_entities(self, entities, prompt_id, registry_update):
        self.register_calls.append((entities, prompt_id, registry_update))

    def get_entity_ids(self, entities):
        return []

    def retrieve_initial_frames(self, entity_ids, scene_texts=None):
        self.retrieve_calls.append((entity_ids, scene_texts))
        return self.frame_active_memory


class DummyFrameInfo:
    def __init__(self, entity_score: float) -> None:
        self.entity_score = entity_score


class DummyMemoryBankEviction:
    def __init__(self) -> None:
        self.frame_archive = {}
        self.id_memory = []
        self.update_id_calls = []

    @property
    def frame_active_memory(self):
        return list(dict.fromkeys(self.id_memory))

    def get_entity_ids(self, entities):
        return []

    def select_frame_from_chunk(self, **kwargs):
        self.frame_archive["p2_c3_f0"] = DummyFrameInfo(entity_score=0.1)
        return "p2_c3_f0", 0.42

    def update_id_memory(self, frame_id, entity_score):
        self.update_id_calls.append((frame_id, entity_score))
        if frame_id not in self.id_memory:
            self.id_memory.append(frame_id)


def test_reset_agent_state_resets_iam_bank_length():
    AgentCausalInferencePipeline = _load_agent_pipeline_class()
    pipeline = AgentCausalInferencePipeline.__new__(AgentCausalInferencePipeline)
    pipeline.current_prompt_id = 3
    pipeline.current_chunk_id = 7
    pipeline.current_entities = ["x"]
    pipeline.current_prompt_text = "hello"
    pipeline.agent_memory_bank = DummyMemoryBank()
    pipeline.llm_agent = DummyLLMAgent()
    pipeline._iam_bank_length = 5

    pipeline._reset_agent_state()

    assert pipeline.current_prompt_id == 0
    assert pipeline.current_chunk_id == 0
    assert pipeline.current_entities == []
    assert pipeline.current_prompt_text == ""
    assert pipeline.agent_memory_bank.cleared is True
    assert pipeline.llm_agent.id_manager._next_id == 1
    assert pipeline._iam_bank_length == 0


def test_process_prompt_start_retrieves_memory():
    AgentCausalInferencePipeline = _load_agent_pipeline_class()
    pipeline = AgentCausalInferencePipeline.__new__(AgentCausalInferencePipeline)
    pipeline.current_prompt_id = 0
    pipeline.current_chunk_id = 0
    pipeline.current_entities = []
    pipeline.current_prompt_text = ""
    pipeline.llm_agent = DummyLLMAgentSceneOnly()
    pipeline.agent_memory_bank = DummyMemoryBankPromptStart()
    pipeline.injected = False
    pipeline._inject_iam_memory_to_bank = lambda: setattr(pipeline, "injected", True)

    pipeline._process_prompt_start(
        prompt_text="A quiet snowy forest landscape",
        prompt_id=2,
        is_first_prompt=False,
    )

    assert len(pipeline.current_entities) == 1
    assert len(pipeline.agent_memory_bank.retrieve_calls) == 1
    assert pipeline.injected is True


def test_process_chunk_eviction_updates_id_memory():
    AgentCausalInferencePipeline = _load_agent_pipeline_class()
    pipeline = AgentCausalInferencePipeline.__new__(AgentCausalInferencePipeline)
    pipeline.current_prompt_id = 2
    pipeline.current_chunk_id = 3
    pipeline.current_entities = ["some_entity"]
    pipeline.current_prompt_text = "A snowy forest scene"
    pipeline.crossattn_cache = [{"is_init": True}]
    pipeline.agent_memory_bank = DummyMemoryBankEviction()
    pipeline._get_evicted_chunk_kv = lambda: [{
        "k": torch.zeros(1, 3, 1, 1),
        "v": torch.zeros(1, 3, 1, 1),
    }]

    pipeline._process_chunk_eviction(current_start_frame=6, current_num_frames=3)

    assert pipeline.agent_memory_bank.update_id_calls == [("p2_c3_f0", 0.1)]
