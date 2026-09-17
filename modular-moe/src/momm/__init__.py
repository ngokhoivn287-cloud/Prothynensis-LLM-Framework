"""MoMMs package - Mix of Many Models."""

from .model import Model, ModelConfig, create_model
from .registry import ModelRegistry, create_registry
from .coordinator import ModelCoordinator, CoordinatorConfig
from .fusion import FusionLayer, FusionConfig, create_fusion
from .inference import InferenceEngine, InferenceConfig
from .memory import MemoryManager, MemoryRecord, MemoryType
from .task import TaskManager, TaskState, Subtask, TaskState as TaskStatus
from .communication import PeerCommunicationProtocol, PeerMessage, MessageType
from .collaboration import CollaborativeSynthesizer, CandidateResult, SynthesisResult, SynthesisStrategy
from .simulation import SimulationEngine, SimulatedModel, SimulationResult, create_simulation_engine
from .recruitment import DynamicRecruiter, RecruitmentContext, ActivationBudget, RecruitmentBudget, RecruitmentStrategy

__all__ = [
    "Model",
    "ModelConfig",
    "create_model",
    "ModelRegistry",
    "create_registry",
    "ModelCoordinator",
    "CoordinatorConfig",
    "FusionLayer",
    "FusionConfig",
    "create_fusion",
    "InferenceEngine",
    "InferenceConfig",
    "MemoryManager",
    "MemoryRecord",
    "MemoryType",
    "TaskManager",
    "TaskState",
    "Subtask",
    "TaskStatus",
    "PeerCommunicationProtocol",
    "PeerMessage",
    "MessageType",
    "CollaborativeSynthesizer",
    "CandidateResult",
    "SynthesisResult",
    "SynthesisStrategy",
    "SimulationEngine",
    "SimulatedModel",
    "SimulationResult",
    "create_simulation_engine",
    "DynamicRecruiter",
    "RecruitmentContext",
    "ActivationBudget",
    "RecruitmentBudget",
    "RecruitmentStrategy",
]

__version__ = "0.2.0"
