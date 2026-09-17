"""Native Inference Engine for MoMMs."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Union
from dataclasses import dataclass, field
from dataclasses import asdict
from collections import OrderedDict

import torch
import torch.nn as nn

from .model import Model, ModelConfig, create_model
from .registry import ModelRegistry, ModelMetadata, create_registry
from .coordinator import ModelCoordinator, CoordinatorConfig, create_coordinator
from .fusion import FusionLayer, FusionConfig, create_fusion


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    max_gpu_models: int = 2
    max_cpu_models: int = 8
    lazy_loading: bool = True
    cpu_offload: bool = True
    default_top_k: int = 2
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    default_max_new_tokens: int = 100
    default_repetition_penalty: float = 1.0
    streaming: bool = True
    compile_models: bool = False
    use_flash_attn: bool = False
    
    def to_dict(self) -> dict:
        return {
            "max_gpu_models": self.max_gpu_models,
            "max_cpu_models": self.max_cpu_models,
            "lazy_loading": self.lazy_loading,
            "cpu_offload": self.cpu_offload,
            "default_top_k": self.default_top_k,
            "default_temperature": self.default_temperature,
            "default_top_p": self.default_top_p,
            "default_max_new_tokens": self.default_max_new_tokens,
            "default_repetition_penalty": self.default_repetition_penalty,
            "streaming": self.streaming,
            "compile_models": self.compile_models,
            "use_flash_attn": self.use_flash_attn,
        }


class ModelCache:
    """LRU cache for models with GPU/CPU management."""
    
    def __init__(
        self,
        max_gpu: int = 2,
        max_cpu: int = 8,
        cpu_offload: bool = True,
    ):
        self.max_gpu = max_gpu
        self.max_cpu = max_cpu
        self.cpu_offload = cpu_offload
        
        self._gpu_cache: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._cpu_cache: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._access_order: List[str] = []
    
    def get(self, model_id: str) -> Optional[torch.nn.Module]:
        """Get model from cache, moving to GPU if needed."""
        # Check GPU cache
        if model_id in self._gpu_cache:
            self._gpu_cache.move_to_end(model_id)
            return self._gpu_cache[model_id]
        
        # Check CPU cache
        if model_id in self._cpu_cache:
            model = self._cpu_cache.pop(model_id)
            return self._promote_to_gpu(model_id, model)
        
        return None
    
    def put(self, model_id: str, model: torch.nn.Module, device: torch.device) -> None:
        """Put model in cache."""
        if device.type == "cuda":
            self._evict_gpu_if_needed()
            self._gpu_cache[model_id] = model
        else:
            self._evict_cpu_if_needed()
            self._cpu_cache[model_id] = model
    
    def _promote_to_gpu(self, model_id: str, model: torch.nn.Module):
        """Move model from CPU to GPU."""
        self._evict_gpu_if_needed()
        model = model.cuda()
        self._gpu_cache[model_id] = model
        return model
    
    def _evict_gpu_if_needed(self):
        while len(self._gpu_cache) >= self.max_gpu:
            model_id, model = self._gpu_cache.popitem(last=False)
            if self.cpu_offload and len(self._cpu_cache) < self.max_cpu:
                self._cpu_cache[model_id] = model.cpu()
            else:
                del model
    
    def _evict_cpu_if_needed(self):
        while len(self._cpu_cache) >= self.max_cpu:
            model_id, model = self._cpu_cache.popitem(last=False)
            del model
    
    def remove(self, model_id: str) -> bool:
        removed = False
        if model_id in self._gpu_cache:
            del self._gpu_cache[model_id]
            removed = True
        if model_id in self._cpu_cache:
            del self._cpu_cache[model_id]
            removed = True
        return removed
    
    def clear(self):
        for model in self._gpu_cache.values():
            del model
        for model in self._cpu_cache.values():
            del model
        self._gpu_cache.clear()
        self._cpu_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class InferenceEngine:
    """
    Native Inference Engine for MoMMs.
    
    Manages model loading, coordination, and generation.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        registry: Optional[ModelRegistry] = None,
        coordinator: Optional[ModelCoordinator] = None,
        registry_dir: str = "models",
    ):
        self.config = config
        self.registry = registry or create_registry(
            registry_dir="models",
            max_gpu_models=config.max_gpu_models,
            max_cpu_models=config.max_cpu_models,
            lazy_loading=config.lazy_loading,
            cpu_offload=config.cpu_offload,
        )
        self.coordinator = coordinator
        self.cache = ModelCache(
            max_gpu=config.max_gpu_models,
            max_cpu=config.max_cpu_models,
            cpu_offload=config.cpu_offload,
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_generation_config = {
            "max_new_tokens": config.default_max_new_tokens,
            "temperature": config.default_temperature,
            "top_k": config.default_top_k,
            "top_p": config.default_top_p,
            "repetition_penalty": config.default_repetition_penalty,
        }
        
        self._active_model_ids: List[str] = []
        self._tokenizer = None
    
    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer for inference."""
        self._tokenizer = tokenizer
    
    def get_tokenizer(self):
        """Get tokenizer."""
        return self._tokenizer
    
    def load_system(
        self,
        coordinator_config: Optional[CoordinatorConfig] = None,
        model_ids: Optional[List[str]] = None,
    ) -> ModelCoordinator:
        """Load the full MoMMs system."""
        
        if self.coordinator is not None:
            return self.coordinator
        
        # Determine model IDs to load
        if model_ids is None:
            model_ids = self.registry.get_model_ids()
        
        # Load models
        models = []
        for model_id in model_ids:
            model = self.registry.load_model(model_id)
            models.append(model)
        
        # Create coordinator
        if coordinator_config is None:
            coordinator_config = CoordinatorConfig(
                hidden_dim=models[0].config.hidden_dim,
                num_models=len(models),
                top_k=2,
            )
        
        self.coordinator = ModelCoordinator(models=models, config=coordinator_config)
        
        if torch.cuda.is_available():
            self.coordinator = self.coordinator.cuda()
        
        if self.config.compile_models:
            self.coordinator = torch.compile(self.coordinator)
        
        return self.coordinator
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        top_k_models: Optional[int] = None,
        stream: bool = False,
        eos_token_id: Optional[int] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Repetition penalty
            top_k_models: Number of models to route to (overrides coordinator config)
            stream: Whether to stream tokens
            eos_token_id: End-of-sequence token ID
            stop_strings: Stop generation on these strings
        
        Returns:
            Generated text (or iterator if stream=True)
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        if self.coordinator is None:
            self.load_system()
        
        # Merge generation config
        gen_config = self.default_generation_config.copy()
        if max_new_tokens is not None:
            gen_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_k is not None:
            gen_config["top_k"] = top_k
        if top_p is not None:
            gen_config["top_p"] = top_p
        if repetition_penalty is not None:
            gen_config["repetition_penalty"] = repetition_penalty
        
        # Override top_k_models if provided
        if top_k_models is not None:
            self.coordinator.top_k = top_k_models
        
        # Tokenize
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.coordinator.device)
        
        # Generate
        if stream:
            return self._stream_generate(
                input_ids,
                gen_config,
                eos_token_id=eos_token_id,
                stop_strings=stop_strings,
            )
        else:
            output_ids = self.coordinator.generate(
                input_ids,
                max_new_tokens=gen_config["max_new_tokens"],
                temperature=gen_config["temperature"],
                top_k=gen_config["top_k"],
                top_p=gen_config["top_p"],
                repetition_penalty=gen_config["repetition_penalty"],
                eos_token_id=eos_token_id,
            )
            return self._tokenizer.decode(output_ids[0].tolist())
    
    def _stream_generate(
        self,
        input_ids: torch.Tensor,
        gen_config: dict,
        eos_token_id: Optional[int] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Stream generated tokens."""
        self.coordinator.eval()
        
        generated = input_ids
        generated_text = ""
        
        for _ in range(gen_config["max_new_tokens"]):
            with torch.no_grad():
                outputs = self.coordinator(
                    generated,
                    use_cache=True,
                    return_aux_loss=False,
                    return_routing_stats=False,
                )
            
            logits = outputs["logits"][:, -1, :] / gen_config["temperature"]
            
            # Repetition penalty
            if gen_config["repetition_penalty"] != 1.0:
                for i in range(generated.size(0)):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= gen_config["repetition_penalty"]
            
            # Top-k
            if gen_config["top_k"] is not None:
                top_k = min(gen_config["top_k"], logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
                logits[indices_to_remove] = float("-inf")
            
            # Top-p
            if gen_config["top_p"] is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > gen_config["top_p"]
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Decode new token
            new_token_text = self._tokenizer.decode(next_token[0].tolist())
            generated_text += new_token_text
            
            # Check stop strings
            if stop_strings:
                for stop_str in stop_strings:
                    if stop_str in generated_text:
                        break
            
            yield new_token_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            ... generation parameters
        """
        # Format conversation
        prompt = self._format_chat(messages)
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
        )
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n".join(formatted)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics from coordinator."""
        if self.coordinator is None:
            return {}
        return self.coordinator.get_routing_stats()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "cache": {
                "gpu_models": len(self.cache._gpu_cache),
                "cpu_models": len(self.cache._cpu_cache),
            },
            "registry": self.registry.get_memory_stats(),
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return stats
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model."""
        self.registry.unload_model(model_id)
        self.cache.remove(model_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    def unload_all(self):
        """Unload all models."""
        self.cache.clear()
        self.registry = create_registry()
        self.coordinator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "coordinator_loaded": self.coordinator is not None,
            "num_models_registered": len(self.registry.models),
            "models_loaded": self.cache._gpu_cache.__len__() + self.cache._cpu_cache.__len__(),
            "cache": self.get_memory_stats(),
        }
    
    def save_state(self, path: str):
        """Save inference engine state."""
        state = {
            "config": self.config.to_dict(),
            "active_models": list(self.cache._gpu_cache.keys()) + list(self.cache._cpu_cache.keys()),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        registry_dir: str = "models",
    ) -> "InferenceEngine":
        """Create inference engine from config file."""
        import yaml
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = InferenceConfig(**config_dict.get("inference", {}))
        return cls(config=config, registry_dir=registry_dir)


def create_inference_engine(
    config: InferenceConfig | Dict[str, Any],
    registry: Optional[ModelRegistry] = None,
    coordinator: Optional[ModelCoordinator] = None,
    registry_dir: str = "models",
) -> InferenceEngine:
    """Factory function to create inference engine."""
    if isinstance(config, dict):
        config = InferenceConfig(**config)
    return InferenceEngine(
        config=config,
        registry=registry,
        coordinator=coordinator,
        registry_dir=registry_dir,
    )