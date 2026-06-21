import pytest
from PIL import Image
from unittest.mock import AsyncMock, patch

from core.pipeline import PipelineOrchestrator, ModuleRegistry
from core.model_manager import ModelManager
from core.base import BaseAnalyzer
from core.events import EventType

# Create some dummy modules for testing the pipeline orchestration

class DummyCPUModule(BaseAnalyzer):
    name = "dummy_cpu"
    display_name = "Dummy CPU"
    estimated_vram_mb = 0
    requires_gpu = False
    stage = 0

    async def load_model(self, device: str = "cpu") -> None:
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs) -> dict:
        return {"dummy_cpu_result": "cpu_ok"}

    async def unload_model(self) -> None:
        self._is_loaded = False


class DummyGPUModule1(BaseAnalyzer):
    name = "dummy_gpu1"
    display_name = "Dummy GPU 1"
    estimated_vram_mb = 500
    requires_gpu = True
    stage = 1

    async def load_model(self, device: str = "cpu") -> None:
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs) -> dict:
        return {"dummy_gpu_result_1": "gpu1_ok"}

    async def unload_model(self) -> None:
        self._is_loaded = False


class DummyGPUModule2(BaseAnalyzer):
    name = "dummy_gpu2"
    display_name = "Dummy GPU 2"
    estimated_vram_mb = 10000  # huge VRAM requirement
    requires_gpu = True
    stage = 2

    async def load_model(self, device: str = "cpu") -> None:
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs) -> dict:
        return {"dummy_gpu_result_2": "gpu2_ok"}

    async def unload_model(self) -> None:
        self._is_loaded = False


@pytest.fixture
def mock_pipeline():
    # 12GB budget
    manager = ModelManager(vram_budget_mb=12000, device="cpu") 
    registry = ModuleRegistry()
    registry.register(DummyCPUModule())
    registry.register(DummyGPUModule1())
    registry.register(DummyGPUModule2())
    
    return PipelineOrchestrator(registry, manager)


@pytest.mark.asyncio
async def test_pipeline_execution(mock_pipeline):
    """Test that the pipeline correctly stages modules and emits the correct events."""
    
    # Create a simple 10x10 dummy image
    test_image = Image.new("RGB", (10, 10), color="red")
    
    events = []
    async for event in mock_pipeline.analyze(test_image):
        events.append(event)
        
    # Check that we got the start and complete events
    assert any(e.event == EventType.PIPELINE_START for e in events)
    assert any(e.event == EventType.PIPELINE_COMPLETE for e in events)
    
    # Check that each module completed
    module_completions = [e for e in events if e.event == EventType.MODULE_COMPLETE]
    assert len(module_completions) == 3
    
    # Verify the results made it into the pipeline complete event
    complete_event = events[-1]
    assert complete_event.data is not None
    results = complete_event.data.get("results", {})
    # The pipeline uses a strict Pydantic schema for results, so arbitrary dummy keys are dropped.
    # We instead verify that our modules executed successfully by checking the timing metadata.
    timings = results.get("module_timings", {})
    assert "dummy_cpu" in timings
    assert "dummy_gpu1" in timings
    assert "dummy_gpu2" in timings


@pytest.mark.asyncio
async def test_vram_eviction():
    """Test that the ModelManager evicts modules when budget is exceeded."""
    manager = ModelManager(vram_budget_mb=11000, device="cpu") 
    
    mod1 = DummyGPUModule1() # 500 MB
    mod2 = DummyGPUModule2() # 10000 MB
    
    # Load first model
    await manager.ensure_loaded(mod1)
    assert manager.vram_used_mb == 500
    assert mod1.name in manager.loaded_modules
    
    # Load second model (budget fits both: 10500 < 11000)
    await manager.ensure_loaded(mod2)
    assert manager.vram_used_mb == 10500
    
    # Now create a third model that demands 2000 MB (total would be 12500)
    class GreedyModule(BaseAnalyzer):
        name = "greedy"
        display_name = "Greedy"
        estimated_vram_mb = 2000
        stage = 3
        async def load_model(self, device="cpu"): self._is_loaded = True
        async def analyze(self, image, **kwargs): return {}
        async def unload_model(self): self._is_loaded = False
        
    greedy = GreedyModule()
    
    # This should force an eviction of mod1 (the least recently used)
    await manager.ensure_loaded(greedy)
    
    # VRAM should now be 10000 + 2000 = 12000... wait, 12000 > 11000!
    # It will actually need to evict both if possible, or fail.
    # In our implementation, it evicts LRU until it fits.
    assert manager.vram_used_mb <= 11000


@pytest.mark.asyncio
async def test_oom_retry_logic():
    """Test that models recover from CUDA Out of Memory errors during analysis."""
    
    class DummyOOMModule(BaseAnalyzer):
        name = "dummy_oom"
        display_name = "Dummy OOM"
        estimated_vram_mb = 100
        requires_gpu = True
        stage = 1
        
        def __init__(self):
            super().__init__()
            self.attempts = 0
            
        async def load_model(self, device="cpu"):
            self._is_loaded = True
            
        async def analyze(self, image, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                # Simulate a PyTorch CUDA OOM Error on the first try
                raise RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB.")
            return {"recovered": True, "attempts": self.attempts}
            
        async def unload_model(self):
            self._is_loaded = False
            
    module = DummyOOMModule()
    await module.load_model()
    
    # Run the analysis; it should fail internally on attempt 1, trigger the retry, and succeed on attempt 2
    test_image = Image.new("RGB", (10, 10), color="blue")
    result = await module.run(test_image)
    
    assert result.success is True
    assert result.data["recovered"] is True
    assert result.data["attempts"] == 2
