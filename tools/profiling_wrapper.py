# src/rl_agent/utils/profiling_wrapper.py
import time
import os
import logging
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Reuse your existing processor from the World Model
from world_model.utils.profiler_python_utils import process_profile_json,  process_torch_trace_json

logger = logging.getLogger(__name__)

class AuraProfiler:
    """
    Wrapper around pyinstrument AND torch.profiler.
    Handles start/stop and auto-summarization in a distributed Ray environment.
    """
    def __init__(self, log_dir: str, actor_name: str, enabled: bool = True, enable_python: bool = True, enable_torch: bool = True, schedule_config: dict = None):
        self.enabled = enabled
        
        # Logic: Master Switch (enabled) must be True for sub-profilers to active
        self.enable_python = enable_python and enabled
        self.enable_torch = enable_torch and enabled
        
        # If actor_name is empty, use log_dir directly (Standalone mode)
        if actor_name:
            self.log_dir = os.path.join(log_dir, "profiling", actor_name)
        else:
            self.log_dir = log_dir

        self.actor_name = actor_name or "Profiler"
        
        self.py_profiler = None
        self.torch_profiler = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
            
        # 1. Initialize Python Profiler (CPU Logic)
        if self.enable_python:
            try:
                from pyinstrument import Profiler
                self.py_profiler = Profiler(interval=0.001) # 1ms resolution
            except ImportError:
                logger.warning("pyinstrument not installed. Python profiling disabled.")

        # 2. Initialize Torch Profiler (GPU/Kernel Logic)
        if self.enable_torch:
            try:
                # [SAFETY CRITICAL] Prevent massive trace files (7GB+)
                # We strictly require a schedule. If missing, we DISABLE Torch profiling.
                if not schedule_config:
                    logger.warning(f"[{self.actor_name}] Torch Profiler enabled but 'schedule_config' is MISSING. "
                                   "Disabling Torch Profiler to prevent accidental huge trace files (Continuous Mode).")
                    self.enable_torch = False
                    self.torch_profiler = None
                else:
                    # Activities: CPU (Operators) and CUDA (Kernels)
                    activities = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(ProfilerActivity.CUDA)

                    # [STRICT CONFIG] Force KeyError if keys are missing. No defaults.
                    sch = torch.profiler.schedule(
                        wait=schedule_config['wait'],
                        warmup=schedule_config['warmup'],
                        active=schedule_config['active'],
                        repeat=schedule_config['repeat']
                    )

                    self.torch_profiler = profile(
                        activities=activities,
                        schedule=sch, 
                        on_trace_ready=self._on_trace_ready,
                        record_shapes=True,
                        # Disable memory to prevent OOM
                        profile_memory=False, 
                        with_stack=True 
                    )
            except KeyError as ke:
                logger.error(f"[{self.actor_name}] Torch Profiler Config Error: Missing key {ke}. Profiling disabled.")
                self.torch_profiler = None
            except Exception as e:
                logger.error(f"Failed to init Torch Profiler: {e}")
                self.torch_profiler = None


    def step(self):
        """Must be called every training step to advance the schedule."""
        if self.enabled and self.torch_profiler:
            self.torch_profiler.step()


    def _on_trace_ready(self, p):
        """Callback for scheduled traces."""
        tag = f"step_{p.step_num}"
        self._save_torch_trace(p, tag)


    def _save_torch_trace(self, p, tag):
        try:
            trace_path = os.path.join(self.log_dir, f"trace_{tag}.json")
            try:
                p.export_chrome_trace(trace_path)
            except (RuntimeError, AssertionError):
                # AssertionError: Common if stopped mid-schedule before valid data
                logger.warning(f"[{self.actor_name}] Profiler invalid state or empty. Skipping export.")
                return

            logger.info(f"[{self.actor_name}] Torch trace saved: {trace_path}")
            summary_path = os.path.join(self.log_dir, f"summary_trace_{tag}.txt")
            process_torch_trace_json(trace_path, summary_path)
            # Remove intermediate JSON after summary is generated
            if os.path.exists(summary_path):
                os.remove(trace_path)
                logger.info(f"[{self.actor_name}] Cleaned up intermediate trace: {trace_path}")

        except BaseException as e:
            logger.error(f"[{self.actor_name}] Error processing/saving trace: {repr(e)}")


    def stop_and_save(self, tag: str):
        if not self.enabled: return

        if self.torch_profiler:
            try: self.torch_profiler.stop()
            except BaseException: pass 

            # [SAFETY FIX] If using a schedule, we rely on _on_trace_ready callback.
            # Forcing an export here often crashes or corrupts data if the schedule isn't complete.
            if self.torch_profiler.schedule is None:
                 self._save_torch_trace(self.torch_profiler, tag)

        # --- Stop & Save Python Profiler ---
        if self.py_profiler and self.py_profiler.is_running:
            try:
                self.py_profiler.stop()
                
                # 1. Save HTML (Intermediate)
                html_path = os.path.join(self.log_dir, f"py_profile_{tag}.html")
                with open(html_path, "w", encoding='utf-8') as f:
                    f.write(self.py_profiler.output_html())
                
                # 2. Save JSON (Intermediate — needed for summary generation)
                json_path = os.path.join(self.log_dir, f"py_profile_{tag}.json")
                from pyinstrument.renderers import JSONRenderer
                with open(json_path, "w", encoding='utf-8') as f:
                    f.write(self.py_profiler.output(renderer=JSONRenderer()))
                    
                # 3. Generate Summary Text
                summary_path = os.path.join(self.log_dir, f"summary_{tag}.txt")
                process_profile_json(json_path, summary_path)
                
                # Remove intermediate .json and .html after summary is generated
                if os.path.exists(summary_path):
                    os.remove(json_path)
                    os.remove(html_path)
                    logger.info(f"[{self.actor_name}] Cleaned up intermediate files: {json_path}, {html_path}")

                self.py_profiler.reset()
            except BaseException as e:
                logger.error(f"[{self.actor_name}] Error saving python profile: {e}")
    
    def start(self):
        if not self.enabled: return
        # Start Python Profiler
        if self.py_profiler and not self.py_profiler.is_running:
            self.py_profiler.start()
        # Start Torch Profiler
        if self.torch_profiler:
            self.torch_profiler.start()