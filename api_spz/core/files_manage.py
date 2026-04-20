# api_spz/core/files_manage.py
import os
from pathlib import Path
from api_spz.core.state_manage import state


class FileManager:
    def __init__(self):
        self.base_path = state.temp_dir  # points to "temp"

        # We store everything for the active generation in "temp/current_generation"
        self.generation_path = self.base_path / "current_generation"
        self.generation_path.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, filename: str) -> Path:
        """Returns the path for a single file (e.g. 'input.png', 'model.glb')."""
        return self.generation_path / filename

    def cleanup_generation_files(self, keep_model: bool = False):
        """Clean up files from the 'current_generation' folder."""
        try:
            for file_name in os.listdir(self.generation_path):
                if not keep_model and file_name.startswith("model"):
                    os.remove(self.get_temp_path(file_name))
                elif file_name == "input.png":
                    os.remove(self.get_temp_path(file_name))
        except Exception as e:
            print(f"Error cleaning up generation files: {e}")


file_manager = FileManager()