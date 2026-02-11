import subprocess
import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUIRED_PYTHON = (3, 11) # Matches cp311 wheels

class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass

def get_current_script_dir() -> Path:
    """Helper to get the directory of the current script."""
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path(os.getcwd()).resolve()

def check_python_version():
    """Ensure we are running on the correct Python version for the wheels."""
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"Error: This installer requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}")
        print(f"You are currently using Python {current[0]}.{current[1]}")
        print("Please use the embedded python or correct environment.")
        sys.exit(1)

def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Check internet connectivity."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        if isinstance(reason, socket.gaierror):
            return False, f"DNS resolution failed: {reason}"
        elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def get_git_env() -> dict:
    """Return a copy of the current environment configured to use the portable Git."""
    env = os.environ.copy()
    CODE_DIR = get_current_script_dir()
    PORTABLE_GIT_BASE = (CODE_DIR / ".." / "tools" / "git").resolve()
    
    if PORTABLE_GIT_BASE.exists():
        git_paths = [
            str(PORTABLE_GIT_BASE / "mingw64" / "bin"),
            str(PORTABLE_GIT_BASE / "cmd"),
            str(PORTABLE_GIT_BASE / "usr" / "bin"),
        ]
        existing_path = env.get("PATH", "")
        env["PATH"] = ";".join(git_paths) + (";" + existing_path if existing_path else "")
        
        ca_bundle = PORTABLE_GIT_BASE / "mingw64" / "etc" / "ssl" / "certs" / "ca-bundle.crt"
        if ca_bundle.exists():
            env["GIT_SSL_CAINFO"] = str(ca_bundle)
            env["SSL_CERT_FILE"]  = str(ca_bundle)
    
    return env

def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES, fatal: bool = True) -> subprocess.CompletedProcess:
    """Run a command with retry logic."""
    last_error = None
    env = get_git_env()
    
    if cmd.startswith('pip install'):
        args = cmd[11:]
        cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'

    if "pip install" in cmd and "--progress-bar" not in cmd:
        cmd += " --progress-bar=on"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue 
            
            if "pip install" in cmd:
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                return result
            
            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")
            
        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))
        
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY)
    
    if fatal:
        raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")
    else:
        print(f"Warning: Command '{desc}' failed. Continuing...")
        return last_error

def install_dependencies():
    """Install Trellis 2 dependencies."""
    CODE_DIR = get_current_script_dir()
    check_python_version()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)
        
        # 1. PyTorch 2.8.0 + CUDA 12.8
        print("\n--- Installing PyTorch 2.8.0 (CUDA 12.8) ---")
        torch_cmd = "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
        run_command_with_retry(torch_cmd, "Installing PyTorch")

        # 2. General Dependencies
        print("\n--- Installing General Dependencies ---")
        general_deps = [
            "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
            "ninja", "trimesh", "transformers", "gradio==6.0.1", "tensorboard",
            "pandas", "lpips", "zstandard", "kornia", "timm", 
            "huggingface_hub", "accelerate", "psutil"
        ]
        run_command_with_retry(f"pip install {' '.join(general_deps)}", "Installing pip packages")

        # 3. Handle Pillow Replacement (Standard -> SIMD)
        print("\n--- Configuring Pillow ---")
        # Attempt to install Pillow-SIMD wheel
        whl_dir = CODE_DIR / "whl"
        simd_wheels = list(whl_dir.glob("Pillow_SIMD*.whl"))
        
        simd_success = False
        if simd_wheels:
            print("Uninstalling standard Pillow...")
            subprocess.run(f'"{sys.executable}" -m pip uninstall -y pillow', shell=True, stdout=subprocess.DEVNULL)
            
            print(f"Installing Pillow-SIMD: {simd_wheels[0].name}")
            try:
                run_command_with_retry(f"pip install {simd_wheels[0]}", "Installing Pillow-SIMD")
                
                # --- SAFETY CHECK ---
                # Verify immediately if Pillow-SIMD actually works
                print("Verifying Pillow-SIMD...")
                check_cmd = f'"{sys.executable}" -c "from PIL import Image; print(\'Pillow OK\')"'
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                
                if check_result.returncode == 0:
                    print("Pillow-SIMD installed and verified successfully.")
                    simd_success = True
                else:
                    print(f"Warning: Pillow-SIMD installed but failed to load. (Error: {check_result.stderr.strip()})")
                    print("Falling back to standard Pillow...")
            except:
                print("Failed to install Pillow-SIMD wheel. Falling back to standard Pillow...")

        if not simd_success:
            # Revert to standard Pillow if SIMD missing or broken
            # We uninstall Pillow-SIMD first just in case
            subprocess.run(f'"{sys.executable}" -m pip uninstall -y Pillow-SIMD', shell=True, stdout=subprocess.DEVNULL)
            run_command_with_retry("pip install pillow", "Installing Standard Pillow")
        
        # 4. Install Local Wheels
        print("\n--- Installing Custom Wheels ---")
        
        # We map descriptions to glob patterns
        wheel_patterns = {
            "Nvdiffrast": "nvdiffrast*.whl",
            "Nvdiffrec": "nvdiffrec*.whl",
            "CuMesh": "cumesh*.whl",
            "FlexGEMM": "flex_gemm*.whl",
            "O-Voxel": "o_voxel*.whl",
            "Flash Attention": "flash_attn*.whl",
            "Utils3D": "utils3d*.whl"
        }

        for desc, pattern in wheel_patterns.items():
            found = list(whl_dir.glob(pattern))
            if found:
                # Install the first match found
                run_command_with_retry(f"pip install {found[0]}", f"Installing {desc} Wheel")
            else:
                if desc == "Utils3D":
                    # Fallback to git for utils3d if not compiled
                    print(f"Utils3D wheel not found. Installing via Git...")
                    run_command_with_retry(
                        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", 
                        "Installing Utils3D (Git)"
                    )
                else:
                    print(f"Warning: Wheel for {desc} (pattern: {pattern}) not found in 'whl' folder.")

        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

def verify_installation():
    """Verify installation."""
    try:
        # Check torch
        import torch
        print(f"\nVerification successful.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            
            # Verify the compiled extensions
            modules_to_check = ["nvdiffrast", "o_voxel", "flash_attn"]
            for mod in modules_to_check:
                try:
                    __import__(mod)
                    print(f"[OK] {mod} detected.")
                except ImportError:
                    print(f"[WARNING] {mod} not found.")

        # Explicit PIL Check
        try:
            from PIL import Image
            print("[OK] PIL (Pillow) detected.")
        except ImportError:
            print("[ERROR] PIL (Pillow) not found! This is required.")
            return False
            
        return True
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified!")
        print("You can now run 'run_app.py' or 'app.py'")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)