# install.py
# File: install.py
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


def download_models():
    """Download and extract model zips from GitHub releases if not already present."""
    CODE_DIR = get_current_script_dir()
    models_dir = CODE_DIR / "MODELS"
    models_dir.mkdir(exist_ok=True)

    MODELS = [
        {
            "name": "dinov3",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/dinov3.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/dinov3.zip/download",
            ],
            "check_file": models_dir / "dinov3" / "model.safetensors",
        },
        {
            "name": "RMBG-2.0",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/RMBG-2.0.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/RMBG-2.0.zip/download",
            ],
            "check_file": models_dir / "RMBG-2.0" / "model.safetensors",
        },
    ]

    for model in MODELS:
        if model["check_file"].exists():
            print(f"[INFO] {model['name']} already present, skipping download.")
            continue

        zip_path = models_dir / f"{model['name']}.zip"
        downloaded = False

        for url_idx, url in enumerate(model["urls"]):
            if downloaded:
                break
            mirror_label = f"mirror {url_idx + 1}/{len(model['urls'])}"
            print(f"\nDownloading {model['name']} from {mirror_label}: {url}")

            for attempt in range(MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retry attempt {attempt + 1}/{MAX_RETRIES}...")
                        time.sleep(RETRY_DELAY)

                    def _reporthook(block_num, block_size, total_size):
                        dl = block_num * block_size
                        if total_size > 0:
                            pct = min(dl * 100 / total_size, 100)
                            mb_down = dl / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

                    urllib.request.urlretrieve(url, str(zip_path), reporthook=_reporthook)
                    print()  # newline after progress
                    downloaded = True
                    break
                except Exception as e:
                    print(f"\n  Download failed: {e}")
                    if zip_path.exists():
                        zip_path.unlink()
                    if attempt == MAX_RETRIES - 1:
                        print(f"  All {MAX_RETRIES} attempts failed for {mirror_label}. Trying next mirror...")

        if not downloaded:
            raise InstallationError(
                f"Failed to download {model['name']} from all {len(model['urls'])} mirrors"
            )

        # Extract
        print(f"  Extracting {model['name']}...")
        import zipfile
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(models_dir))
        zip_path.unlink()

        if not model["check_file"].exists():
            raise InstallationError(
                f"Extraction succeeded but {model['check_file'].name} not found. "
                f"Check that the zip contains a '{model['name']}/' folder at its root."
            )
        print(f"  {model['name']} ready.")


def download_hf_models():
    """Pre-download HuggingFace model weights so the app doesn't timeout on first launch."""

    print("\n============================================================================= ")
    print(" Downloading TRELLIS.2 model weights from HuggingFace.")
    print(" This may take 10-30 minutes on first install (~20GB total). Please wait")
    print(  "============================================================================= ")
    
    # Import here because huggingface_hub is installed in the previous step
    from huggingface_hub import snapshot_download
    
    CODE_DIR = get_current_script_dir()
    cache_dir = str(CODE_DIR / "models" / "hub")
    
    repos = [
        "microsoft/TRELLIS.2-4B",
        "microsoft/TRELLIS-image-large",
    ]
    
    for repo_id in repos:
        print(f"  Downloading {repo_id}...")
        for attempt in range(MAX_RETRIES):
            try:
                snapshot_download(
                    repo_id,
                    cache_dir=cache_dir,
                    resume_download=True,
                )
                print(f"  {repo_id} ready.")
                break
            except Exception as e:
                print(f"  Download failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise InstallationError(f"Failed to download {repo_id} after {MAX_RETRIES} attempts")


def install_dependencies():
    """Install Trellis 2 dependencies."""
    CODE_DIR = get_current_script_dir()
    check_python_version()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)
        
        # 0. Download model weights (dinov3, RMBG-2.0) if not already present
        download_models()

        # 1. PyTorch 2.8.0 + CUDA 12.8
        print("\n--- Installing PyTorch 2.8.0 (CUDA 12.8) ---")
        torch_cmd = "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
        run_command_with_retry(torch_cmd, "Installing PyTorch")

        # 2. General Dependencies
        print("\n--- Installing General Dependencies ---")
        general_deps = [
            "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
            "ninja", "trimesh", "transformers==4.57.3", "gradio==6.0.1", "tensorboard",
            "pandas", "lpips", "zstandard", "kornia", "timm", 
            "huggingface_hub", "accelerate", "psutil", "triton-windows==3.4.0.post21"
        ]
        run_command_with_retry(f"pip install {' '.join(general_deps)}", "Installing pip packages")

        # 2.5. Pre-download HuggingFace model weights
        download_hf_models()

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
                run_command_with_retry(f'pip install "{simd_wheels[0]}"', "Installing Pillow-SIMD")
                
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
        
        for whl_file in sorted(whl_dir.glob("*.whl")):
            # Pillow-SIMD is handled above
            if whl_file.name.lower().startswith("pillow"):
                continue
            print(f"Installing: {whl_file.name}")
            run_command_with_retry(f'pip install "{whl_file}"', f"Installing {whl_file.name}")

        # Fallback for utils3d if no wheel was present
        if not list(whl_dir.glob("utils3d*.whl")):
            print("Utils3D wheel not found. Installing via Git...")
            run_command_with_retry(
                "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
                "Installing Utils3D (Git)"
            )

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