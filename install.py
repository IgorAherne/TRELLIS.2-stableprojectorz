# install.py
# File: install.py
import subprocess
import sys
import os
import time
import json
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUIRED_PYTHON = (3, 11) # Matches cp311 wheels

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES = "torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128"
XFORMERS_PACKAGE = "xformers==0.0.32.post2"

# Exact versions of every package taken from PyPI. Without it pip picks the newest releases, and an
# install that worked last month stops working (OpenCV 5.0 came without EXR reading and broke every new install).
LOCK_FILE = "requirements-lock.txt"

# The longest path inside the venv is about 150 characters; Windows allows 260 in total.
MAX_SAFE_CODE_DIR_LENGTH = 100

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

def warn_if_path_too_long():
    """Windows refuses paths over 260 characters unless long paths are enabled; pip then fails half way."""
    code_dir = str(get_current_script_dir())
    if os.name == 'nt' and len(code_dir) > MAX_SAFE_CODE_DIR_LENGTH:
        print("\n[WARNING] This folder's path is long:")
        print(f"   {code_dir}")
        print("  If the installation fails with 'No such file or directory' or 'path too long',")
        print("  move the whole folder closer to the drive's root (for example C:\\trellis2) and launch again.\n")

def check_connectivity(timeout: int = 8) -> Tuple[bool, Optional[str]]:
    """Check internet connectivity: any one of the sites the installer downloads from is enough."""
    last_error = None
    for url in ("https://pypi.org", "https://download.pytorch.org/whl/cu128/", "https://github.com"):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True, None
        except urllib.error.HTTPError:
            return True, None   # the site answered, even if not with a page
        except urllib.error.URLError as e:
            reason = getattr(e, 'reason', str(e))
            if isinstance(reason, socket.gaierror):
                last_error = f"DNS resolution failed: {reason}"
            elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
                last_error = "Connection timed out"
            else:
                last_error = f"Connection failed: {reason}"
        except Exception as e:
            last_error = f"Unknown error: {str(e)}"
    return False, last_error

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


def gpu_capability() -> Tuple[int, int]:
    """CUDA compute capability (major, minor) of the card, as the freshly installed PyTorch sees it."""
    # prints "major minor lowest", lowest = the oldest architecture this PyTorch build has code for (61 = sm_61)
    code = ("import torch; "
            "print(*torch.cuda.get_device_capability(), min(int(a[3:]) for a in torch.cuda.get_arch_list() if a[3:].isdigit())) "
            "if torch.cuda.is_available() else print('none')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    answer = result.stdout.split()
    if result.returncode != 0 or not answer:
        raise InstallationError(f"PyTorch does not start:\n{result.stderr.strip()[-600:]}")
    if answer == ["none"]:
        raise InstallationError(
            "PyTorch sees no NVIDIA graphics card. Update the NVIDIA driver to its newest version, then launch again.\n"
            "Nothing needs to be deleted: the installation continues from this point.")
    major, minor, lowest = (int(x) for x in answer)
    if major * 10 + minor < lowest:
        raise InstallationError(
            f"This graphics card is too old for PyTorch {TORCH_PACKAGES.split()[0].split('==')[1]}: "
            f"its CUDA capability is {major}.{minor}, the lowest supported is {lowest // 10}.{lowest % 10} (GTX 10xx).")
    return major, minor


def attention_backends_for(capability: Tuple[int, int]) -> list:
    """The app's own rule for which attention backends a card can run. The file is loaded on its own:
    importing the trellis2 package would need every wheel in place already."""
    import importlib.util
    path = get_current_script_dir() / "trellis2" / "modules" / "attention" / "backend_support.py"
    spec = importlib.util.spec_from_file_location("backend_support", path)
    backend_support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_support)
    return backend_support.backends_for(capability)


def _pip_download_options() -> str:
    """Patience for slow or unstable connections: the PyTorch wheel alone is 3.4 GB."""
    options = "--retries 10 --timeout 60"
    try:
        import pip
        major, minor = (int(x) for x in pip.__version__.split(".")[:2])
        if (major, minor) >= (25, 1):   # older pips restart a broken download from zero
            options += " --resume-retries 20"
    except Exception:
        pass
    return options


def _installed_version(package: str) -> Optional[str]:
    """Version of an installed package, or None. Asks a fresh Python: this process may hold stale metadata."""
    code = f"import importlib.metadata as m; print(m.version('{package}'))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES, fatal: bool = True) -> subprocess.CompletedProcess:
    """Run a command with retry logic."""
    last_error = None
    env = get_git_env()

    if cmd.startswith('pip install'):
        args = cmd[11:]
        cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {_pip_download_options()} {args}'

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


def _hf_file(repo_id: str, filename: str, on_disk_only: bool) -> str:
    """One file of a HuggingFace repo, in the cache the app itself reads (HF_HOME).
    Raises EntryNotFoundError when the file is not on disk (on_disk_only) or not in the repo at all."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
    if on_disk_only:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    for attempt in range(MAX_RETRIES):
        try:
            return hf_hub_download(repo_id, filename)
        except Exception as e:
            # LocalEntryNotFoundError is the hub's "no connection", although it subclasses EntryNotFoundError
            if isinstance(e, EntryNotFoundError) and not isinstance(e, LocalEntryNotFoundError):
                raise
            print(f"  Download failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                raise InstallationError(f"Failed to download {repo_id}/{filename} after {MAX_RETRIES} attempts")
            print(f"  Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)


def _hf_files(candidates: list, on_disk_only: bool) -> Optional[str]:
    """All files of the first candidate (a list of (repo, file) pairs) that exists; None when none does."""
    from huggingface_hub.utils import EntryNotFoundError
    for files in candidates:
        try:
            paths = [_hf_file(repo_id, filename, on_disk_only) for repo_id, filename in files]
            return paths[0]
        except EntryNotFoundError:
            continue
    return None


def _hf_have(what: str, candidates: list) -> str:
    """The files on disk are used as they are, and the hub is asked only for what is missing: an install
    that works neither depends on the connection nor downloads everything again when the repo changes."""
    path = _hf_files(candidates, on_disk_only=True) or _hf_files(candidates, on_disk_only=False)
    if path is None:
        raise InstallationError(f"{what} was not found on HuggingFace")
    return path


def download_hf_models():
    """Pre-download HuggingFace model weights so the app doesn't timeout on first launch."""

    print("\n============================================================================= ")
    print(" Checking the TRELLIS.2 model weights; missing files are downloaded from HuggingFace.")
    print(" This may take 10-30 minutes on first install (~14GB total). Please wait")
    print(  "============================================================================= ")

    # Windows without Developer Mode cannot make symlinks; the hub then prints a long warning that worries people
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    main_repo = "microsoft/TRELLIS.2-4B"

    # Only the checkpoints pipeline.json names, the same files the app asks for when it starts.
    # (The repos also hold encoders and older models the app never loads: 4 GB more.)
    with open(_hf_have("pipeline.json", [[(main_repo, "pipeline.json")]]), "r") as f:
        model_paths = json.load(f)["args"]["models"]

    for key, model_path in model_paths.items():
        print(f"  {key}  ({model_path})")
        # As trellis2/pipelines/base.py: a path inside the main repo first, else "<org>/<repo>/<path>"
        candidates = []
        for full_path in (f"{main_repo}/{model_path}", model_path):
            parts = full_path.split("/")
            repo_id, name = "/".join(parts[:2]), "/".join(parts[2:])
            candidates.append([(repo_id, f"{name}.json"), (repo_id, f"{name}.safetensors")])
        _hf_have(f"Model '{key}' ({model_path})", candidates)
    print("  TRELLIS.2 model weights ready.")


def ensure_single_opencv():
    """cv2 must come from opencv-python-headless alone. The other OpenCV packages install into the same
    cv2 folder and overwrite it (people add them by hand while chasing the EXR error of OpenCV 5)."""
    others = ["opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"]
    present = [p for p in others if _installed_version(p) is not None]
    if not present:
        return
    print(f"Removing extra OpenCV packages: {', '.join(present)}")
    subprocess.run(f'"{sys.executable}" -m pip uninstall -y {" ".join(present)} opencv-python-headless',
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def install_dependencies():
    """Install Trellis 2 dependencies."""
    CODE_DIR = get_current_script_dir()
    check_python_version()
    warn_if_path_too_long()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)

        lock_file = CODE_DIR / LOCK_FILE
        if not lock_file.exists():
            raise InstallationError(f"'{LOCK_FILE}' is missing from {CODE_DIR}")

        # 0. Download model weights (dinov3, RMBG-2.0) if not already present
        download_models()

        # 1. PyTorch 2.8.0 + CUDA 12.8
        #    --no-deps: its dependencies come from the lock file in the next step, at tested versions
        print("\n--- Installing PyTorch 2.8.0 (CUDA 12.8) ---")
        torch_cmd = f"pip install {TORCH_PACKAGES} --no-deps --index-url {TORCH_INDEX}"
        run_command_with_retry(torch_cmd, "Installing PyTorch")

        # 2. General Dependencies, at the exact versions of the lock file
        print("\n--- Installing General Dependencies ---")
        ensure_single_opencv()
        run_command_with_retry(f'pip install -r "{lock_file}"', "Installing pip packages")

        # Stops here, before the 14 GB of models, when the driver is too old for PyTorch to see the card
        attention_backends = attention_backends_for(gpu_capability())

        # 2.1. Install xformers (the attention of RTX 20xx cards, and the fallback of newer ones)
        if "xformers" in attention_backends:
            print("\n--- Installing xformers ---")
            run_command_with_retry(
                f"pip install {XFORMERS_PACKAGE} --no-deps --index-url {TORCH_INDEX}",
                "Installing xformers"
            )

        # 2.5. Pre-download HuggingFace model weights
        download_hf_models()

        # 3. Install Local Wheels
        #    --force-reinstall: after an update a rebuilt wheel keeps its version number, pip would skip it.
        #    --no-deps: their dependencies are in the lock file (and must not be reinstalled along).
        print("\n--- Installing Custom Wheels ---")
        whl_dir = CODE_DIR / "whl"
        for whl_file in sorted(whl_dir.glob("*.whl")):
            # Older releases shipped a Pillow-SIMD wheel that cannot load (it needs a jpeg62.dll nobody has);
            # an updated install may still have the file. Pillow comes from the lock file.
            if whl_file.name.lower().startswith("pillow"):
                continue
            if whl_file.name.lower().startswith("flash_attn") and "flash_attn" not in attention_backends:
                print(f"Skipping {whl_file.name} (this graphics card cannot run Flash Attention)")
                # One left by an earlier, newer card must go: xformers refuses to load beside it
                subprocess.run(f'"{sys.executable}" -m pip uninstall -y flash-attn',
                               shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                continue
            print(f"Installing: {whl_file.name}")
            run_command_with_retry(f'pip install --force-reinstall --no-deps "{whl_file}"', f"Installing {whl_file.name}")

        # Fallback for utils3d if no wheel was present
        if not list(whl_dir.glob("utils3d*.whl")):
            print("Utils3D wheel not found. Installing via Git...")
            run_command_with_retry(
                "pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
                "Installing Utils3D (Git)"
            )

        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


# Runs in a fresh Python, so it sees the packages exactly as the app will.
_VERIFY_SCRIPT = r'''
import os, sys, importlib, warnings
warnings.simplefilter("ignore")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
code_dir = sys.argv[1]
sys.path.insert(0, code_dir)
failed = False

def check(label, fn):
    global failed
    try:
        detail = fn()
        print(f"[OK] {label}" + (f": {detail}" if detail else ""))
    except Exception as e:
        print(f"[ERROR] {label}: {type(e).__name__}: {e}")
        failed = True

import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")

# The app's own modules, so that whatever they import is checked with them; then what the two
# entry points add, and what the downloaded code of the background-removal model imports.
for mod in ("trellis2.pipelines.trellis2_image_to_3d", "trellis2.renderers.pbr_mesh_renderer",
            "trellis2.utils.render_utils", "o_voxel.postprocess",
            "gradio", "fastapi", "uvicorn", "kornia.filters", "timm.models.layers"):
    check(mod, lambda m=mod: importlib.import_module(m) and None)

def read_exr():
    import cv2
    image = cv2.imread(os.path.join(code_dir, "assets", "hdri", "forest.exr"), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV {cv2.__version__} cannot read EXR files")
    return f"OpenCV {cv2.__version__}"
check("EXR environment maps", read_exr)

def attention_backend():
    from trellis2.modules.attention import config
    return config.BACKEND
check("Attention backend", attention_backend)

sys.exit(1 if failed else 0)
'''


def verify_installation():
    """Verify installation: every package the app imports must load, and OpenCV must read the EXR maps."""
    print("\n--- Verifying the installation ---")
    result = subprocess.run([sys.executable, "-c", _VERIFY_SCRIPT, str(get_current_script_dir())])
    if result.returncode != 0:
        print("\n[ERROR] Some packages did not install correctly (see the lines marked ERROR above).")
        print("        Launch again to retry. If it keeps failing, delete the 'code\\venv' folder and launch again.")
        return False
    print("\nVerification successful.")
    return True

if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified!")
        print("You can now run 'app.py'")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)
