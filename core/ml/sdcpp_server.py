import atexit
import base64
import io
import json
import os
import platform
import socket
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from utils.exceptions import ModelError
from utils.logging import log_message


def pil_to_base64_png(image_pil: Image.Image) -> str:
    image_rgb = image_pil.convert("RGB")
    buffer = io.BytesIO()
    image_rgb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _json_request(
    url: str, payload: Optional[dict] = None, timeout: Optional[int] = None
) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _image_from_result(result: dict) -> Image.Image:
    result = result.get("result") or result
    images = result.get("images") or result.get("data")
    if not images:
        raise ModelError("sd.cpp completed without returning an image.")

    image = images[0]
    if isinstance(image, dict):
        image_b64 = (
            image.get("b64_json")
            or image.get("image")
            or image.get("data")
            or image.get("url")
        )
    else:
        image_b64 = image
    if not isinstance(image_b64, str):
        raise ModelError("sd.cpp returned an image in an unexpected format.")

    if "," in image_b64 and image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            return decoded.convert("RGB").copy()
    except Exception as e:
        raise ModelError(f"Failed to decode sd.cpp output image: {e}") from e


def _log_offset(log_path) -> int:
    if log_path is None:
        return 0
    try:
        return Path(log_path).stat().st_size
    except OSError:
        return 0


def _read_log_lines(log_path, offset: int) -> tuple[list[str], int]:
    if log_path is None:
        return [], offset

    path = Path(log_path)
    try:
        size = path.stat().st_size
        if offset > size:
            offset = 0
        with open(path, "rb") as log_file:
            log_file.seek(offset)
            data = log_file.read(64 * 1024)
            new_offset = log_file.tell()
    except OSError:
        return [], offset

    lines = [
        line.strip()
        for line in data.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines, new_offset


def run_image_job(
    server: dict, payload: dict, verbose: bool = False, timeout_sec: int = 900
) -> Image.Image:
    base_url = server["url"].rstrip("/")
    log_path = server.get("log_path")
    log_suffix = f" Log: {log_path}" if log_path else ""
    log_offset = _log_offset(log_path)
    start = time.monotonic()
    log_message("  - Submitting sd.cpp inference job...", always_print=True)
    job = _json_request(f"{base_url}/sdcpp/v1/img_gen", payload=payload, timeout=30)
    if job.get("status") == "completed" or job.get("result") or job.get("images"):
        log_message(
            f"  - sd.cpp inference completed in {time.monotonic() - start:.1f}s.",
            always_print=True,
        )
        return _image_from_result(job)

    poll_url = job.get("poll_url")
    if poll_url is None and job.get("id"):
        poll_url = f"/sdcpp/v1/jobs/{job['id']}"
    if not poll_url:
        raise ModelError(f"sd.cpp did not return a job id.{log_suffix}")
    if poll_url.startswith("/"):
        poll_url = f"{base_url}{poll_url}"

    log_message("  - Waiting for sd.cpp server job...", always_print=True)
    deadline = time.monotonic() + timeout_sec
    next_log = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = _json_request(poll_url, timeout=30)
        state = status.get("status")
        if state == "completed":
            log_message(
                f"  - sd.cpp inference completed in {time.monotonic() - start:.1f}s.",
                always_print=True,
            )
            return _image_from_result(status)
        if state in ("failed", "cancelled"):
            error = status.get("error") or {}
            message = error.get("message") or state
            raise ModelError(f"sd.cpp job {state}: {message}.{log_suffix}")
        now = time.monotonic()
        if now >= next_log:
            lines, log_offset = _read_log_lines(log_path, log_offset)
            if lines:
                for line in lines[-3:]:
                    log_message(f"  - sd.cpp: {line}", always_print=True)
            else:
                log_message(
                    f"  - Waiting for sd.cpp server output "
                    f"({now - start:.0f}s elapsed, status={state or 'pending'})...",
                    always_print=True,
                )
            next_log = now + 10
        time.sleep(0.5)

    raise ModelError(f"Timed out waiting for sd.cpp image generation.{log_suffix}")


class SDCppServerManager:
    RELEASE_TAG = "master-778-c00a9e9"
    RELEASE_COMMIT = "c00a9e9"
    RELEASE_BASE_URL = (
        "https://github.com/leejet/stable-diffusion.cpp/releases/download"
    )
    SERVER_LOG_GLOB = "*-server.log"
    SERVER_LOG_RETENTION_COUNT = 3
    BINARY_ENV = "MANGATRANSLATOR_SDCPP_BINARY"
    SERVER_BINARY_ENV = "MANGATRANSLATOR_SDCPP_SERVER_BINARY"
    RELEASE_ASSET_ENV = "MANGATRANSLATOR_SDCPP_RELEASE_ASSET"
    # Prefer the newer matching ROCm build
    WIN_ROCM_ASSET = "bin-win-rocm-7.13.0-x64.zip"
    LINUX_ROCM_ASSET = "bin-Linux-Ubuntu-24.04-x86_64-rocm-7.13.0.zip"
    DARWIN_ARM64_ASSET = "bin-Darwin-macOS-26.4-arm64.zip"

    def __init__(self, install_dir: Path):
        self.install_dir = Path(install_dir)
        self._servers = {}
        self._lock = threading.Lock()
        atexit.register(self.shutdown_servers)

    def _download_file(self, path: Path, url: str, verbose: bool = False) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        log_message(f"Downloading {path.name}...", always_print=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(path, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                last_log = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_log >= 5:
                        if total:
                            percent = min(100.0, downloaded * 100.0 / total)
                            log_message(
                                f"  - {path.name}: {downloaded / (1024**2):.1f}/"
                                f"{total / (1024**2):.1f} MiB ({percent:.1f}%)",
                                always_print=True,
                            )
                        else:
                            log_message(
                                f"  - {path.name}: {downloaded / (1024**2):.1f} MiB",
                                always_print=True,
                            )
                        last_log = now
            log_message(f"Downloaded {path.name} successfully.", always_print=True)
        except Exception as e:
            if path.exists():
                path.unlink()
            raise ModelError(f"Failed to download {path.name}: {e}") from e

    def _remove_archive(self, archive_path: Path, verbose: bool = False) -> None:
        try:
            archive_path.unlink()
            log_message(
                f"Removed extracted sd.cpp archive {archive_path.name}.",
                verbose=verbose,
            )
        except FileNotFoundError:
            pass
        except OSError as e:
            log_message(
                f"Warning: Could not remove sd.cpp archive {archive_path}: {e}",
                always_print=True,
            )

    def _cleanup_stale_logs(
        self, current_log_path: Optional[Path] = None, verbose: bool = False
    ) -> None:
        if not self.install_dir.exists():
            return

        logs = []
        for log_path in self.install_dir.glob(self.SERVER_LOG_GLOB):
            try:
                if log_path.is_file():
                    logs.append((log_path.stat().st_mtime, log_path))
            except OSError as e:
                log_message(
                    f"Warning: Could not inspect sd.cpp log {log_path}: {e}",
                    verbose=verbose,
                )

        logs.sort(key=lambda item: item[0], reverse=True)
        retained = {current_log_path} if current_log_path is not None else set()
        removed = 0
        for _, log_path in logs:
            if log_path in retained:
                continue
            if len(retained) < self.SERVER_LOG_RETENTION_COUNT:
                retained.add(log_path)
                continue
            try:
                log_path.unlink()
                removed += 1
            except OSError as e:
                log_message(
                    f"Warning: Could not remove stale sd.cpp log {log_path}: {e}",
                    verbose=verbose,
                )
        if removed:
            log_message(
                f"Removed {removed} stale sd.cpp server log file(s).",
                verbose=verbose,
            )

    def _release_asset(self, suffix: str) -> str:
        return f"sd-master-{self.RELEASE_COMMIT}-{suffix}"

    @staticmethod
    def _torch_hip_available() -> bool:
        hip = getattr(torch.version, "hip", None)
        return bool(hip) and torch.cuda.is_available()

    def _select_archive_name(self) -> str:
        override = os.environ.get(self.RELEASE_ASSET_ENV)
        if override:
            return override

        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "windows":
            if self._torch_hip_available():
                return self._release_asset(self.WIN_ROCM_ASSET)
            if torch.cuda.is_available():
                return self._release_asset("bin-win-cuda12-x64.zip")
            # Dynamic CPU backends replace the old avx/avx2/avx512 matrix.
            return self._release_asset("bin-win-cpu-x64.zip")

        if system == "darwin" and machine in ("arm64", "aarch64"):
            return self._release_asset(self.DARWIN_ARM64_ASSET)

        if system == "linux" and machine in ("x86_64", "amd64"):
            if self._torch_hip_available():
                return self._release_asset(self.LINUX_ROCM_ASSET)
            return self._release_asset("bin-Linux-Ubuntu-24.04-x86_64.zip")

        raise ModelError(
            "No default sd.cpp binary is configured for this platform. "
            f"Set {self.SERVER_BINARY_ENV} to an sd-server executable path, "
            f"or {self.RELEASE_ASSET_ENV} to a release asset name "
            f"(e.g. win-vulkan / Linux vulkan builds)."
        )

    def _archive_url(self, archive_name: str) -> str:
        if archive_name.startswith(("http://", "https://")):
            return archive_name
        return f"{self.RELEASE_BASE_URL}/{self.RELEASE_TAG}/{archive_name}"

    def _find_server_executable(self, extract_dir: Path) -> Optional[Path]:
        names = (
            ("sd-server.exe", "sd-server")
            if platform.system().lower() == "windows"
            else ("sd-server",)
        )
        for name in names:
            matches = sorted(path for path in extract_dir.rglob(name) if path.is_file())
            if matches:
                return matches[0]
        return None

    def _runtime_archive_names(self, archive_name: str) -> tuple[str, ...]:
        archive_file = archive_name.rsplit("/", 1)[-1]
        if platform.system().lower() == "windows" and "win-cuda12" in archive_file:
            return ("cudart-sd-bin-win-cu12-x64.zip",)
        return ()

    def _ensure_runtime_archives(
        self, archive_name: str, executable: Path, verbose: bool = False
    ) -> None:
        for runtime_archive in self._runtime_archive_names(archive_name):
            archive_path = self.install_dir / runtime_archive
            sentinel = executable.parent / f".{Path(runtime_archive).stem}.extracted"
            if sentinel.exists():
                self._remove_archive(archive_path, verbose=verbose)
                continue

            self._download_file(
                archive_path,
                self._archive_url(runtime_archive),
                verbose=verbose,
            )

            log_message(f"Extracting {runtime_archive}...", verbose=verbose)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(executable.parent)
                sentinel.write_text("ok", encoding="utf-8")
                self._remove_archive(archive_path, verbose=verbose)
            except Exception as e:
                raise ModelError(f"Failed to extract {runtime_archive}: {e}") from e

    def ensure_server_executable(self, verbose: bool = False) -> Path:
        for env_name in (self.SERVER_BINARY_ENV, self.BINARY_ENV):
            override = os.environ.get(env_name)
            if not override:
                continue
            path = Path(os.path.expandvars(override)).expanduser()
            if not path.exists():
                raise ModelError(f"{env_name} points to a missing file: {path}")
            return path

        archive_name = self._select_archive_name()
        archive_file = archive_name.rsplit("/", 1)[-1]
        archive_path = self.install_dir / archive_file
        extract_dir = self.install_dir / Path(archive_file).stem

        executable = (
            self._find_server_executable(extract_dir) if extract_dir.exists() else None
        )
        if executable is not None:
            self._ensure_runtime_archives(archive_name, executable, verbose=verbose)
            self._remove_archive(archive_path, verbose=verbose)
            return executable

        self.install_dir.mkdir(parents=True, exist_ok=True)
        self._download_file(
            archive_path, self._archive_url(archive_name), verbose=verbose
        )

        log_message(f"Extracting {archive_file}...", verbose=verbose)
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_dir)
        except Exception as e:
            raise ModelError(f"Failed to extract {archive_file}: {e}") from e

        executable = self._find_server_executable(extract_dir)
        if executable is None:
            raise ModelError(f"sd-server executable was not found in {archive_file}.")

        self._ensure_runtime_archives(archive_name, executable, verbose=verbose)

        if platform.system().lower() != "windows":
            try:
                executable.chmod(executable.stat().st_mode | 0o111)
            except OSError:
                pass

        self._remove_archive(archive_path, verbose=verbose)
        log_message("sd.cpp server is ready.", verbose=verbose)
        return executable

    def _cache_args(
        self, cache_mode: str, num_inference_steps: int
    ) -> tuple[list[str], str, int, int]:
        mode = (cache_mode or "none").lower()
        steps = max(1, int(num_inference_steps))
        warmup = max(1, (steps + 3) // 4)
        if mode == "none":
            return ([], mode, 0, steps)
        if mode == "spectrum":
            return (
                [
                    "--cache-mode",
                    "spectrum",
                    "--cache-option",
                    f"warmup={warmup},window=2,stop=0.8",
                ],
                mode,
                warmup,
                steps,
            )
        if mode == "cache-dit":
            return (
                [
                    "--cache-mode",
                    "cache-dit",
                    "--cache-option",
                    f"Fn=4,Bn=0,threshold=0.10,warmup={warmup}",
                    "--scm-policy",
                    "dynamic",
                ],
                mode,
                warmup,
                steps,
            )
        if mode == "taylorseer":
            return (
                [
                    "--cache-mode",
                    "taylorseer",
                    "--cache-option",
                    f"Fn=4,Bn=0,warmup={warmup}",
                ],
                mode,
                warmup,
                steps,
            )
        if mode == "dbcache":
            return (
                [
                    "--cache-mode",
                    "dbcache",
                    "--cache-option",
                    f"Fn=8,Bn=0,threshold=0.08,warmup={warmup}",
                ],
                mode,
                warmup,
                steps,
            )

        raise ModelError(f"Unknown sd.cpp cache mode: {cache_mode}")

    def _server_key(
        self,
        model_key: str,
        cache_mode: str,
        num_inference_steps: int,
    ) -> tuple[str, str, int]:
        _, normalized_cache_mode, _, steps = self._cache_args(
            cache_mode, num_inference_steps
        )
        return (model_key, normalized_cache_mode, steps)

    def _safe_server_key(self, server_key: tuple[str, str, int]) -> str:
        model_key, cache_mode, steps = server_key
        return f"{model_key}-offload-{cache_mode}-steps{steps}"

    def _free_tcp_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _server_ready(self, base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2):
                return True
        except Exception:
            return False

    def _log_tail(self, log_path: Optional[Path], limit: int = 2000) -> str:
        if log_path is None or not log_path.exists():
            return ""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - limit))
                return f.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _close_windows_handle(self, handle) -> None:
        if not handle:
            return
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _attach_to_windows_kill_job(
        self, process: subprocess.Popen, verbose: bool = False
    ):
        if platform.system().lower() != "windows":
            return None

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [
                wintypes.HANDLE,
                wintypes.HANDLE,
            ]
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                raise ctypes.WinError(ctypes.get_last_error())

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                error = ctypes.WinError(ctypes.get_last_error())
                self._close_windows_handle(job)
                raise error

            if not kernel32.AssignProcessToJobObject(job, process._handle):
                error = ctypes.WinError(ctypes.get_last_error())
                self._close_windows_handle(job)
                raise error

            return job
        except Exception as e:
            log_message(
                f"Warning: Could not attach sd.cpp server to Windows shutdown job: {e}",
                always_print=True,
            )
            return None

    def _wait_for_server(
        self,
        base_url: str,
        process: subprocess.Popen,
        log_path: Optional[Path],
        timeout_sec: int = 900,
        label: str = "sd.cpp",
        verbose: bool = False,
    ) -> None:
        start = time.monotonic()
        next_log = start + 10
        log_message(
            f"Waiting for {label} sd.cpp server to load models...",
            always_print=True,
        )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = self._log_tail(log_path)
                raise ModelError(
                    f"sd.cpp server exited before becoming ready. {detail}"
                )
            if self._server_ready(base_url):
                elapsed = time.monotonic() - start
                log_message(
                    f"{label} sd.cpp server loaded in {elapsed:.1f}s.",
                    always_print=True,
                )
                return
            now = time.monotonic()
            if now >= next_log:
                log_message(
                    f"Still loading {label} sd.cpp server "
                    f"({now - start:.0f}s elapsed). Log: {log_path}",
                    always_print=True,
                )
                next_log = now + 10
            time.sleep(0.5)

        detail = self._log_tail(log_path)
        raise ModelError(f"Timed out waiting for sd.cpp server. {detail}")

    def ensure_flux_server(
        self,
        model_key: str,
        assets: dict,
        cache_mode: str = "none",
        num_inference_steps: int = 4,
        verbose: bool = False,
    ) -> dict:
        server_key_model = assets.get("server_model_key", model_key)
        server_key = self._server_key(server_key_model, cache_mode, num_inference_steps)
        with self._lock:
            server = self._servers.get(server_key)
            if server is not None:
                process = server.get("process")
                base_url = server.get("url")
                if (
                    process is not None
                    and process.poll() is None
                    and base_url
                    and self._server_ready(base_url)
                ):
                    return server
                self._servers.pop(server_key, None)
                self._stop_server_record(server, verbose=verbose)

            port = self._free_tcp_port()
            base_url = f"http://127.0.0.1:{port}"
            self.install_dir.mkdir(parents=True, exist_ok=True)
            log_path = (
                self.install_dir / f"{self._safe_server_key(server_key)}-server.log"
            )
            self._cleanup_stale_logs(log_path, verbose=verbose)
            log_file = open(log_path, "ab")
            cache_args, normalized_cache_mode, warmup, steps = self._cache_args(
                cache_mode, num_inference_steps
            )

            # Scheduler / ref-image mode left to sd.cpp model auto-detect
            cmd = [
                str(assets["executable"]),
                "--listen-ip",
                "127.0.0.1",
                "--listen-port",
                str(port),
                "--diffusion-model",
                str(assets["diffusion_model"]),
                "--vae",
                str(assets["vae"]),
                "--fa",
                "--eager-load",
                "--cfg-scale",
                "1.0",
                "--img-cfg-scale",
                "1.0",
                "--guidance",
                "2.5" if model_key == "flux_kontext" else "1.0",
                "--sampling-method",
                "euler",
                "--steps",
                str(steps),
                "--offload-to-cpu",
            ]
            cmd.extend(cache_args)
            for flag, key in zip(
                assets["server_args"][::2], assets["server_args"][1::2]
            ):
                cmd.extend([flag, str(assets[key])])
            if verbose:
                cmd.append("-v")

            creationflags = (
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            log_message(
                f"Starting persistent sd.cpp server for {assets['label']}...",
                always_print=True,
            )
            process = None
            job_handle = None
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(Path(assets["executable"]).parent),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
                job_handle = self._attach_to_windows_kill_job(process, verbose=verbose)
                self._wait_for_server(
                    base_url,
                    process,
                    log_path,
                    label=assets["label"],
                    verbose=verbose,
                )
            except Exception:
                if process is not None:
                    self._stop_server_record(
                        {
                            "process": process,
                            "log_file": log_file,
                            "model_key": model_key,
                            "server_key": server_key,
                            "job_handle": job_handle,
                        },
                        verbose=verbose,
                    )
                else:
                    log_file.close()
                raise

            server = {
                "process": process,
                "url": base_url,
                "assets": assets,
                "log_file": log_file,
                "log_path": log_path,
                "model_key": model_key,
                "server_key": server_key,
                "cache_mode": normalized_cache_mode,
                "cache_warmup": warmup,
                "steps": steps,
                "job_handle": job_handle,
            }
            self._servers[server_key] = server
            log_message(
                f"sd.cpp server for {assets['label']} is ready at {base_url}.",
                always_print=True,
            )
            return server

    def _stop_server_record(
        self, server: Optional[dict], verbose: bool = False
    ) -> None:
        if not server:
            return
        process = server.get("process")
        if process is not None and process.poll() is None:
            log_message(
                f"Stopping sd.cpp server {server.get('model_key', '')}...",
                verbose=verbose,
            )
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_file = server.get("log_file")
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
        self._close_windows_handle(server.get("job_handle"))

    def shutdown_server(self, model_key: str, verbose: bool = False) -> None:
        with self._lock:
            matching_keys = [
                key
                for key, server in self._servers.items()
                if server.get("model_key") == model_key
            ]
            servers = [self._servers.pop(key) for key in matching_keys]
        for server in servers:
            self._stop_server_record(server, verbose=verbose)

    def shutdown_servers(self, verbose: bool = False) -> None:
        with self._lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for server in servers:
            self._stop_server_record(server, verbose=verbose)
