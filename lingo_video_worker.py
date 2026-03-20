"""Lingo worker tasks for LiveAvatar orchestration.

This worker intentionally exposes exactly two verbs:
1. Kokoro TTS -> stores generated audio as a local Corpus.
2. LiveAvatar render -> consumes TTS Corpus and stores video as a MinIO Corpus.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

from pydantic import BaseModel, Field

try:
    from lingo import (
        Corpus,
        Reference,
        get_celery_app,
        validate_channels,
        verb,
        scribble,
    )
except ModuleNotFoundError:
    workspace_lingo = Path(__file__).resolve().parents[3] / "lingo"
    if workspace_lingo.exists():
        sys.path.insert(0, str(workspace_lingo))
    from lingo import (
        Corpus,
        Reference,
        get_celery_app,
        validate_channels,
        verb,
        scribble,
    )


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


class KokoroSettings(BaseModel):
    language_code: str = "a"
    repo_id: str = "weights/Kokoro-82M"
    speed: float = 1.0
    split_pattern: str = r"\n+"
    output_sample_rate: int = Field(default=16000, ge=1)


class LiveAvatarSettings(BaseModel):
    video_prompt: str = Field('A professional speaks confidently directly to the camera', min_length=1)

    video_storage_base_path: str | None = None

    task: str = "s2v-14B"
    size: str = "384*256"
    base_seed: int = 420
    training_config: str = "liveavatar/configs/s2v_causal_sft.yaml"
    ckpt_dir: str = "ckpt/Wan2.2-S2V-14B/"

    ulysses_size: int = Field(default=1, ge=1)
    offload_model: bool = True
    convert_model_dtype: bool = True
    infer_frames: int = Field(default=48, ge=1)
    load_lora: bool = True
    lora_path_dmd: str | None = "Quark-Vision/Live-Avatar"
    sample_steps: int = Field(default=4, ge=1)
    sample_shift: float | None = None
    sample_guide_scale: float = 0.0
    num_clip: int = Field(default=100, ge=1)
    num_gpus_dit: int = Field(default=1, ge=1)
    sample_solver: str = "euler"
    single_gpu: bool = True
    enable_vae_parallel: bool = False
    fp8: bool = True
    offload_kv_cache: bool = True
    t5_cpu: bool = True
    enable_online_decode: bool = False
    start_from_ref: bool = False
    pose_video: str | None = None

    # Runtime launch controls
    nproc_per_node: int = Field(default=1, ge=1)
    master_port: int = Field(default=29101, ge=1)
    cuda_visible_devices: str | None = None
    nccl_debug: str = "WARN"
    nccl_debug_subsys: str = "OFF"
    enable_compile: bool = False
    extra_args: list[str] = Field(default_factory=list)


def _resolve_voice_path(voice: str, repo_dir: Path) -> str:
    normalized = voice.strip()
    if not normalized:
        raise RuntimeError("voice_id is required")

    if "/" in normalized or "\\" in normalized or normalized.endswith(".pt"):
        resolved = Path(normalized)
        if not resolved.is_absolute():
            resolved = (repo_dir / resolved).resolve()
        return str(resolved)

    candidate = (repo_dir / "weights" / "Kokoro-82M" / "voices" / f"{normalized}.pt").resolve()
    return str(candidate)


def _find_generated_video(requested_output_path: Path) -> Path:
    if requested_output_path.exists():
        return requested_output_path

    parent = requested_output_path.parent
    stem = requested_output_path.stem
    candidates = sorted(
        parent.glob(f"{stem}*.mp4"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise RuntimeError(f"No output video found near: {requested_output_path}")


def _run_command(command: list[str], cwd: Path, env_overrides: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    process = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(command)}\n"
            f"stdout:\n{process.stdout[-5000:]}\n"
            f"stderr:\n{process.stderr[-5000:]}"
        )


# This worker requires Redis broker + MinIO for the video Corpus claim-check.
validate_channels(redis=True, minio=True, mongo=False)
app = get_celery_app()


@verb('kokoro_tts')
def kokoro_tts(speech_text: str, voice_id: str, param: Optional[KokoroSettings] = None) -> Corpus:
    if param is None:
        param = KokoroSettings()

    run_dir = scribble()
    
    # Local imports keep worker startup light and only require Kokoro/Torch on TTS nodes.
    from kokoro import KPipeline
    import librosa
    import soundfile as sf
    import torch

    output_path = (run_dir / "tts.wav").resolve()

    repo_dir = _repo_dir()
    voice_path = _resolve_voice_path(voice_id, repo_dir)
    if not Path(voice_path).exists():
        raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

    pipeline = KPipeline(lang_code=param.language_code, repo_id=param.repo_id)
    voice_tensor = torch.load(voice_path, weights_only=True)

    chunks = []
    generator = pipeline(
        speech_text,
        voice=voice_tensor,
        speed=param.speed,
        split_pattern=param.split_pattern,
    )
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio samples")

    merged = torch.concat(chunks, dim=0)
    sf.write(str(output_path), merged, 24000)

    # Normalize sample rate for downstream consistency.
    wav_normalized, _ = librosa.load(str(output_path), sr=param.output_sample_rate)
    sf.write(str(output_path), wav_normalized, param.output_sample_rate)

    return Corpus.from_file(str(output_path), content_type="audio/wav")


@verb('liveavatar.video.render')
def render_liveavatar(audio: Corpus, avatar: Corpus, video_prompt: str = None, dest: Optional[Reference] = None,
                      param: Optional[LiveAvatarSettings] = None) -> Corpus:
    if param is None:
        param = LiveAvatarSettings()
    if video_prompt is None:
        video_prompt = param.video_prompt

    audio_path = Path(audio.materialize_path())
    if not audio_path.exists():
        raise FileNotFoundError(f"TTS audio could not be materialized: {audio_path}")

    avatar_path = Path(avatar.materialize_path())
    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar could not be materialized: {avatar_path}")

    repo_dir = _repo_dir()
    run_dir = scribble()
    output_path = run_dir.joinpath("output.mp4").resolve()

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(param.nproc_per_node),
        "--master_port",
        str(param.master_port),
        str((repo_dir / "minimal_inference" / "s2v_streaming_interact.py").resolve()),
        "--ulysses_size",
        str(param.ulysses_size),
        "--task",
        param.task,
        "--size",
        param.size,
        "--base_seed",
        str(param.base_seed),
        "--training_config",
        param.training_config,
        "--offload_model",
        "True" if param.offload_model else "False",
        "--prompt",
        video_prompt,
        "--image",
        str(avatar_path.resolve()),
        "--audio",
        str(audio_path.resolve()),
        "--infer_frames",
        str(param.infer_frames),
        "--sample_steps",
        str(param.sample_steps),
        "--sample_guide_scale",
        str(param.sample_guide_scale),
        "--num_clip",
        str(param.num_clip),
        "--num_gpus_dit",
        str(param.num_gpus_dit),
        "--sample_solver",
        param.sample_solver,
        "--ckpt_dir",
        param.ckpt_dir,
        "--save_file",
        str(output_path.resolve()),
    ]
    if param.convert_model_dtype:
        command.append("--convert_model_dtype")
    if param.load_lora:
        command.append("--load_lora")
        if param.lora_path_dmd:
            command.extend(["--lora_path_dmd", param.lora_path_dmd])
    if param.sample_shift is not None:
        command.extend(["--sample_shift", str(param.sample_shift)])
    if param.single_gpu:
        command.append("--single_gpu")
    if param.enable_vae_parallel:
        command.append("--enable_vae_parallel")
    if param.fp8:
        command.append("--fp8")
    if param.offload_kv_cache:
        command.append("--offload_kv_cache")
    if param.t5_cpu:
        command.append("--t5_cpu")
    if param.enable_online_decode:
        command.append("--enable_online_decode")
    if param.start_from_ref:
        command.append("--start_from_ref")
    if param.pose_video:
        command.extend(["--pose_video", param.pose_video])
    if param.extra_args:
        command.extend(param.extra_args)

    env_overrides = {
        "NCCL_DEBUG": param.nccl_debug,
        "NCCL_DEBUG_SUBSYS": param.nccl_debug_subsys,
        "ENABLE_COMPILE": "true" if param.enable_compile else "false",
    }
    if param.cuda_visible_devices:
        env_overrides["CUDA_VISIBLE_DEVICES"] = param.cuda_visible_devices

    _run_command(command, cwd=repo_dir, env_overrides=env_overrides)
    
    output_video = _find_generated_video(run_dir)

    if dest is None:
        video_corpus = Corpus.from_file(str(output_video), content_type="video/mp4")
    else:
        video_corpus = dest.dump(output_video)
    
    return video_corpus
