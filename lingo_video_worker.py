"""Lingo worker tasks for LiveAvatar orchestration.

This worker intentionally exposes exactly two verbs:
1. Kokoro TTS -> stores generated audio as a local Corpus.
2. LiveAvatar render -> consumes TTS Corpus and stores video as a MinIO Corpus.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import requests

from pydantic import BaseModel, Field

try:
    from lingo import (
        BaseJobPayload,
        Corpus,
        StorageContext,
        get_celery_app,
        grammar,
        phrase,
        validate_channels,
        verb,
    )
except ModuleNotFoundError:
    workspace_lingo = Path(__file__).resolve().parents[3] / "lingo"
    if workspace_lingo.exists():
        sys.path.insert(0, str(workspace_lingo))
    from lingo import (
        BaseJobPayload,
        Corpus,
        StorageContext,
        get_celery_app,
        grammar,
        phrase,
        validate_channels,
        verb,
    )

TASK_TTS = "kokoro_tts"
TASK_RENDER = "liveavatar.video.render"
TASK_RENDER_LEGACY = "multitalk.video.render"


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


class LiveAvatarPayload(BaseJobPayload):
    speech_text: str = Field(min_length=1)
    video_prompt: str = Field(min_length=1)
    kokoro_voice: str = Field(min_length=1)
    avatar_path: str = Field(min_length=1)

    # Used by server orchestration and MinIO object layout.
    job_id: str | None = None
    creator_email: str = "unknown"

    output_dir: str = "./backend_runs"
    output_name: str = "generated.mp4"
    video_storage_base_path: str | None = None

    # Kokoro options
    tts_language_code: str = "a"
    tts_repo_id: str = "weights/Kokoro-82M"
    tts_speed: float = 1.0
    tts_split_pattern: str = r"\n+"
    tts_output_sample_rate: int = Field(default=16000, ge=1)

    # LiveAvatar CLI options (aligned with mini.sh defaults)
    liveavatar_task: str = "s2v-14B"
    liveavatar_size: str = "384*256"
    liveavatar_base_seed: int = 420
    liveavatar_training_config: str = "liveavatar/configs/s2v_causal_sft.yaml"
    liveavatar_ckpt_dir: str = "ckpt/Wan2.2-S2V-14B/"

    liveavatar_ulysses_size: int = Field(default=1, ge=1)
    liveavatar_offload_model: bool = True
    liveavatar_convert_model_dtype: bool = True
    liveavatar_infer_frames: int = Field(default=48, ge=1)
    liveavatar_load_lora: bool = True
    liveavatar_lora_path_dmd: str | None = "Quark-Vision/Live-Avatar"
    liveavatar_sample_steps: int = Field(default=4, ge=1)
    liveavatar_sample_shift: float | None = None
    liveavatar_sample_guide_scale: float = 0.0
    liveavatar_num_clip: int = Field(default=100, ge=1)
    liveavatar_num_gpus_dit: int = Field(default=1, ge=1)
    liveavatar_sample_solver: str = "euler"
    liveavatar_single_gpu: bool = True
    liveavatar_enable_vae_parallel: bool = False
    liveavatar_fp8: bool = True
    liveavatar_offload_kv_cache: bool = True
    liveavatar_t5_cpu: bool = True
    liveavatar_enable_online_decode: bool = False
    liveavatar_start_from_ref: bool = False
    liveavatar_pose_video: str | None = None

    # Runtime launch controls
    liveavatar_nproc_per_node: int = Field(default=1, ge=1)
    liveavatar_master_port: int = Field(default=29101, ge=1)
    liveavatar_cuda_visible_devices: str | None = None
    liveavatar_nccl_debug: str = "WARN"
    liveavatar_nccl_debug_subsys: str = "OFF"
    liveavatar_enable_compile: bool = False
    liveavatar_extra_args: list[str] = Field(default_factory=list)


# Backward-compatible alias for existing integrations.
MultiTalkPayload = LiveAvatarPayload


class KokoroTTSResult(BaseModel):
    pipeline_id: str
    audio_corpus: Corpus[Any]
    local_audio_path: str
    output_bytes: int = Field(ge=0)


class MultiTalkRenderResult(BaseModel):
    pipeline_id: str
    video_corpus: Corpus[Any]
    video_locator: str
    local_output_video_path: str
    output_bytes: int = Field(ge=0)



def _normalize_output_name(name: str) -> str:
    if name.lower().endswith(".mp4"):
        return name
    return f"{name}.mp4"


def _resolve_voice_path(voice: str, repo_dir: Path) -> str:
    normalized = voice.strip()
    if not normalized:
        raise RuntimeError("kokoro_voice is required")

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


def _run_kokoro_tts(payload: LiveAvatarPayload, run_dir: Path) -> Path:
    # Local imports keep worker startup light and only require Kokoro/Torch on TTS nodes.
    from kokoro import KPipeline
    import librosa
    import soundfile as sf
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = (run_dir / "tts.wav").resolve()

    repo_dir = _repo_dir()
    voice_path = _resolve_voice_path(payload.kokoro_voice, repo_dir)
    if not Path(voice_path).exists():
        raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

    pipeline = KPipeline(lang_code=payload.tts_language_code, repo_id=payload.tts_repo_id)
    voice_tensor = torch.load(voice_path, weights_only=True)

    chunks = []
    generator = pipeline(
        payload.speech_text,
        voice=voice_tensor,
        speed=payload.tts_speed,
        split_pattern=payload.tts_split_pattern,
    )
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio samples")

    merged = torch.concat(chunks, dim=0)
    sf.write(str(output_path), merged, 24000)

    # Normalize sample rate for downstream consistency.
    wav_normalized, _ = librosa.load(str(output_path), sr=payload.tts_output_sample_rate)
    sf.write(str(output_path), wav_normalized, payload.tts_output_sample_rate)
    return output_path


def _materialize_avatar(avatar_path: str, work_dir: Path) -> Path:
    parsed = urlparse(avatar_path)
    if parsed.scheme in {"http", "https"}:
        ext = Path(parsed.path).suffix or ".png"
        target = (work_dir / f"avatar{ext}").resolve()
        response = requests.get(avatar_path, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
        return target

    candidate = Path(avatar_path)
    if not candidate.is_absolute():
        candidate = (_repo_dir() / candidate).resolve()
    return candidate


def _run_liveavatar_render(payload: LiveAvatarPayload, audio_file: Path) -> Path:
    repo_dir = _repo_dir()
    pipeline_id = str(payload.pipeline_context.pipeline_id)

    output_dir = (repo_dir / payload.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = _normalize_output_name(payload.output_name)
    output_video_path = output_dir / output_name

    run_dir = output_dir / f"run_{pipeline_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    avatar = _materialize_avatar(payload.avatar_path, run_dir)
    if not avatar.exists():
        raise FileNotFoundError(f"avatar_path not found after materialization: {payload.avatar_path}")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(payload.liveavatar_nproc_per_node),
        "--master_port",
        str(payload.liveavatar_master_port),
        str(repo_dir / "minimal_inference" / "s2v_streaming_interact.py"),
        "--ulysses_size",
        str(payload.liveavatar_ulysses_size),
        "--task",
        payload.liveavatar_task,
        "--size",
        payload.liveavatar_size,
        "--base_seed",
        str(payload.liveavatar_base_seed),
        "--training_config",
        payload.liveavatar_training_config,
        "--offload_model",
        "True" if payload.liveavatar_offload_model else "False",
        "--prompt",
        payload.video_prompt,
        "--image",
        str(avatar.resolve()),
        "--audio",
        str(audio_file.resolve()),
        "--infer_frames",
        str(payload.liveavatar_infer_frames),
        "--sample_steps",
        str(payload.liveavatar_sample_steps),
        "--sample_guide_scale",
        str(payload.liveavatar_sample_guide_scale),
        "--num_clip",
        str(payload.liveavatar_num_clip),
        "--num_gpus_dit",
        str(payload.liveavatar_num_gpus_dit),
        "--sample_solver",
        payload.liveavatar_sample_solver,
        "--ckpt_dir",
        payload.liveavatar_ckpt_dir,
        "--save_file",
        str(output_video_path),
    ]
    if payload.liveavatar_convert_model_dtype:
        command.append("--convert_model_dtype")
    if payload.liveavatar_load_lora:
        command.append("--load_lora")
        if payload.liveavatar_lora_path_dmd:
            command.extend(["--lora_path_dmd", payload.liveavatar_lora_path_dmd])
    if payload.liveavatar_sample_shift is not None:
        command.extend(["--sample_shift", str(payload.liveavatar_sample_shift)])
    if payload.liveavatar_single_gpu:
        command.append("--single_gpu")
    if payload.liveavatar_enable_vae_parallel:
        command.append("--enable_vae_parallel")
    if payload.liveavatar_fp8:
        command.append("--fp8")
    if payload.liveavatar_offload_kv_cache:
        command.append("--offload_kv_cache")
    if payload.liveavatar_t5_cpu:
        command.append("--t5_cpu")
    if payload.liveavatar_enable_online_decode:
        command.append("--enable_online_decode")
    if payload.liveavatar_start_from_ref:
        command.append("--start_from_ref")
    if payload.liveavatar_pose_video:
        command.extend(["--pose_video", payload.liveavatar_pose_video])
    if payload.liveavatar_extra_args:
        command.extend(payload.liveavatar_extra_args)

    env_overrides = {
        "NCCL_DEBUG": payload.liveavatar_nccl_debug,
        "NCCL_DEBUG_SUBSYS": payload.liveavatar_nccl_debug_subsys,
        "ENABLE_COMPILE": "true" if payload.liveavatar_enable_compile else "false",
    }
    if payload.liveavatar_cuda_visible_devices:
        env_overrides["CUDA_VISIBLE_DEVICES"] = payload.liveavatar_cuda_visible_devices

    _run_command(command, cwd=repo_dir, env_overrides=env_overrides)
    return _find_generated_video(output_video_path)


def _safe_creator(creator_email: str) -> str:
    return creator_email.replace("@", "_at_").replace("/", "_")


def _video_storage_context(payload: LiveAvatarPayload) -> StorageContext:
    if payload.video_storage_base_path:
        base_path = payload.video_storage_base_path
    else:
        object_stem = payload.job_id or str(payload.pipeline_context.pipeline_id)
        base_path = f"{_safe_creator(payload.creator_email)}/{object_stem}"
    return StorageContext(strategy="minio", base_path=base_path)


def dispatch_liveavatar_pipeline(
    payload: LiveAvatarPayload,
    *,
    timeout_seconds: int | None = None,
) -> MultiTalkRenderResult:
    """Dispatch the two-step lingo pipeline and wait for completion."""

    tts_step = phrase(TASK_TTS, wait=True)(payload)
    render_step = phrase(TASK_RENDER, wait=True)(tts_step, payload)
    dispatched = render_step.say()
    result = dispatched.get(timeout=timeout_seconds)
    if isinstance(result, MultiTalkRenderResult):
        return result
    return MultiTalkRenderResult.model_validate(result)


def dispatch_multitalk_pipeline(
    payload: LiveAvatarPayload,
    *,
    timeout_seconds: int | None = None,
) -> MultiTalkRenderResult:
    """Backward-compatible alias to LiveAvatar dispatcher."""

    return dispatch_liveavatar_pipeline(payload, timeout_seconds=timeout_seconds)


def minio_object_key_from_locator(locator: str) -> str | None:
    """Extract MinIO object key from an `s3://bucket/key` locator."""

    if not locator.startswith("s3://"):
        return None
    match = re.match(r"^s3://[^/]+/(.+)$", locator)
    if not match:
        return None
    return match.group(1)


# This worker requires Redis broker + MinIO for the video Corpus claim-check.
validate_channels(redis=True, minio=True, mongo=False)
app = get_celery_app()
app.conf.worker_hostname = "liveavatar@%h"


@verb(TASK_TTS, bind=True)
def kokoro_tts(self, payload: LiveAvatarPayload) -> KokoroTTSResult:
    pipeline_id = str(payload.pipeline_context.pipeline_id)
    run_dir = (_repo_dir() / payload.output_dir / f"run_{pipeline_id}" / "tts").resolve()
    audio_path = _run_kokoro_tts(payload, run_dir)

    # No explicit storage_context on purpose: default Corpus backend is local.
    audio_corpus = Corpus.from_file(
        str(audio_path),
        content_type="audio/wav",
        metadata={
            "pipeline_id": pipeline_id,
            "task": TASK_TTS,
            "task_id": self.request.id or "unknown",
        },
    )

    return KokoroTTSResult(
        pipeline_id=pipeline_id,
        audio_corpus=audio_corpus,
        local_audio_path=str(audio_path),
        output_bytes=audio_path.stat().st_size,
    )


@verb(TASK_RENDER_LEGACY, bind=True)
@verb(TASK_RENDER, bind=True)
def render_liveavatar(
    self,
    tts: KokoroTTSResult,
    payload: LiveAvatarPayload,
) -> MultiTalkRenderResult:
    source_audio = Path(tts.audio_corpus.materialize())
    if not source_audio.exists():
        raise FileNotFoundError(f"TTS audio corpus could not be materialized: {source_audio}")

    output_video = _run_liveavatar_render(payload, source_audio)

    # Stage with deterministic filename before MinIO upload; ArtifactManager uses source.name.
    stage_dir = output_video.parent / "publish"
    stage_dir.mkdir(parents=True, exist_ok=True)
    object_name = _normalize_output_name(payload.output_name)
    staged_video = stage_dir / object_name
    if output_video.resolve() != staged_video.resolve():
        shutil.copy2(output_video, staged_video)

    video_corpus = Corpus.from_file(
        str(staged_video),
        storage_context=_video_storage_context(payload),
        content_type="video/mp4",
        metadata={
            "pipeline_id": str(payload.pipeline_context.pipeline_id),
            "task": TASK_RENDER,
            "task_id": self.request.id or "unknown",
            "source_audio_locator": tts.audio_corpus.locator,
        },
    )

    return MultiTalkRenderResult(
        pipeline_id=str(payload.pipeline_context.pipeline_id),
        video_corpus=video_corpus,
        video_locator=video_corpus.locator,
        local_output_video_path=str(output_video),
        output_bytes=output_video.stat().st_size,
    )
