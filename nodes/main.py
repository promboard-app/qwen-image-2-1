"""promboard.qwen-image-2-1 — Qwen-Image 2.1(Qwen) 글 → 이미지 · 이미지 편집 · 투명 배경 노드 코드.

노드 하나(generate): 프롬프트 글 + 참조 이미지(선택, 4장까지)를 받아 이미지 한 장(PNG, 투명 배경이면 RGBA)을 낸다.
  참조 이미지가 없으면 글 → 이미지, 있으면 그 이미지를 고치거나 여러 장을 섞는다(같은 모델이 둘 다 한다 — 공식 카드).
  '투명 배경'을 켜면 공식 카드가 권하는 투명 이미지 프롬프트 틀로 감싸 RGBA 이미지를 만든다:
    "This is an RGBA image with transparency. <프롬프트> The image has alpha channel and the background is transparent."
경량화 설정(quantization) — 판과 정밀도를 한 칸으로 고른다(manifest requires.models 의 when 이 고른 것만 받는다. '자동'은
  서버가 이 PC 그래픽카드 메모리로 turbo-bf16 · turbo-q8 · turbo-q4 중 하나로 풀어 넘긴다).
  판: turbo = Qwen-Image-2.1-Turbo — 저장된 8단계 일정(sample_sigmas) 그대로, CFG 없음
      base  = Qwen-Image-2.1 — 공식 예시 40단계
  정밀도: bf16 = 공식 트랜스포머(14.2GB) · q8 / q4 = unsloth 가 GGUF Q8_0(7.6GB) · Q4_K_M(4.2GB) 으로 줄인 판
    (unsloth/Qwen-Image-2.1-Turbo-GGUF · unsloth/Qwen-Image-2.1-GGUF)
  판과 정밀도를 두 칸으로 나누지 않은 까닭: requires.models 의 when 은 설정 하나만 보므로, 고른 트랜스포머 파일만 받으려면 한 칸이어야 한다.
  글 · 그림 인코더
  (Qwen3-VL-8B, bf16 17.5GB) · 프로세서 · VAE(RGBA 4채널)는 공식 Qwen/Qwen-Image-2.1-Turbo 의 것을 두 판이 같이 쓴다.
  2.1(Base)은 스케줄러 설정만 공식 Qwen/Qwen-Image-2.1 에서 받고, Turbo 의 sample_sigmas 를 지운다(두 판의 diffusers 설정
  차이는 이 둘과 버전 표기뿐이다).
크기: 공식 해상도 표는 2K(2048² 등)지만 파이프라인 기본(output_resolution)은 1024 다. 이 노드는 100만 픽셀 판을 기본으로 두고
  2048² 를 고를 수 있게 했다. 크기 '자동'이면 마지막 참조 이미지의 가로세로 비로 100만 픽셀, 참조 이미지가 없으면 1024².
🔴 라이선스: Qwen Research License — 연구 · 평가 목적만(비상업). manifest nonCommercial · licenseNotice.

실행: diffusers 0.42.0 QwenImage21Pipeline + QwenImage21Transformer2DModel(bf16 은 from_pretrained, GGUF 는 from_single_file +
  GGUFQuantizationConfig bf16 계산).
그래픽카드 메모리에 맞게 부품을 올리는 법은 gpu.py(글 인코더 17.5GB 는 12GB 카드에서 조각씩 올리기로 돈다).
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from . import gpu  # noqa: E402

TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
BASE_REPO = "Qwen/Qwen-Image-2.1"
MODELS = {
    # 설정 값: (판, GGUF 모델 이름 = <저장소>/<파일>, bf16 이면 None — 그 판 공식 저장소의 transformer/ 를 읽는다)
    "turbo-bf16": ("turbo", None),
    "turbo-q8": ("turbo", "unsloth/Qwen-Image-2.1-Turbo-GGUF/qwen-image-2.1-turbo-Q8_0.gguf"),
    "turbo-q4": ("turbo", "unsloth/Qwen-Image-2.1-Turbo-GGUF/qwen-image-2.1-turbo-Q4_K_M.gguf"),
    "base-bf16": ("base", None),
    "base-q8": ("base", "unsloth/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q8_0.gguf"),
    "base-q4": ("base", "unsloth/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf"),
}
BASE_STEPS = 40  # 공식 카드 예시(2.1). Turbo 는 저장된 8단계 일정을 쓴다
TURBO_STEPS = 8
MAX_REFERENCES = 4
TRANSPARENT_PREFIX = "This is an RGBA image with transparency. "
TRANSPARENT_SUFFIX = " The image has alpha channel and the background is transparent."
# 부품이 계산할 때 더 쓰는 VRAM(1024² 기준, 넓이 · 참조 이미지 수에 비례해 늘린다 — reserve_for)
RESERVE_GIB = {"text_encoder": 1.0, "transformer": 1.5, "vae": 2.5}
REF_RESERVE_GIB = 0.5
SEED_MAX = 2 ** 31 - 1

_STATE = {"key": None, "pipe": None}


def reserve_for(width, height, refs):
    scale = max(1.0, (width * height) / (1024 * 1024))
    out = {name: int(gib * gpu.GIB * (1.0 if name == "text_encoder" else scale)) for name, gib in RESERVE_GIB.items()}
    out["transformer"] += int(REF_RESERVE_GIB * gpu.GIB * refs)
    out["text_encoder"] += int(REF_RESERVE_GIB * gpu.GIB * refs)
    return out


def repo_dir(ctx, repo):
    """받은 저장소 폴더(<모델 폴더>/_plugin-models/<저장소>@<커밋>) — 스케줄러 설정 파일 자리에서 두 단계 위."""
    return os.path.dirname(os.path.dirname(ctx.model_path(f"{repo}/scheduler/scheduler_config.json")))


def load_pipeline(ctx, model_key):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, QwenImage21Pipeline, QwenImage21Transformer2DModel

    variant, gguf_name = MODELS[model_key]
    turbo_dir = repo_dir(ctx, TURBO_REPO)
    dtype = torch.bfloat16
    ctx.progress(0.03, "load-model")
    if gguf_name:
        transformer = QwenImage21Transformer2DModel.from_single_file(
            ctx.model_path(gguf_name),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            dtype=dtype,
            config=turbo_dir,
            subfolder="transformer",
            local_files_only=True,
        )
    else:
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            turbo_dir if variant == "turbo" else repo_dir(ctx, BASE_REPO),
            subfolder="transformer",
            dtype=dtype,
            local_files_only=True,
        )
    ctx.progress(0.10, "load-model")
    extra = {}
    if variant == "base":
        extra["scheduler"] = FlowMatchEulerDiscreteScheduler.from_pretrained(repo_dir(ctx, BASE_REPO), subfolder="scheduler", local_files_only=True)
        extra["sample_sigmas"] = None
    pipe = QwenImage21Pipeline.from_pretrained(turbo_dir, transformer=transformer, dtype=dtype, local_files_only=True, **extra)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_pipeline(ctx, model_key, reserve):
    key = (model_key, ctx.device)
    if _STATE["key"] != key:
        drop_pipeline()
        pipe = load_pipeline(ctx, model_key)
        ctx.progress(0.16, "load-model")
        # GGUF 트랜스포머는 조각씩 올리기를 하지 않는다(gpu.py). bf16 트랜스포머는 혼자 안 들어가면 조각씩 올린다.
        no_group = ("transformer",) if MODELS[model_key][1] else ()
        plan = gpu.place(pipe, ["text_encoder", "transformer", "vae"], ctx.device, reserve, no_group=no_group)
        ctx.log(f"placement {plan}")
        _STATE.update(key=key, pipe=pipe)
    return _STATE["pipe"]


def drop_pipeline():
    old = _STATE["pipe"]
    _STATE.update(key=None, pipe=None)
    if old is not None:
        gpu.release(old)


def load_references(ctx, paths):
    """참조 이미지 — 투명 PNG 는 RGBA 그대로 넘긴다(파이프라인이 VAE 에는 4채널, 그림 인코더에는 흰 바탕에 얹어 넘긴다)."""
    from PIL import Image, ImageOps

    out = []
    for p in paths:
        with Image.open(ctx.in_path(p)) as im:
            im = ImageOps.exif_transpose(im)
            out.append(im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB"))
    return out


def oom_message(model_key):
    light = "" if model_key.endswith("-q4") else " 경량화를 더 가벼운 판(GGUF Q4)으로 바꾸거나"
    return (
        "그래픽카드 메모리(VRAM)가 모자라 이미지를 만들지 못했습니다." + light
        + " 크기 · 참조 이미지 수를 줄이고, 그래픽카드를 쓰는 다른 프로그램(게임 · 다른 AI)을 닫은 뒤 다시 실행해 주세요."
    )


def generate(ctx, inputs, settings):
    import torch

    prompt = (inputs.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("프롬프트를 이어 주세요(프롬프트 노드 → 프롬프트 칸).")
    if ctx.device == "cpu":
        raise RuntimeError("Qwen-Image 2.1 은 그래픽카드(NVIDIA CUDA 또는 Apple Silicon)가 있어야 돕니다. CPU 로는 너무 느려 막아 두었습니다.")
    refs = inputs.get("images") or []
    if len(refs) > MAX_REFERENCES:
        raise ValueError(f"참조 이미지는 {MAX_REFERENCES}장까지 이어 주세요(지금 {len(refs)}장).")

    model_key = settings["quantization"]
    variant = MODELS[model_key][0]
    steps = TURBO_STEPS if variant == "turbo" else (int(settings["steps"]) or BASE_STEPS)
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, SEED_MAX)
    transparent = bool(settings["transparent"])
    full_prompt = f"{TRANSPARENT_PREFIX}{prompt}{TRANSPARENT_SUFFIX}" if transparent else prompt
    width = height = None
    if settings["size"] != "auto":
        width, height = (int(v) for v in settings["size"].split("x"))
    images = load_references(ctx, refs) if refs else None
    area_w, area_h = (width, height) if width else (1024, 1024)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        pipe = get_pipeline(ctx, model_key, reserve_for(area_w, area_h, len(refs)))

        def on_step(_pipe, i, _t, kwargs):
            ctx.check_cancel()
            ctx.progress(0.2 + 0.75 * (i + 1) / steps, "denoise")
            return kwargs

        ctx.progress(0.18, "encode")
        call = dict(
            prompt=full_prompt,
            image=images,
            height=height,
            width=width,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=on_step,
        )
        if variant == "base":
            call["num_inference_steps"] = steps
        image = pipe(**call).images[0]
    except MemoryError as exc:
        ctx.log(f"placement failed: {exc}", "warn")
        drop_pipeline()
        raise RuntimeError(oom_message(model_key)) from None
    except torch.OutOfMemoryError:
        drop_pipeline()
        raise RuntimeError(oom_message(model_key)) from None
    except RuntimeError as exc:
        if not gpu.is_cpu_oom(exc):
            raise
        drop_pipeline()
        raise RuntimeError(gpu.RAM_MESSAGE) from None

    ctx.progress(0.98, "save")
    from PIL.PngImagePlugin import PngInfo

    keep_alpha = transparent or any(im.mode == "RGBA" for im in images or [])
    if not keep_alpha and image.mode == "RGBA":
        # 투명 배경을 고르지 않았고 투명한 참조 이미지도 없으면 알파를 버린다(보통 그림은 불투명이 맞다).
        image = image.convert("RGB")
    info = PngInfo()
    info.add_text(
        "parameters",
        f"{full_prompt}\nQwen-Image 2.1 {model_key} · steps {steps} · seed {seed} · {image.width}x{image.height} · references {len(refs)}",
    )
    out = ctx.out_path(f"qwen-image-{seed}.png")
    image.save(out, pnginfo=info)
    report = gpu.peak_report()
    if report:
        ctx.log(report)
    ctx.progress(1.0, "done")
    return {"image": out}


NODES = {"generate": generate}
