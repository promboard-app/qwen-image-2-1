# Qwen-Image 2.1 — 글로 만들기 · 고치기 · 투명 배경

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Qwen-Image 2.1 생성 · 편집** 노드 하나를 더합니다. 프롬프트만 이으면 글로 이미지 한 장(PNG)을 만들고, 참조 이미지(4장까지)를 같이 이으면 그 이미지를 고치거나 섞습니다. '투명 배경'을 켜면 배경이 투명한 PNG 를 만듭니다. 모델은 Qwen 의 [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)입니다.

> **검증 안 됨 · 비상업 전용.** PromBoard 팀은 이 플러그인을 만들기만 하고 실제로 돌려 보지 않았습니다. 결과 그림이 없는 것도 그래서입니다. 코드는 diffusers 0.42.0 의 공식 파이프라인(QwenImage21Pipeline)과 GGUF 읽기 기능을 그대로 씁니다. 모델 라이선스는 연구 · 평가 목적만 허락합니다(아래 「라이선스와 조건」).

## 경량화 (판과 정밀도)

노드의 **경량화** 알약에서 고릅니다. 고른 판의 파일만 받습니다. **자동**은 이 PC 그래픽카드 메모리로 Turbo 판을 고릅니다 — 20GB 이상 원본(bf16), 10GB 이상 GGUF Q8, 그 밖(맥 · AMD · 'CPU 만 쓰기' 포함)은 GGUF Q4.

| 경량화 알약 | 판 | 트랜스포머 | 그래픽카드 메모리 |
|---|---|---|---|
| Turbo · 원본 (bf16) | [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | 14.2GB | 20GB 이상 |
| Turbo · GGUF Q8 | 같은 판 | 7.6GB | 10GB 이상 |
| Turbo · GGUF Q4 | 같은 판 | 4.2GB | 8GB 이상 |
| 2.1 · 원본 (bf16) | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 14.2GB | 20GB 이상 |
| 2.1 · GGUF Q8 | 같은 판 | 7.6GB | 10GB 이상 |
| 2.1 · GGUF Q4 | 같은 판 | 4.2GB | 8GB 이상 |

- Turbo 는 저장된 8단계 일정 그대로 돌아 빠르고, 2.1 은 증류하지 않은 원래 판으로 40단계(공식 예시)라 느립니다.
- GGUF 판은 unsloth 가 줄인 파일입니다([Turbo](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) · [2.1](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF), Q8_0 · Q4_K_M). 원본보다 메모리를 덜 쓰는 대신 그림이 조금 거칠 수 있습니다.
- 글 · 그림 인코더(Qwen3-VL-8B, bf16 17.5GB) · 프로세서 · VAE(RGBA 4채널)는 공식 Turbo 저장소의 것을 모든 판이 같이 씁니다. 2.1 판은 스케줄러 설정을 공식 2.1 저장소에서 따로 받습니다.
- 그래픽카드 메모리 값은 트랜스포머 크기에 계산 몫을 더한 어림값입니다(실측 아님).

## 설정

| 알약 | 뜻 |
|---|---|
| 크기 | 자동(마지막 참조 이미지의 가로세로 비로 약 100만 픽셀, 참조 이미지가 없으면 1024×1024) · 정사각 1024 · 가로/세로 16:9 · 4:3 · 3:2 · 정사각 2048(공식 해상도, 느림) |
| 투명 배경 | 켜면 공식 카드의 투명 이미지 프롬프트 틀("This is an RGBA image with transparency. … The image has alpha channel and the background is transparent.")로 감싸 RGBA PNG 를 만듭니다 |
| 시드 | -1 이면 매번 다른 그림. 쓴 시드는 PNG 정보(parameters)에 남습니다 |
| 단계 수 (2.1) | 2.1 판만. 0 이면 공식 예시 40. Turbo 는 8단계 고정 |

## 받는 것과 용량

| 고른 판 | 처음 받는 양 |
|---|---|
| 원본 (bf16) | 32.45GB(공통 18.22GB + 트랜스포머 14.23GB) |
| GGUF Q8 | 25.86GB |
| GGUF Q4 | 22.42GB |

Turbo · 2.1 모두 같습니다. 판을 바꾸면 그 판의 트랜스포머만 더 받습니다.

## 필요한 그래픽카드 · 메모리

- 플러그인은 부품을 그래픽카드 메모리에 맞게 올립니다. 다 들어가면 전부 올리고, 아니면 글 · 그림 인코더 → 트랜스포머 → VAE 를 쓸 차례에만 올립니다. 혼자도 안 들어가는 부품은 조각씩 올립니다(글 · 그림 인코더 17.5GB 는 대부분의 소비자 그래픽카드에서 이렇게 돌아 느립니다). GGUF 트랜스포머는 조각씩 올리지 않고 "더 가벼운 판으로 바꾸라"는 오류를 냅니다.
- 시스템 메모리(RAM)는 GGUF 판 32GB 이상, 원본 판 64GB 이상을 권합니다(가중치를 모두 메모리에 올립니다).
- NVIDIA(CUDA) 그래픽카드용입니다. CPU 로는 너무 느려 막았습니다. 리눅스 · 맥(애플 칩) 잠금 파일도 있지만 돌려 보지 않았습니다.

## 라이선스와 조건

- 모델: **Qwen Research License**([원문](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)). 연구 · 평가 목적으로만 쓸 수 있습니다. 상업 사용은 Alibaba 와 따로 계약해야 합니다. 이 모델로 학습 · 개선한 AI 모델을 배포하면 'Built with Qwen' 을 표기하고, 'Qwen' 을 그 모델의 주 이름으로 쓰지 않습니다. unsloth GGUF 판도 같은 라이선스를 따릅니다.
- PromBoard 는 설치할 때 이 조건을 보여 주고, 만든 결과물에 '비상업' 표시를 붙입니다.
- 이 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).

## 설치

1. PromBoard 앱(1.0.3 이상) → 플러그인 매니저 → **커뮤니티 플러그인**에서 설치합니다(라이선스 조건 확인).
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(diffusers 0.42.0 · transformers 5.20.2 · accelerate · gguf, torch 는 앱이 고정한 2.8.0)와 고른 판의 모델을 받습니다.
4. 캔버스 빈 곳을 우클릭하면 메뉴 맨 아래 플러그인 묶음에 **Qwen-Image 2.1 생성 · 편집**이 있습니다.

## English

A PromBoard community plugin that adds one node, **Qwen-Image 2.1 생성 · 편집** (generate). With a prompt only it renders one PNG; with up to four reference images it edits or combines them; *transparent background* produces an RGBA PNG. Model: Qwen's [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).

> **Unverified, non-commercial only.** The PromBoard team built this plugin but did not run it, which is why there are no result pictures. The code uses diffusers 0.42.0's official QwenImage21Pipeline and its GGUF loader.

- **Quantization** setting picks the variant and precision; only the chosen files are downloaded. *Auto* picks Turbo: bf16 on 20 GB+ VRAM, GGUF Q8 on 10 GB+, otherwise GGUF Q4 (also on Macs, AMD or CPU-only setups).
  - Turbo (stored 8-step schedule, fast) or 2.1 (undistilled, 40 steps), each as bf16 (14.2 GB transformer, ~20 GB VRAM), unsloth GGUF Q8 (7.6 GB, ~10 GB) or GGUF Q4 (4.2 GB, ~8 GB). VRAM figures are estimates, not measurements.
  - Text/vision encoder (Qwen3-VL-8B, 17.5 GB), processor and RGBA VAE come from the official Turbo repo and are shared by all options.
- Transparent background wraps the prompt in the official card's RGBA template.
- Downloads: 32.45 GB (bf16) · 25.86 GB (Q8) · 22.42 GB (Q4). Switching options downloads only the new transformer.
- Memory: everything goes on the GPU when it fits; otherwise each part is moved on only while it runs and streamed block by block if needed (the 17.5 GB encoder is streamed on most consumer GPUs, which is slow). 32 GB+ system RAM for GGUF, 64 GB+ for bf16. NVIDIA CUDA; CPU is blocked; Linux and Apple-silicon lock files exist but are untested.
- License: **Qwen Research License** — research and evaluation only; commercial use needs a separate agreement with Alibaba; models trained or improved with it must say "Built with Qwen" and must not use "Qwen" as their main name. PromBoard shows these terms on install and marks outputs as non-commercial. Plugin code: Apache-2.0.

## License

Plugin code: Apache-2.0. Model weights are downloaded from their original sources at first run under their own license (Qwen Research License).
