<p align="center">
  <a href="../../../README.md">English</a> |
  <a href="../zh/README.md">简体中文</a> |
  <a href="README.md">한국어</a> |
  <a href="../ja/README.md">日本語</a>
</p>

<h1 align="center"><b>MangaTranslator</b></h1>

<p align="center">
  <img src="https://img.shields.io/github/v/release/meangrinch/MangaTranslator?label=Release&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/downloads/meangrinch/MangaTranslator/total?label=Downloads&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/license/meangrinch/MangaTranslator?labelColor=181717&color=2ea44f" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&labelColor=181717" />
</p>

<div align="center">
AI를 사용하여 만화, 웹툰(Manhwa), 코믹스를 번역하는 엔드투엔드 파이프라인입니다. 말풍선 및 말풍선 외부 텍스트를 감지하고, 확산 모델 인페인팅으로 원본 텍스트를 제거하며, LLM 및 커스텀 폰트를 사용하여 60개 이상의 언어로 식자 번역을 수행합니다.
</div>

<br/>

<div align="center">
  <table>
    <tr>
      <th style="text-align: center">원본</th>
      <th style="text-align: center">번역본 (원클릭 완료)</th>
    </tr>
    <tr>
      <td><img src="../../images/example_original.jpg" width="400" /></td>
      <td><img src="../../images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

---

## 빠른 시작

### 1. 다운로드

- **포터블 패키지 (권장):** [Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)에서 포터블 빌드를 다운로드합니다.
  - Windows: 추가 요구 사항 없음.
  - Linux / macOS: Python 3.10+ 및 Git 설치 필요.
- **소스 코드 설치:**
  ```bash
  git clone https://github.com/meangrinch/MangaTranslator.git
  cd MangaTranslator
  python -m venv venv
  .\venv\Scripts\activate # 또는 source venv/bin/activate (Linux/macOS)
  pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130 # 또는 pip install torch==2.11.0 torchvision==0.26.0 (macOS)
  pip install -r requirements.txt
  ```

*자세한 내용은 [설치](INSTALLATION.md)를 참조하세요.*

### 2. 설정

- **제공자:** Web UI의 Config → Translation 탭에서 LLM 제공자(예: Google, OpenAI, Anthropic, DeepSeek...)를 선택하고 API 키를 입력하거나 로컬 OpenAI 호환 엔드포인트(예: llama.cpp)를 사용합니다. 설정을 저장하면 세션 간에 API 키 및 설정이 유지됩니다.
- **말풍선 외부 텍스트 (선택 사항):** Config → OSB Text 탭으로 이동하여 [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)에 접근할 수 있는 Hugging Face 토큰을 설정합니다.
- **환경 변수:** 또는 시스템 환경 변수(예: `GEMINI_API_KEY`, `HF_TOKEN`)에 자격 증명을 구성하여 Web UI 및 CLI 모두에서 사용할 수 있습니다.

*자세한 내용은 [설정](CONFIGURATION.md)을 참조하세요.*

### 3. 번역

- **Web UI:** 이미지를 Translator/Batch 탭에 업로드하고 Translate를 클릭합니다.
- **CLI:**
  ```bash
  python main.py --input "path/to/page.jpg" --input-language "Japanese" --output-language "English" --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>
  ```
*추가 예시는 [CLI](CLI.md)를 참조하세요.*

---

## 주요 기능

- **감지**: 말풍선 및 말풍선 외부 텍스트 감지 (YOLO, SAM 2.1/3)
- **클리닝**: 말풍선 및 배경 텍스트 인페인팅 (FLUX.2 Klein, FLUX.1 Kontext 또는 OpenCV)
- **번역**: 60개 이상의 언어를 지원하는 OCR 및 번역 (클라우드 API 또는 로컬 LLM)
- **렌더링**: 텍스트 정렬, 자동 줄바꿈 및 커스텀 폰트 팩을 지원하는 커스텀 텍스트 렌더링
- **업스케일링**: 텍스트 영역 및 전체 페이지 아트워크 업스케일링 (2x-AnimeSharpV4)
- **처리**: 단일 이미지, 폴더 및 ZIP 압축 파일 배치 처리 (디렉토리 구조 보존)
- **설정**: 다양한 페이지 레이아웃에 대응하고 출력 품질을 미세 조정할 수 있는 유연한 설정
- **인터페이스**: Web UI (Gradio) 및 CLI
- **자동화**: 원클릭 번역, 수동 개입 불필요

---

## 문서

- [하드웨어 요구 사항](HARDWARE_REQUIREMENTS.md)
- [설치](INSTALLATION.md)
- [설정](CONFIGURATION.md)
- [CLI](CLI.md)
- [폰트](FONTS.md)
- [문제 해결](TROUBLESHOOTING.md)

---

## 프로젝트 후원

MangaTranslator는 오픈 소스이며 무료입니다. 이 프로젝트가 시간을 절약해 주었거나 독서 경험을 향상시켰다면, 개발 후원을 고려해 주세요!

<p align="center">
  <a href="https://ko-fi.com/grinnch" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Support on Ko-fi" height="38"/>
  </a>
</p>

---

## 라이선스 및 크레딧

- 라이선스: Apache-2.0 ([LICENSE](../../../LICENSE) 참조)
- 제작자: [grinnch](https://github.com/meangrinch)

<details>
<summary><b>ML 모델 및 라이브러리</b></summary>

- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 Speech Bubble Detector: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic Text and Bubble Detector RT-DETR-v2: [ogkalu](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- SAM 3: [Meta AI](https://huggingface.co/facebook/sam3)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- PaddleOCR-VL-1.6: [PaddlePaddle](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 4B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- FLUX.2 Klein 9B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- Nunchaku: [Nunchaku AI](https://github.com/nunchaku-ai/nunchaku)
- SDNQ Quants: [Disty0](https://huggingface.co/Disty0)
- Unsloth Quants: [Unsloth](https://huggingface.co/unsloth)
- stable-diffusion.cpp: [leejet](https://github.com/leejet/stable-diffusion.cpp)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)

</details>
