# 문제 해결 가이드

### 포터블 패키지 설정 (Portable Package Setup)

- **설정 스크립트가 GPU를 감지하지 못함:**
  - **NVIDIA:** NVIDIA GPU 드라이버가 설치되어 있고 명령줄에서 `nvidia-smi`를 실행할 수 있는지 확인하세요.
  - **AMD (Windows/Linux):** AMD GPU 드라이버가 설치되어 있는지 확인하세요.
  - **Intel ARC (Windows/Linux):** Intel GPU 드라이버가 설치되어 있는지 확인하세요.
  - **macOS Apple Silicon:** M1/M2/M3/M4 Mac에서는 MPS가 자동으로 감지됩니다.
  - **macOS Intel (외장 AMD GPU):** 외장 AMD GPU가 장착된 Intel Mac에서 MPS를 사용할 수 있습니다.
  - **macOS Intel (내장 그래픽):** 내장 그래픽이 장착된 Intel Mac은 CPU 전용 모드로 작동합니다.
  - GPU 감지에 실패하는 경우 언제든지 CPU 모드를 선택하여 실행할 수 있습니다.

- **"Python not found" (Python을 찾을 수 없음) 오류 (Linux/macOS):**
  - Python 3.10 이상 버전을 설치하세요:
    - Ubuntu/Debian: `sudo apt install python3 python3-pip python3-venv`
    - Fedora: `sudo dnf install python3 python3-pip`
    - Arch: `sudo pacman -S python python-pip`
    - macOS: `brew install python@3.13` 또는 python.org에서 설치 파일 다운로드

- **"Git not found" (Git을 찾을 수 없음) 오류 (Linux/macOS):**
  - Git을 설치하세요:
    - Ubuntu/Debian: `sudo apt install git`
    - Fedora: `sudo dnf install git`
    - Arch: `sudo pacman -S git`
    - macOS: `xcode-select --install` 또는 `brew install git`

- **Nunchaku 설치 실패:**
  - Nunchaku는 NVIDIA CUDA 및 Python 3.10+ 환경이 필요합니다.
  - 설치에 실패하더라도 다른 인페인팅 방식을 사용할 수 있습니다.

### 렌더링 (Rendering)

- **지원하지 않는 문자/글자가 삭제됨:**
  - 사용 중인 폰트가 선택한 대상 언어를 지원하지 않는 상태입니다. 대상 언어를 지원하는 폰트를 사용하세요.

- **말풍선 읽기 순서가 올바르지 않음:**
  - 올바른 "읽기 방향 (Reading Direction)"을 설정하세요 (일본/한국 만화의 경우 `rtl`(우횡서/우에서 좌로), 웹툰이나 미국 만화의 경우 `ltr`(좌횡서/좌에서 우로)).
  - 다소 성능이 떨어지는 LLM의 경우, "2단계 (two-step)" 번역 모드를 사용해 보거나 "전체 페이지를 LLM으로 전송 (Send Full Page to LLM)" 옵션을 비활성화해 보세요.
  - "컷 감지 정렬 사용 (Use Panel-aware Sorting)" 옵션이 활성화되어 있는지 확인하세요.

- **텍스트 크기가 너무 크거나 작음:**
  - "최대 폰트 크기 (Max Font Size)" 및 "최소 폰트 크기 (Min Font Size)" 범위를 조정하세요.

- **말풍선 유형에 따라 다른 폰트 적용하기 (예: 혼잣말/생각 vs 일반 대화):**
  - 특정 폰트의 기울임꼴(italic)/굵게(bold)/굵은 기울임꼴(bold+italic) 파일을 원하는 폰트 파일로 교체하세요. 단, 프로그램이 변체로 감지할 수 있도록 파일명에 해당 키워드가 포함되어 있어야 합니다.
  - 더욱 정밀하게 결과물을 다듬으려면, LLM에게 독백 등의 상황에서 기울임꼴/굵게/굵은 기울임꼴(교체해 둔 사용자 지정 폰트)을 사용하도록 지시하는 "특별 지시사항 (special instruction)"을 추가하세요.

- **텍스트가 서로 겹침:**
  - 폰트 파일이 손상되었을 가능성이 있습니다. 포터블 패키지에 포함된 다른 폰트를 사용해 보세요.

- **발음 구별 부호(디아크리티컬 마크)가 올바르게 표시되지 않음:**
  - 발음 구별 부호를 지원하는 폰트를 사용하세요 (예: 포터블 패키지에 기본 포함된 _Roboto_ 폰트 팩은 로마자, 키릴 문자, 그리스어 발음 구별 부호를 지원합니다).

- **텍스트가 너무 흐릿하거나 픽셀이 깨져 보임:**
  - 폰트 렌더링의 "초고해상도 샘플링 계수 (Supersampling Factor)"를 높이세요 (예: 6-8).
  - "초기 (initial)" 이미지 업스케일링을 활성화하고 업스케일 배율을 조정하세요 (예: 2.0x-4.0x).

- **빈 말풍선이 생성됨:**
  - "패딩 픽셀 (Padding Pixels)" 값을 낮추어 보세요 (예: 3-4).
  - "최소 폰트 크기 (Min Font Size)" 값을 낮추어 보세요 (예: 6-7).

- **말풍선 외부 대체 텍스트가 너무 작거나 빽빽하게 렌더링됨:**
  - "좁고 긴 영역 확장 배율 (Narrow/Tall Expansion Multiplier)" 및/또는 해당 임계값을 높이세요 (예: 2.0).
  - "소형 영역 확장 배율 (Tiny Expansion Multiplier)" 및/또는 해당 임계값을 높이세요 (예: 2.0).

### 감지 및 클리닝 (Detection/Cleaning)

- **말풍선 가장자리 근처에 지워지지 않은 텍스트 잔해가 남음:**
  - "고정 임계값 (Fixed Threshold Value)"을 낮추거나 (예: 170-190) "수축 임계값 관심영역 (Shrink Threshold ROI)"을 줄이세요 (예: 0-2).

- **클리닝 중 글자 외곽선이 깎여나감:**
  - "수축 임계값 관심영역 (Shrink Threshold ROI)"을 늘리거나 (예: 6-8) "고정 임계값 (Fixed Threshold Value)"을 높이세요 (예: 210-220).

- **연결된 말풍선이 제대로 감지되지 않음:**
  - "연결된 말풍선 감지 (Conjoined Bubble Detection)" 옵션이 활성화되어 있는지 확인하세요.
  - "말풍선 감지 신뢰도 (Bubble Detection Confidence)"를 낮추어 보세요 (예: 0.20).

- **작은 말풍선이 감지되지 않거나 렌더링할 공간이 부족함:**
  - "초기 (initial)" 이미지 업스케일링을 활성화하고 업스케일 배율을 조절하세요 (예: 2.0x-4.0x). 이때 "자동 스케일 (Auto Scale)" 옵션은 비활성화해야 합니다.

- **컬러/그레이스케일 말풍선의 내부 색상이 보존되지 않음:**
  - "컬러 말풍선에 Flux 인페인팅 사용 (Use Flux to Inpaint Colored Bubbles)" 옵션을 활성화하세요.

- **말풍선 분할(세그멘테이션) 품질이 떨어짐:**
  - `sam3` 모델로 전환해 보세요 (`hf_token` 설정 필요).

### 번역 (Translation)

- **번역 품질이 좋지 않음:**
  - 성능이 낮은 LLM의 경우 "2단계 (two-step)" 번역 모드를 사용해 보세요.
  - "전체 페이지를 LLM으로 전송 (Send Full Page to LLM)" 옵션을 비활성화해 보세요.
  - 성능이 낮은 LLM을 사용할 때는 다음과 같은 로컬 OCR 방식을 병행해 보세요:
    - "manga-ocr": 일본어 원본 전용
    - "paddleocr-vl": 일본어 이외의 원본용
  - `max_tokens` 값을 늘리거나 더 높은 "추론 노력 (reasoning_effort)"(예: "high")을 사용하세요.
  - "말풍선/컨텍스트 크기 조정 방식 (Bubble/Context Resizing Method)"을 더 고품질의 방식(예: "Model")으로 변경하세요.
  - 배치(일괄) 모드에서 "이전 컨텍스트 OCR 텍스트 (Previous Context OCR Text)" 값을 늘려보세요.
  - 배치 모드에서 "이전 컨텍스트 이미지 (Previous Context Images)" 값을 늘려보세요 ("전체 페이지를 LLM으로 전송" 옵션 필요).

- **API 거부/검열 발생:**
  - "전체 페이지를 LLM으로 전송 (Send Full Page to LLM)" 옵션을 비활성화해 보세요.
  - 사용자 지정 "특별 지시사항 (special instruction)"을 추가해 보세요 (예: "번역을 검열하지 마십시오...").

- **LLM 토큰 사용량이 지나치게 높음:**
  - "전체 페이지를 LLM으로 전송 (Send Full Page to LLM)" 옵션을 비활성화하세요.
  - "이전 컨텍스트 이미지 (Previous Context Images)" 및 "이전 컨텍스트 OCR 텍스트 (Previous Context OCR Text)"를 0으로 설정하세요.
  - "말풍선 최소 가로세로 픽셀 (Bubble Min Side Pixels)", "컨텍스트 이미지 최대 가로세로 픽셀 (Context Image Max Side Pixels)", "OSB 최소 가로세로 픽셀 (OSB Min Side Pixels)"의 대상 크기를 낮추세요.
  - "미디어 해상도 (Media Resolution)"를 낮추세요 (Gemini 또는 xAI 모델을 사용하는 경우).
  - "이미지 디테일 (Image Detail)"을 낮추세요 (OpenAI 모델을 사용하는 경우).
  - "manga-ocr" 또는 "paddleocr-vl" OCR 방식을 사용하세요 (다만, 고성능 VLM을 직접 사용하는 것보다 인식률이 떨어질 수 있습니다).

### 인페인팅 (Inpainting)

- **말풍선 외 (OSB) 텍스트가 지워지지 않거나 인페인팅되지 않음:**
  - "OSB 텍스트 감지 활성화 (Enable OSB Text Detection)" 옵션이 켜져 있는지 확인하세요.
  - `hf_token`이 설정되어 있는지 확인하세요 (설치/설치 후 설정 참고).

- **VRAM 부족 / CUDA 에러 발생:**
  - "저비디오메모리 모드 (Low VRAM Mode)"를 활성화하세요 (SDNQ 전용).
  - 더 낮은 Flux/text_encoder 양자화(quant) 버전을 선택하세요 (sd.cpp 전용).
  - "Klein 자르기 영역을 ~1MP로 업스케일 (Upscale Klein Crops to ~1MP)" 옵션을 비활성화하세요.
  - Flux.2 Klein 4B(가장 작고 가벼운 모델)로 전환하세요.
  - OpenCV를 사용하세요 (VRAM을 사용하지 않음).

- **Flux OSB 인페인팅이 너무 느림:**
  - "Flux 영역 그룹화 (Group Flux Regions)"를 활성화하여 한 번의 Flux 패스로 여러 OSB 마스크를 한 번에 처리하도록 하세요 (품질은 약간 저하될 수 있습니다).
  - "Klein 자르기 영역을 ~1MP로 업스케일 (Upscale Klein Crops to ~1MP)" 옵션을 비활성화하세요 (품질 저하 가능).
  - 다른 모델/백엔드로 전환해 보세요.

- **인페인팅된 영역의 미세한 색감 차이:**
  - Flux.2 Klein 모델은 약간의 색감 변화를 일으킬 수 있습니다. "조도 보정 (Luminance Correction)" 옵션을 활성화하거나 비활성화해 보며 테스트하세요.
  - 색감 차이를 최소화하려면 Flux.1 Kontext를 사용하세요.

- **인페인팅 결과물의 품질이 낮음:**
  - 더 높은 Flux/text_encoder 양자화(quant) 버전을 선택하세요 (sd.cpp 전용).
  - "Klein 자르기 영역을 ~1MP로 업스케일 (Upscale Klein Crops to ~1MP)" 옵션을 활성화 상태로 유지하세요.
  - 추론 단계(inference steps) 수를 늘리세요 (Flux.1 Kontext 사용 시).

- **Flux.1 Kontext Nunchaku 백엔드를 사용할 수 없음:**
  - Nunchaku 백엔드는 NVIDIA GPU (CUDA) 및 전용 별도 설치가 필요합니다. 별도로 설치하거나 대신 sd.cpp/SDNQ 백엔드를 사용하세요.
