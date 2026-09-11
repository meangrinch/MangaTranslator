# CLI

배치 처리, 스크립트 실행 및 헤드리스 번역을 위해 `python main.py`를 실행합니다.

## 예시

```bash
# 단일 이미지, 일본어 -> 영어, Google 제공자, 말풍선 외부 텍스트 파이프라인, 커스텀 말풍선 외부 폰트
python main.py --input <image_path> \
  --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> \
  --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>

# 폴더 배치 처리, 한국어 -> 중국어(간체), OpenAI 호환 제공자 (llama.cpp), 말풍선 외부 텍스트 파이프라인, 커스텀 말풍선 외부 폰트
python main.py --input <folder_path> --batch --font-dir "fonts/Noto Sans SC" \
  --input-language "Korean" --output-language "Chinese (Simplified)" \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output --osb-enable --osb-font-dir "fonts/Noto Sans SC" --osb-hf-token <...>

# 클리닝 전용 모드 (번역 없음)
python main.py --input <image_path> --cleaning-only

# 업스케일링 전용 모드 (번역 없음)
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# 전체 옵션 확인
python main.py --help
```
