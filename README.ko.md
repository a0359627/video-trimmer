# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 개요 (Overview)

**Video Trimmer**는 토킹헤드 비디오, 튜토리얼 및 발표 녹화물을 위한 AI 자동 러프컷 및 트리밍 엔진입니다. **Google Vertex AI Gemini 3.8 Flash** 멀티모달 비디오 추론, **Whisper 단어 단위 음향 타임스탬프**(`mlx-whisper`), **4계층 통합 아키텍처(4-Layer Unified Architecture)** 및 **음향 온셋 스내핑(Acoustic Onset Snapping)**을 결합하여 NG 테이크, 말더듬, 무음 구간을 제거하면서 연속적인 문장 흐름을 유지하고 NLE 타임라인(`.xml`, `.fcpxml`, `.edl`, `.csv`)과 렌더링된 MP4 비디오를 생성합니다.

---

## 4계층 통합 아키텍처 및 핵심 기능

1. **계층 1: 순수 음향 및 구두점 기반 절 분할 (`transcribe.py`)**:
   - 호흡 휴지(`gap >= 0.20s`), 발화 지연 장음(`word_dur >= 1.20s`), 문장 부호 종결, 화자 교체의 물리적 경계만으로 `Sentence ID`를 분할하며, 미세 휴지(`gap < 0.25s`) 구간의 접속사 결합(`CONJUNCTIONS`)을 보존합니다. Python 문자열 유사도를 통한 의미 추측을 완전히 배제합니다.
2. **계층 2: 듀얼 모드 LLM 테이크 중재 (`gemini-3.8-flash`)**:
   - **Mode A: 대본 기반 단조 정렬 (`--script` 지정 시)**: 대본을 `[Script Block 01] .. [Script Block NN]`으로 구성하고 단조 순서로 각 블록당 최대 1개의 최종 성공 테이크만 선택합니다.
   - **Mode B: 무대본 의도 윈도우 중재 (`--script` 생략 시)**: 중단된 미완성 조각(Abandoned Fragment)은 제거하고 의도적인 수사적 반복 강조(3회 반복 강조 등)는 보존합니다.
3. **계층 3: 서브 유닛 확장 및 단어 경계 트리밍 (`resolve_clip_sub_units`)**:
   - 다중 문장 범위를 개별 `Sentence ID`로 확장하고 `transcript`에 맞춰 시작/끝 Whisper 단어 경계(`_trim_matched_words_by_transcript`)를 정밀하게 정렬합니다.
4. **계층 4: 글로벌 클립 간 무결성 병합 (`coalesce_adjacent_sub_units`)**:
   - 인접한 클립이 연속된 `Sentence ID`이고 물리적 단어 간격이 `< 0.40s`인 경우 단일 연속 클립으로 병합하여 문장 내부의 불필요한 점프컷을 제거합니다.
5. **음향 온셋 스내핑 및 15 ms 등전력 마이크로 크로스페이드**:
   - 성대 진동 80 ms 전에 컷 포인트를 배치하고 모든 편집 경계에 15 ms 마이크로 페이드(`afade=t=in:d=0.015:curve=iqsin` 및 `afade=t=out:d=0.015:curve=oqsin`)를 적용하여 팝 노이즈를 방지합니다.
6. **Agent Plugins 1.0 표준 아키텍처 및 멀티 NLE 타임라인 지원**:
   - 핵심 스크립트와 프롬프트는 `skills/video-trimmer/scripts/` 및 `skills/video-trimmer/prompts/`(SSOT)에 위치하며 루트 심볼릭 링크 없는 순수 구조와 2계층 `AGENTS.md` / `rules/AGENTS.md`를 제공합니다. **FCP7 XML**, **FCPXML**, **CMX 3600 EDL** 및 **CSV**를 내보냅니다.

---

## 설치 및 Google Cloud 설정

```bash
# 1. FFmpeg 설치 및 Agent Plugin으로 클론 (권장)
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# (선택 사항) 기존 단일 Skill 디렉터리 설치 (~/.gemini/config/skills/ 호환 방식)
ln -s ~/.gemini/config/plugins/video-trimmer/skills/video-trimmer ~/.gemini/config/skills/video-trimmer

pip install -r requirements.txt
pip install mlx-whisper

# 2. ADC 인증 및 setup.sh 실행
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 명령줄 사용법 (CLI Usage)

```bash
# Mode B: 무대본 자동 러프컷 (기본값은 빠른 Static Multimodal 모드)
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4"

# Mode A: 촬영 대본 기반 단조 정렬
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md"

# Agentic 비디오 이해 모드 명시적 활성화
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# Google Drive 공유 링크에서 직접 러프컷 실행
python3 skills/video-trimmer/scripts/video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" -o output/
```

---

## Google Drive 연동 및 GCS 2단계 수명 주기 정책

| GCS 경로 접두사 (`matchesPrefix`) | 저장 객체 | 보관 기간 (`age`) | 목적 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 스테이징된 원본 비디오 (`raw/<filename>.mp4`) | **2일 (`age: 2`)** | 추론 완료 후 즉시 삭제되며 안전 백업으로 2일 후 자동 삭제됩니다. |
| **`output/`**, **`deliverables/`**, **`trimmed/`** | 트리밍된 비디오, XML/FCPXML 타임라인 | **15일 (`age: 15`)** | 팀 검토를 위해 15일간 보관한 후 자동 삭제합니다. |

---

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)에 따라 라이선스가 부여됩니다.
