# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 概要 (Overview)

**Video Trimmer** は、トーク動画、チュートリアル、プレゼンテーション録画向けの AI 自動ラフカット＆トリミングエンジンです。**Google Vertex AI Gemini 3.8 Flash** のマルチモーダル動画推論、**Whisper 単語レベル音響タイムスタンプ**（`mlx-whisper`）、**4 レイヤー統合アーキテクチャ**、および **音響オンセット・スナッピング** を組み合わせ、NG テイクや言い淀み、無音区間を自動除去しながら自然な連続発話を維持し、NLE タイムライン（`.xml`, `.fcpxml`, `.edl`, `.csv`）と MP4 動画を出力します。

---

## 4 レイヤー統合アーキテクチャと主な機能

1. **レイヤー 1：純音響・句読点ベースの節分割 (`transcribe.py`)**：
   - 息継ぎ休止（`gap >= 0.20s`）、言い淀みによる長音化（`word_dur >= 1.20s`）、句読点閉鎖、および話者交替の物理境界のみで `Sentence ID` を分割し、微細休止（`gap < 0.25s`）を跨ぐ接続詞結合（`CONJUNCTIONS`）を保持します。Python 側の文字列類似度によるリテイク推測は一切行いません。
2. **レイヤー 2：デュアルモード LLM テイク選定 (`gemini-3.8-flash`)**：
   - **Mode A：スクリプトアンカー単調アライメント（`--script` 指定時）**：台本を `[Script Block 01] .. [Script Block NN]` に分割し、単調順序で各ブロック最大 1 つの最終成功テイクのみを採用します。
   - **Mode B：台本なしインテントウィンドウ調停（`--script` 省略時）**：途中放棄された断片（Abandoned Fragment）を除去しつつ、意図的な反復強調表現（3 回繰り返す強調など）を保護します。
3. **レイヤー 3：サブユニット展開と単語境界トリミング (`resolve_clip_sub_units`)**：
   - 複数文の範囲を個別の `Sentence ID` に展開し、`transcript` に合わせて語頭・語尾の Whisper 単語境界（`_trim_matched_words_by_transcript`）を精密に整列させます。
4. **レイヤー 4：グローバル・クリップ間結合 (`coalesce_adjacent_sub_units`)**：
   - 隣接クリップが連続する `Sentence ID` であり、単語間ギャップが `< 0.40s` の場合、単一の連続クリップに自動統合し、文中の不自然なジャンプカットを排除します。
5. **音響オンセット・スナッピング、15 ms 等パワー音声マイクロクロスフェード＆キーフレーム・ハードウェア高速レンダリング (`acoustic.py` / `render.py`)**：
   - 声帯振動の 80 ms 前にカット点を配置し、各クリップで `-ss` / `-to` 前置キーフレーム高速シーク（NG 区間のデコード省略）、Apple Silicon `VideoToolbox` ハードウェア加減速（`-hwaccel videotoolbox` + `h264_videotoolbox`、`libx264` 自動フォールバック付き）、1 秒 GOP（`-g 30`）、および 15 ms マイクロフェード（`afade=t=in:d=0.015:curve=iqsin` / `afade=t=out:d=0.015:curve=oqsin`）を適用します。
6. **Agent Plugins 1.0 準拠構造とマルチ NLE タイムライン出力**：
   - コアスクリプトとプロンプトは `skills/video-trimmer/scripts/` および `skills/video-trimmer/prompts/`（SSOT）に配置され、ルート直下のシンボリックリンクを持たない純粋な構成と 2 層 `AGENTS.md` / `rules/AGENTS.md` を備えています。**FCP7 XML**、**FCPXML**、**CMX 3600 EDL**、**CSV** を出力します。

---

## セットアップと Google Cloud 構成

```bash
# 1. FFmpeg のインストールと Agent Plugin としてのクローン（推奨）
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# （オプション）従来の単一 Skill ディレクトリへのインストール（~/.gemini/config/skills/ 互換）
ln -s ~/.gemini/config/plugins/video-trimmer/skills/video-trimmer ~/.gemini/config/skills/video-trimmer

pip install -r requirements.txt
pip install mlx-whisper

# 2. ADC 認証と setup.sh の実行
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## コマンドライン使用法 (CLI Usage)

```bash
# Mode B：台本なし自動ラフカット（デフォルトは高速な Static Multimodal モード）
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4"

# Mode A：台本（スクリプト）を用いた単調アンカーアライメント
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md"

# Agentic 動画理解モードを明示的に有効化
python3 skills/video-trimmer/scripts/video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# Google Drive 共有リンクからの直接ラフカット
python3 skills/video-trimmer/scripts/video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" -o output/
```

---

## Google Drive 連携と GCS 2 階層ライフサイクルポリシー

| GCS パス接頭辞 (`matchesPrefix`) | 保存対象 | 保持期間 (`age`) | 目的 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 一時ステージング動画 (`raw/<filename>.mp4`) | **2 日間 (`age: 2`)** | 推論完了後に即時削除され、バックアップとして 2 日後に自動削除されます。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | トリミング済み動画、XML/FCPXML タイムライン | **15 日間 (`age: 15`)** | チーム確認用に 15 日間保持した後、自動削除します。 |

---

## ライセンス (License)

本プロジェクトは [MIT License](LICENSE) の下で提供されています。
