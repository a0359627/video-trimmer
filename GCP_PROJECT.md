# Google Cloud & Gemini Enterprise 專案資訊

本文件記錄本專案所對應之 Google Cloud Platform (GCP) 與 Gemini Enterprise 核心設定資訊。

---

## 專案核心參數

| 項目 | 設定值 | 說明 |
| :--- | :--- | :--- |
| **GCP 專案 ID** | `panmedia-internal-ge` | 專案名稱與 GCP Project ID |
| **服務位置 (Location)** | `global` | Gemini Enterprise 與 Vertex AI 運作區域 |
| **Gemini Enterprise Engine ID** | `ge-panmedia-1787638523173_1787638629135` | 企業版 Agentic 引擎專屬識別碼 |

---

## 控制台快速連結

* **Gemini Enterprise Agentic 控制台**：  
  [Panmedia Internal GE 控制台](https://console.cloud.google.com/gemini-enterprise/locations/global/engines/ge-panmedia-1787638523173_1787638629135/agentic/agents?hl=zh-TW&project=panmedia-internal-ge&folder=&organizationId=)

---

## 後續配置備註 (Setup Reference)

若要將本機 `video-trimmer` 專案環境連結至此專案，標準指令如下：

```bash
# 1. 切換 gcloud 專案
gcloud config set project panmedia-internal-ge

# 2. 執行本專案 GCP 儲存桶與環境自動配置
./setup.sh --project panmedia-internal-ge --region us-central1
```
