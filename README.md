# Video Labelling — Gameplay Content Moderation (PoC)

게임 관련 영상을 **shot(segment) 단위**로 자동 분류하는 **Agentic Auto-Labelling 시스템**.
최우선 목표는 **일관적(consistent)** 이고 **근거 추적(traceable)** 가능한 데이터 구축이다.
(실서비스용 경량 모더레이션 모델 서빙은 범위 밖.)

PoC 범위: PEGI 하위 **Gambling / Bad Language / Sex** 3개 카테고리, 연령 스코어 **0–5**.

## System Architecture

```
                 ┌──────────────────────────────────────────────────────────┐
  YouTube video  │  INGESTION  (fixed MLLM batch — no agent)                 │
   ids ─────────▶│  fetch(yt-dlp) → segment(30s/1fps) → ASR(transcript)      │
                 │   → MLLM caption(vision+audio) → embed(text|image) → store │
                 └───────────────────────────────┬──────────────────────────┘
                                                 │  Video · Segment · base_attributes
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │  STORAGE (Data Warehouse)                                         │
        │  Postgres + pgvector (dense) + BM25 (lexical)                     │
        │  blobs → local FS / MinIO (video·keyframes·audio; DB holds ptrs)  │
        └───────┬───────────────────────────────────────────────┬──────────┘
                │                                                │
                ▼                                                ▼
 ┌──────────────────────────────────┐          ┌───────────────────────────────┐
 │  LABELLING AGENT (LangGraph)     │  policy  │  POLICY layer                 │
 │  shot-window sequential:         │◀────────▶│  node tree (Rubric/Attribute/ │
 │  LOAD→RETRIEVE→DERIVE→SCORE→      │  RAG     │  Edge), node ver + set snapshot│
 │  CHECK→SIDE_FX→COMMIT            │          │  change-request queue         │
 │  tools: search_policies,         │          │  bootstrap (PEGI seed → v1)   │
 │   find_similar_segments(precedent),         └───────────────────────────────┘
 │   get_frames, expand_window,     │
 │   revise_ingestion(auto+log),    │  emits Label (full trace):
 │   propose_policy_change(→queue), │  score · rationale · cited_policies ·
 │   emit_label                     │  evidence · used_segments · tool_trace
 └──────────────────────────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  SERVICE LAYER (packages/tools) — one implementation, two callers          │
 │  ├─ FastAPI (app/backend)  → Next.js UI: Data Viewer + Monitoring          │
 │  └─ LangGraph agent tools  (same functions, no duplication)                │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Layout

| Path | Role |
|---|---|
| `packages/schemas` | 공유 Pydantic 계약 (Video/Segment/Attribute/Label/Policy…) |
| `packages/tools` | stateless service layer (storage·blob·embeddings·retrieval·policy_store) |
| `packages/models` | 모델 provider 추상화 + config/profile 로더 |
| `ingestion` | 고정 MLLM batch 파이프라인 |
| `labelling` | LangGraph orchestrator agent + agent tools |
| `policy` | bootstrap (PEGI seed → policy-set v1) |
| `app/backend` | FastAPI (service layer 래핑) |
| `app/ui` | Next.js (Data Viewer + Monitoring) |
| `db` | SQLAlchemy models + Alembic |
| `config` | `config.yaml` + `profiles/*.yaml` (proprietary ↔ local vLLM) |

## Setup

```bash
# 1. Infra (Postgres + pgvector)
docker compose up -d postgres

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Config
cp .env.example .env          # fill provider API keys; pick MODEL_PROFILE

# 4. DB schema (first migration must CREATE EXTENSION vector + tsvector indexes)
alembic -c db/alembic.ini upgrade head

# 5. Backend / UI
uvicorn backend.main:app --reload           # http://localhost:8000
cd app/ui && npm install && npm run dev      # http://localhost:3000
```

### Video 데이터
PoC는 **YouTube에서 취득 가능한 영상**으로 한정한다. 저장소에는 **Video ID만 기록**하며,
실제 영상은 [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) 등으로 받을 수 있다.

## Model providers
역할별(mllm / asr / text_embedding / image_embedding / agent_llm) provider를 `config/profiles/<name>.yaml`
에서 지정한다. 기본 profile은 proprietary API(OpenAI/Gemini/Claude)이며, 실험 시 local vLLM
엔드포인트를 가리키는 profile로 교체한다. 코드 변경 없이 config로 전환.
