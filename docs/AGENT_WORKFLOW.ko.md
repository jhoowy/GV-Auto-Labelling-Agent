# 에이전트 워크플로 (Agent Workflow)

*언어: [English](AGENT_WORKFLOW.md) · **한국어***

labelling 에이전트는 **LangGraph 상태 머신**이다 — 고정된 단계(stage)들이 각각
제한된 도구 집합을 가지며, 자유로운 ReAct가 **아니다**. shot window가 영상 위를
슬라이딩하며 모든 shot이 라벨링될 때까지 진행한다. 전역 영상 컨텍스트와 롤링
carry-over 요약이 항상 주입된다.

```mermaid
flowchart LR
  S((시작)) --> LOAD["LOAD"]
  LOAD --> RETRIEVE["RETRIEVE"] --> JUDGE["JUDGE"] --> SIDE_FX["SIDE_FX"] --> COMMIT["COMMIT"]
  COMMIT -->|"cursor += stride (shot 남음)"| LOAD
  COMMIT -->|"cursor ≥ n (완료)"| E((종료))
```

이전 초안의 `DERIVE`와 `CHECK`는 **JUDGE로 흡수**되었다: 합성된 정책이 있는
카테고리는 JUDGE가 명시적 **SELECT → EXTRACT → DECIDE → REVIEW → STORE**
파이프라인을 실행한다(트리가 score를 결정적으로 산출하며, REVIEW는 적절성을
검토하지만 score를 뒤집을 수 없다). 아직 holistic 폴백인 카테고리는 판정 단계
안에서 구조화 신호를 파생하고 선례 일관성을 체크한다. `SIDE_FX`는 유지된다.

## Orchestrator 모델

Orchestrator는 config의 `agent_llm` 역할로 선택되는 **멀티모달 LLM**이다(config
기반, 교체 가능). 현재는 **Gemini 3.5 Flash**(API). 향후 shot의 raw 오디오를
입력으로 추가할 수 있도록 **오디오 가능 모델**을 의도적으로 선택했다.

인지(perception)는 **가볍게** 유지한다: 모델은 clip 전체를 보지 않는다. shot마다
**uniform 샘플링된 프레임 ≤ 5장** + coarse `summary` + 해당 shot의 ASR 텍스트를
받는다. 프레임이 부족하면 에이전트는 **`expand_frames`**를 호출해 추가/조밀한
프레임을 요청할 수 있다.

> **알려진 공백 (GitHub issue로 추적):** raw **오디오는 아직 orchestrator에
> 입력되지 않는다** — 음성은 ASR 텍스트로만 대체된다. raw shot 오디오를 JUDGE에
> 배선하는 작업은 per-run trace 플래그가 아니라 **저장소 issue**로 추적한다.

## Window

`config/config.yaml → labelling`에서:

- `window_size = 5` — 스텝당 보이는 shot 수 (confirm shot + 이웃 컨텍스트)
- `window_stride = 3` — 스텝당 확정(commit)되는 shot 수 (overlap = size − stride = 2)
- `carry_over = rolling_summary` — 확정 판정들의 러닝 요약. 항상 존재하는
  `global_overview`와 함께 매 스텝 주입됨

## 카테고리는 파라미터화

카테고리 집합은 **하드코딩하지 않는다.** 활성 정책 세트(`config
policy.categories`, 현재 gambling / bad_language / sex)에서 N-길이 리스트로
주입된다. 프롬프트와 코드가 이 리스트를 순회하므로, 카테고리 추가/제거는
코드 변경이 아니라 config/정책 변경이다. N개 카테고리는 shot당 **한 번의
패스**로 판정한다.

## 단계 (Stages)

### LOAD
`all_segments[cursor : cursor + window_size]`에서 `window`를 구성한다. 앞쪽
`window_stride`개 shot이 **confirm** shot(이번 스텝에 commit)이고, 나머지는 이웃
컨텍스트다. 각 shot의 `[t_start, t_end]`와 겹치는 `utterances`를 ASR 텍스트로
병합한다. `clip_blob`에서 confirm shot마다 uniform 프레임 ≤ 5장을 샘플링한다.
디버깅용 run-scoped 스텝 `tool_trace`를 매 스텝 초기화하지만, 이는 라벨이 담는
것이 **아니다** — 라벨의 감사 기록은 JUDGE(아래)에서 설정된다. 도구: storage 읽기,
프레임 샘플러.

### RETRIEVE
- `search_policies` — 활성 카테고리의 정책 노드(scoring 루브릭 / attribute
  definition / decision rule / edge-case)에 대한 hybrid(pgvector dense + BM25)
  검색.
- `find_similar_segments` — 가장 유사한 shot + **그 확정 라벨**(선례 조회; 주된
  일관성 신호).

도구: retrieval 전용.

### JUDGE  *(DERIVE + CHECK 흡수)*
판정은 **confirm shot마다** 이뤄진다. 합성된 정책이 있는 카테고리(ATTRIBUTE
definition **및** DECISION_RULE 트리 보유, 그리고 bootstrap 아님)는 attribute
기반 **SELECT → EXTRACT → DECIDE → REVIEW → STORE** 파이프라인을 실행하고,
나머지(합성 정책 없음, 그리고 bootstrap 중의 *모든* 카테고리)는 holistic 폴백을
유지한다. 프레임(≤5)은 shot당 **한 번만** 샘플링해 모든 호출에서 재사용한다.

```mermaid
flowchart TB
  IN["confirm shot마다"]
  IN --> Q{"attribute def + decision 트리<br/>보유 카테고리?<br/>(그리고 bootstrap 아님)"}

  Q -->|"attribute 기반 카테고리"| SEL["SELECT — 해당 카테고리 전체 대상 호출 1회:<br/>어떤 카테고리 + 어떤 attribute 이름을 라벨링할지"]
  SEL --> EXT["EXTRACT — 선택된 카테고리마다 호출 1회:<br/>선택 attribute → value + evidence (값은 per-value rule과 함께 제시)"]
  EXT --> DEC["DECIDE — 추출값에 결정 트리를 결정적으로 적용<br/>→ score + trajectory (LLM 호출 없음)"]
  DEC --> REV["REVIEW — 카테고리 전체 대상 호출 1회:<br/>적절성 판정; score는 바꿀 수 없음"]
  REV --> STO["STORE — 카테고리당 Label 1개<br/>evidence는 모든 attr 포함 · cited pin(attr-def + rule) · trajectory"]
  REV -->|"needs_change"| RC["rule-change 요청 큐잉<br/>(decision-rule 노드 타깃)"]

  Q -->|"정책 없음 / bootstrap — holistic"| HD["구조화 신호 파생<br/>(예: ASR에 term-level word-list로 욕설 매칭)"]
  HD --> HS["N개 카테고리에 멀티모달 채점 호출 1회<br/>0..5 · rationale · cited pin · evidence"]
  HS --> CK{"선례와 비교"}
  CK -->|"배치(divergent)"| ISS["rationale에 선례 divergence 기록<br/>(자동 재판정 없음)"]

  EXT -.->|"프레임 부족"| EXP["expand_frames → 1회 재추출"]
  HS -.->|"프레임 부족"| EXP
```

**Attribute 기반 파이프라인**(ATTRIBUTE definition + DECISION_RULE 트리 보유
카테고리):
- **SELECT** — attribute 기반 카테고리 전체를 대상으로 한 orchestrator 호출 1회.
  각 카테고리를 attribute **이름만** 보여주고, 어떤 카테고리가 관련되는지와
  카테고리별로 어떤 attribute를 라벨링할지 받는다. 에이전트가 생략한 카테고리/
  attribute는 라벨링하지 **않는다** — 해당 attribute는 empty로 취급되어 트리는
  그 값 없이 실행된다(empty-safe → 보통 default score, 0 = 부재). 파싱은 알려진
  카테고리/attribute 이름만 남긴다.
- **EXTRACT** — 선택된 카테고리마다 호출 1회로 **선택된** attribute만 닫힌 enum /
  ordinal에 맞춰 value + evidence로 해석한다. 각 attribute의 허용값은
  **per-value edge-case rule과 함께** 렌더링된다. 선택된 attribute가 없는
  카테고리는 EXTRACT 호출을 건너뛴다.
- **DECIDE** — 추출값만으로 코드에서 트리를 **결정적으로** 적용(LLM 호출 없음)해
  `score`와 **trajectory**(`{selected, extracted, rule_index, rule_note, score}`)를
  산출한다. trajectory는 REVIEW 프롬프트·rationale·score를 만드는 데 in-process로만
  쓰이며 **저장되지 않는다**.
- **REVIEW** — 카테고리 전체를 대상으로 한 호출 1회로 각 카테고리의 score +
  trajectory를 주입한다. 모델은 적절성을 판정하지만 **score를 바꿀 수 없다**;
  `needs_change`를 표시하면 JUDGE는 그 decision-rule 노드를 타깃으로 **rule-change
  요청을 큐잉**한다 — 큐잉만, 자동 적용 없음.
- **STORE** — 카테고리당 Label 1개. `evidence_attributes`는 정의된 **모든**
  attribute를 포함한다: 선택+추출된 것은 value + evidence를 담고
  (`source=judge/extract`), 선택되지 않은/empty인 것은 EMPTY 값으로 저장된다
  (`value=""`, `evidence=None`, `source=judge/unselected`) — 다운스트림에서
  "고려했으나 empty"를 구분할 수 있게. cited pin = 모든 attr-def 노드 + rule 노드;
  rationale은 매칭 규칙 note다. **라벨별 `tool_trace`는 저장하지 않는다** —
  발화된 규칙은 필요 시 `evidence_attributes`에서 재도출한다(아래 *노드 → segment
  추적* 참고).

**Holistic 폴백**(합성된 정책이 없는 카테고리, 그리고 bootstrap 중의 *모든*
카테고리): 멀티모달 호출 1회가 프레임 + summary + ASR에서 직접 카테고리를 채점하며,
먼저 구조화 term-level 신호를 파생하고 선례 divergence를 **rationale에** 기록한다.

구조화 출력은 **강제하지 않는다**; 모델이 반환한 JSON 유사 텍스트를 관대하게
파싱한다. 일관성 divergence는 자동 보정하지 않고 **기록**된다.
도구: `sample_frames` / `expand_frames`, `search_policies` 결과, 정책 트리,
orchestrator 호출.

### SIDE_FX
JUDGE가 만든 proposal을 shape별로 dispatch한다(여기서만 도달 가능):
- `propose_policy_change` — **항상** human 승인 대기 큐로; 자동 적용되지 않는다.
  attribute 기반 **rule-change 요청**(`node_type=decision_rule`,
  `target_policy_id` = 규칙 노드)과 bootstrap 자유 텍스트 gap 제안
  (`node_type` = attribute / edge_case)을 모두 포함한다.
- `define_structured_attribute` — **bootstrap 전용 직접 upsert**로 구조화
  term-level ATTRIBUTE 노드를 초안 작성(human 게이트 없음; 전체 초안 트리는
  policy-set v1 스냅샷 전에 검토됨).

승인 시 큐잉된 요청은 해당 카테고리의 scoring 루브릭 아래 ATTRIBUTE / EDGE_CASE
노드로 **materialise**된다(`resolve_change_request`). 편집된 attribute나 summary에
대한 콘텐츠 변경 이력(revision log)은 계획된 후속 작업이며 아직 구현되지 않았다.

### COMMIT
각 draft `Label`을 저장(`storage.save_label`)하고 롤링 `carry_over` 요약을
갱신한다. `cursor += window_stride`; `cursor < len(all_segments)`면 LOAD로 루프,
아니면 END.

## 노드 → segment 추적

검토자가 정책 노드를 클릭하면 그 노드를 통해 라벨링된 segment를 볼 수 있다
(`tools.tracking`, 읽기 전용, 저장 없음):
- **attribute 값 → segment**(`segments_for_attribute_value`) — 그 key/value를
  `evidence_attributes`에 담은 라벨을 매칭한다.
- **결정 트리 규칙 → segment**(`segments_for_rule`) — 라벨에 trace를 **저장하지
  않으므로** 발화된 규칙을 **재도출**한다: 카테고리의 각 라벨에 대해 비어 있지 않은
  `evidence_attributes`에서 `values`를 재구성하고, 카테고리의 **현재** 결정 트리를
  공용·DB-free `tools.decision_tree` 모듈(에이전트가 채점에 쓴 바로 그 코드)로 다시
  적용해, 매칭된 규칙의 인덱스가 요청한 인덱스와 같을 때 그 segment를 포함한다.

## Issues 로그

라벨별 rationale 외에, holistic 경로는 점수가 유사 shot의 확정 라벨과 크게
어긋날 때 라벨의 **rationale에** 선례 divergence를 기록한다 — human manager가
group화해 분류(triage)하도록 보존되며 자동 보정되지 않는다. (오디오 입력 부재
같은 더 큰 기능 공백은 per-run trace가 아니라 **저장소 issue**로 추적한다.)

## 출력 계약(Output contract)

판정된 각 shot은 `Label` row를 만든다(`packages/schemas` 참고): `category`,
`score`, `rationale`, `cited_policy_ids`, `evidence_attributes`,
`used_segment_ids`, `confidence`. 라벨의 감사 기록은
**evidence_attributes**(추출된 attribute + evidence + attribute 노드의 version),
정규 `(policy_id, version)` pin인 **cited_policy_ids**(attribute-def 노드 + 규칙
노드, 환각 id는 제거되고 실제 version이 재부착됨), 그리고 **rationale**에 담긴 매칭
규칙 note다. 라벨은 **라벨별 `tool_trace`를 담지 않는다**(컬럼은 유지하되 빈 `[]`로
저장). 발화된 결정 트리 규칙은 필요할 때 `evidence_attributes`에서 재도출한다
(노드 → segment 추적). 이 pin들은 `policy_versions` 히스토리를 통해 사용된 정확한
텍스트로 해석되어 모든 라벨을 재현·감사 가능하게 만든다.
