## 개요

4주차 RAG를 수동 채점했지만 실무에서는 매번 채점할 수 없고, 정답률 하나로는 검색/생성 중 어디가 문제인지 모릅니다. 이번 주에 Ragas 기반 자동·체계적 평가를 학습합니다.

1. Golden Dataset: 변경의 회귀를 객관적으로 검증하는 기준
2. LLM-as-a-Judge: LLM을 판정자로 활용해 정답·품질을 자동 채점하는 방식
3. Ragas 프레임워크: LLM-as-a-Judge 원리를 기반으로 검색·생성 단계를 분리 진단하는 메트릭 세트

### RAG 평가가 왜 어려운가

RAG는 두 단계이므로 답변이 틀렸을 때 원인을 쪼개서 봐야 합니다.

```
질문 → [Retrieval] → [Generation] → 답변
```

정답률 하나로는 어디가 문제인지(청킹? 검색? Re-ranker? 프롬프트?) 알 수 없음. Ragas는 네 메트릭으로 단계를 분리 진단.

| 메트릭 | 단계 | 정의 | ground_truth 필요? |
|--------|-----|-----|----------------|
| Context Recall | Retrieval | 정답에 필요한 내용이 검색에 들어왔는가 | 필요 |
| Context Precision | Retrieval | 관련 청크가 상위에 있는가 | 필요 |
| Faithfulness | Generation | 답변이 컨텍스트로 뒷받침되는가 (환각 체크) | 불필요 |
| Answer Relevancy | Generation | 답변이 질문에 답하고 있는가 | 불필요 |
| Answer Correctness | End-to-End | 답변이 ground_truth와 의미적으로 일치 | 필요 |

이번 주는 4주차 Basic/Advanced RAG를 재사용해 같은 시스템을 새 평가 렌즈로 측정합니다. "Advanced가 좋아 보였다"를 "어느 메트릭이 얼마나 좋아졌다"로 바꾸는 것이 목표.

### 이전 주차와의 연결

| 주차 | 이번 주와의 연결 |
|------|---------------|
| 2주차 | 실습 심화 C에서 2주차 정답률과 Ragas Ans Correctness 비교 |
| 3주차 | `evidence_text` → `ground_truth_contexts` (문자열 → 리스트) |
| 4주차 | 파이프라인·Golden Dataset 재사용, `expected_answer`는 `ground_truth` 출발점 |

## 조사 시 참고 원칙

공식 문서·논문 2개 이상 교차 참조. 본인 언어로 작성, 원문 출처 표기.

## 필수 조사 항목

### 1. Golden Dataset

#### 1-1 정의: RAG 시스템 변경 시 성능을 객관적으로 검증하기 위한 기준 데이터셋
- 해당 데이터셋이 없을 시 개선되었는지 아닌지 판단 불가능하며, 실험 결과 재현이 불가능

#### 1-2 필수 스키마: 
  - `user_input`: 사용자 질문
  - `ground_truth`: 질문에 대한 정답
  - `ground_truth_contexts`: LLM이 답을 생성할 때 참조해야할 정보들
  - `response`: LLM이 생성한 답
  - `retrieved_contexts`: RAG 파이프라인에서 LLM이 답을 생성할 때 참조한 정보


**v0.1**
```python
{
  "question": str,
  "answer": str,
  "contexts": List[str],
  "ground_truths": List[str]
}
```

**v0.2+**
```python
{
  "question": str,
  "ground_truth": str,
  "ground_truth_contexts": List[str],
  "answer": str,               
  "contexts": List[str]        
}
```

#### 1-3 권장 규모:

| 단계 |	개수	| 범위 |
| --- | --- | --- |
| 초기 |	30~50 |	주요 유즈케이스 중심으로만 구성 |
| 중간 |	100~300 | 다양한 도메인 지식을 활용해 구성 |
| 대규모 |	500+ | 희귀 케이스까지 모두 반영 |

#### 1-4 좋은 Dataset 조건
- 실제 측정된 데이터 기반의 다양한 시나리오 제공
- 충분한 수의 샘플 제공
- 데이터 변화에 따른 지속적인 업데이트

#### 1-5 `ground_truth_contexts` 수동 어노테이션 이유:
- `Context Recall` 측정 시 반드시 읽어야 하는 `context`의 정의가 필요하기 때문

---

### 2. 평가의 필요성과 LLM-as-a-Judge


#### 2-1. 왜 체계적 평가가 필요한가

**소프트웨어 테스트와의 차이**
- 기존 소프트웨어의 경우 출력이 결정적이고 정답/오답으로 테스트가 가능
- LLM의 경우 확률적이고 문맥에 따른 모호성이 존재함. Sweet Spot을 정의

**회귀 검증(regression test) 개념과 Golden Dataset의 연결**
- 회귀 검증은 프롬프트 수정이나 파이프라인 수정 시 영향을 확인하는 과정을 의미
- Golden Dataset를 기준으로 평가의 객관적 근거로 활용


**자동 평가 vs 사람 평가 스펙트럼과 trade-off**

| 구분 |	사람 평가	| 자동 평가 |
| -- | -- | -- |
| 정밀도 |	문맥의 미묘한 차이 |
| 비용 | 인건비가 지속적으로 발생 | API 호출 비용만 발생 |
| 속도	| 느림 | 빠름 |
| 일관성	| 낮음 | 높음 |
| 확장성 |	불가능 | 자동화 가능 |

#### 2-2. LLM-as-a-Judge

텍스트 기반 자동 채점에 LLM을 판정자(judge)로 활용하는 방식. Ragas의 Faithfulness, Answer Correctness 등 핵심 메트릭이 이 원리 위에 구축되어 있으므로, Ragas 메트릭을 이해하기 전에 먼저 익혀야 하는 선행 개념입니다.

- 등장 배경: BLEU/ROUGE 등 n-gram 자동 평가의 한계, 사람 평가의 비용·시간 한계
- 작동 원리: 루브릭과 판정 기준을 프롬프트로 제공 → LLM이 답변을 읽고 점수·근거 출력
- 사람 평가와의 일치율: GPT-4 Judge가 사람 평가와 얼마나 일치하는가 (권장, 필수 아님)
- 한계와 신뢰성: 프롬프트·모델에 따른 결과 변동, 비결정성, 호출 비용

**루브릭 설계 원칙**

| **좋은 루브릭** | **나쁜 루브릭** |
| -- | -- |
| 측정 가능한 기준, 점수를 제시 | 추상적인 표현 |
| 순차적 지시사항 | 모호한 지시사항 |
| 점수를 부여하는 근거 제시 | 점수 부여의 근거나 설명이 없음 |
| Few-shot 예시 포함 | 예시 없음 |

**좋은 루브릭** 

```
### 점수 기준
- 1점: 답변이 관련성 없음
- 3점: 주요 사실관계는 맞으나, 오류가 있거나 핵심 내용이 누락됨
- 5점: 답변의 모든 내용이 컨텍스트에 완벽히 근거하며, 질문에 대해 정확하고 간결하게 답함

### 출력 형식
- 근거:
- 점수:
```

**나쁜 루브릭** 

```
답변을 1점에서 5점 사이로 점수
```



**4대 메트릭 구분**

| 메트릭 |	방식 |
| -- | -- |
| Context Recall |	규칙 기반 |
| Context Precision |	LLM Judge |
| Faithfulness |	LLM Judge |
| Answer Relevancy |	LLM Judge |

---

### 3. Ragas 4대 메트릭 (+ Answer Correctness)

#### 3-1. 검색 단계

| 구분 | Context Recall | Context Precision |
|------|---------------|------------------|
| 정의 | 정답에 필요한 정보가 retrieval 결과에 포함되어 있는가 | 관련 있는 문서를 상위로 검색해 왔는가 |
| 계산 방식 | (찾은 정답 정보) / (전체 정답 정보) | 검색된 청크가 질문과 연관이 있는지 확인 후 precision을 계산 뒤 청크들의 평균을 계산 |
| 낮을 때 의심할 점 | 청크 크기가 너무 작음, 임베딩 모델의 성능 부족, 질의 시 k가 작음 | 질문과 관련 없는 문서가 상위에 노출됨, 검색 알고리즘의 순위 산정 문제 |
| 개선 기법 | chunk size 조정, query 재작성 | Re-rank 활용 |

#### 3-2. 생성 단계

| 구분 | Faithfulness | Answer Relevancy |
|------|-------------|------------------|
| 정의 | 생성된 답변이 검색된 컨텍스트만을 근거로 작성되었는가 | 답변이 사용자의 질문 의도에 얼마나 직접적으로 부합하는가 |
| 계산 방식 | (주어진 컨텍스트를 기반으로하는 주장의 갯수) / (생성한 답변에 포함된 주장의 총 갯수) | 정답과 LLM이 생성한 답의 임베딩 값의 cosine similarity 계산 |
| 낮을 때 의심할 점 | 모델이 학습되어 있던 지식을 활용 | 답변이 너무 장황하거나 무관한 답변도 포함됨 |
| 개선 기법 | 시스템 프롬프트에 컨텍스트만을 확인할 것을 강제 | Few-shot 기반으로 질문 의도를 다시 파악하도록 프롬프트 수정 |

#### 3-3. End-to-End

**Answer Correctness 정의**
- 모델의 답변이 `ground_truth`와 얼마나 의미적으로 같고, 동시에 사실적으로 맞는지 평가

**Answer Correctness 계산 방식**
- F1-score 계산

**Answer Correctness만으로 부족한 이유**
- `ground_truth` 품질에 의존
  - `ground_truth` 정답이 여러개이나 하나만 존재할 경우
  - `ground_truth`가 틀린 경우
  - `ground_truth`의 정답이 추상적이고 객관적 평가가 어려운 경우


#### 3-4. 메트릭 간 관계

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---------|---------------|-----|------|
| 정답 청크 자체를 검색이 놓침 | Context Recall | Retrieval 실패 | VectorDB 와 검색 범위 개선 |
| 정답 청크는 있지만 8~10위로 밀림 | Context Precision | Ranking 문제 | Re-ranker 활용 |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness | Generation 문제 | Prompt 지시 개선 |
| LLM이 질문을 잘못 이해 | Answer Relevancy | 이해도 문제 | 질문 재작성 |
| 답은 맞는데 장황 | (해당 없음) | | |

> 장황함·간결성 같은 주관적 품질은 Ragas 기본 메트릭으로 잡히지 않음. 도메인 임계값을 정하거나 Ragas 커스텀 메트릭(`MetricWithLLM` 상속)으로 보완. 실습 심화 A가 그 예.


## 선택 심화 조사 항목

### 심화 1. Ragas 메트릭 내부 LLM 동작

각 메트릭이 LLM에게 어떤 판단을 시키는지 공식 문서·소스로 확인. 예: Faithfulness의 claim 단위 대조, Context Precision의 청크별 관련성 판단. 호출 비용이 큰/작은 메트릭 구분.


### 심화 2. Ragas 커스텀 메트릭 설계

`MetricWithLLM` 상속으로 도메인 특화 메트릭을 만드는 방식. 프롬프트 템플릿, 출력 파서, `_single_turn_ascore()` 메서드. 실습 심화 A의 YearAccuracy가 이 방식을 사용.



## 실습 과제 예측

실습 전 아래 가설을 세우고, 실습 후 실제 결과와 비교하여 제출 README에 포함.

1. 4주차 정답률(사람)과 Ragas Ans Correctness(자동)의 일치 정도? 어느 문항에서 차이 예상? -
2. Basic/Advanced의 네 메트릭 중 가장 크게 벌어질 메트릭은? 이유는? - Context Precision, Re-rank 사용으로 Rank 변환이 클 것으로 예상
3. 년도 혼동 문제는 어느 Ragas 메트릭에 주로 반영될 것인가? -
4. Advanced에서 Faithfulness가 오히려 낮아질 시나리오가 있을까? -


# 5주차 실습 과제: RAG 시스템 정량 평가 — Ragas

## 배경

4주차 RAG를 수동 채점으로 평가했지만 한계가 명확합니다.

- 확장성 부족: 문제가 많아지면 채점 불가
- 진단 불가: 실패 원인이 검색인지 생성인지 구분 안 됨
- 재현성 낮음: 채점 기준이 매번 달라짐

이번 과제는 4주차 RAG를 재사용해 Ragas 기반 자동·정량 평가 파이프라인을 구축합니다.

1. Golden Dataset 확장: `ground_truth_contexts` 수동 어노테이션
2. Ragas 자동 평가: 4대 메트릭 + Answer Correctness (Basic/Advanced)
3. Basic vs Advanced 비교 분석, 실패 케이스 Deep Dive

### 이전 주차와의 연결

| 주차 | 이번 주와의 연결 |
|------|---------------|
| 2주차 | 실습 심화 C에서 2주차 정답률과 Ragas Ans Correctness 비교 |
| 3주차 | `evidence_text` → `ground_truth_contexts` (문자열 → 리스트) |
| 4주차 | 파이프라인·Golden Dataset 재사용, `expected_answer`는 `ground_truth`의 출발점 |

## 데이터

- 4주차와 동일한 의료급여 PDF (`data/2025 알기 쉬운 의료급여제도.pdf`, `data/2026 알기 쉬운 의료급여제도.pdf`)
- 4주차 벡터 저장소(`source_year` 메타데이터 포함) 재사용 권장

### Golden Dataset 확장

4주차 Dataset(20문제)에 두 필드를 추가합니다.

| 필드 | 의미 | 준비 방법 | 이전 주차 대응 |
|------|------|---------|------------|
| `ground_truth` | 기대 답변 | `expected_answer`를 완전한 문장으로 정제 | 4주차 `expected_answer` 확장 |
| `ground_truth_contexts` | 정답 근거 청크 (리스트) | PDF에서 근거 문단 발췌 | 3주차 `evidence_text` (문자열 → 리스트) |

#### `ground_truth`는 완전한 문장으로

Ragas Answer Correctness는 의미 유사도(임베딩) + 사실 일치도(LLM)의 가중 평균입니다. RAG 답변이 완전한 문장이므로 `ground_truth`도 같은 형태여야 유사도가 제대로 측정됩니다.

| 형태 | 예시 | 체감 |
|------|------|------|
| 나쁨 | `"1,000원"` | 유사도 낮아 저평가 |
| 좋음 | `"2025년 의료급여 1종 수급권자의 외래 본인부담금은 1,000원입니다."` | 유사도·사실 일치도 모두 높음 |
| 장황 | 2~3줄 장문 | RAG 답변보다 길어져 유사도 하락 |

> 정제 원칙(예: "년도 + 대상 + 조건 + 값 순으로 한 문장")을 README.md에 기록.

#### 확장 Dataset 예시 (JSONL)

```jsonl
{"question": "2025년 의료급여 1종 수급권자의 외래 본인부담금은?", "ground_truth": "2025년 의료급여 1종 수급권자의 외래 본인부담금은 1,000원입니다.", "ground_truth_contexts": ["1종 수급권자 외래 본인부담금은 1,000원이며..."], "difficulty": "easy", "source_year": "2025"}
{"question": "2025년 대비 2026년에 달라진 본인부담률은?", "ground_truth": "2026년에는 외래 본인부담금이 1,000원에서 1,500원으로 인상되었습니다.", "ground_truth_contexts": ["2025년 본인부담금 1,000원...", "2026년 본인부담금 1,500원..."], "difficulty": "cross-year", "source_year": "2025+2026"}
```

> 파일에는 `question` / `ground_truth` / `ground_truth_contexts`로 저장. Ragas v0.2+ 입력 시 `user_input` / `reference` / `reference_contexts`로 매핑 (Step 1-2).

#### 주의사항

- `ground_truth_contexts`는 리스트 (cross-year 문항은 두 년도 청크 포함)
- 벡터 저장소 청크와 일치할 필요 없음. PDF 원본 문단을 의미 단위로 발췌
- 최소 10문제, 권장 15문제. 난이도별(`easy`/`medium`/`hard`/`cross-year`) 고른 분포
- 파일명: `golden_dataset_v2.jsonl`

## 실습 구조

### Step 1: Ragas 자동 평가 환경 구축

Ragas 0.4.x 기준. v0.2에서 스키마가 바뀌었으니 `pip show ragas`로 0.2 이상 확인. `question`/`answer`/`contexts` 예시는 대부분 v0.1이라 최신 버전과 호환되지 않음.

#### 1-1. 설치 및 API 키

Ragas 0.2 이상과 LangChain 연동 패키지(Anthropic, OpenAI)를 설치하고 `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` 환경변수를 설정합니다.

평가용 LLM / 임베딩 선택:

- 생성 LLM이 GPT-4o면 평가용은 Claude Sonnet 계열 권장
- 임베딩: OpenAI `text-embedding-3-small` 또는 4주차 한국어 임베딩 재사용
- Ragas 내부 프롬프트는 기본 영어 → `adapt_prompts(language="korean", llm=evaluator_llm)`로 한국어 전환

#### 1-2. 평가 데이터 준비 (v0.2+ 스키마)

각 질문에 대해 Basic/Advanced RAG를 실행하여 `retrieved_contexts`(검색 청크)와 `response`(답변)를 수집. `SingleTurnSample` + `EvaluationDataset`으로 구성합니다.

| JSONL 필드 | `SingleTurnSample` 필드 |
|-----------|----------------------|
| `question` | `user_input` |
| (RAG 실행 결과) | `response` |
| (RAG 실행 결과) | `retrieved_contexts` |
| `ground_truth` | `reference` |
| `ground_truth_contexts` | `reference_contexts` |

JSONL을 한 줄씩 읽어 RAG에 질문을 넣고, 파일 필드와 실행 결과를 매핑 표에 따라 `SingleTurnSample`로 묶어 파이프라인별 `EvaluationDataset`을 만듭니다.

> RAG의 `invoke()` 반환 형태는 구현마다 다름. "검색 청크 리스트"와 "답변 문자열"을 분리해 꺼낼 수 있어야 함.

#### 1-3. 평가용 LLM / 임베딩 래핑

평가용 LLM(ChatAnthropic `claude-sonnet-4-5`, temperature=0)과 임베딩(`text-embedding-3-small`)을 `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`로 래핑. 메트릭 다섯 개(`ContextRecall`, `LLMContextPrecisionWithReference`, `Faithfulness`, `ResponseRelevancy`, `AnswerCorrectness`) 인스턴스를 만들고 `adapt_prompts(language="korean", llm=evaluator_llm)` → `set_prompts(**adapted)`로 한국어 프롬프트 적용.

> 첫 실행 시 번역 결과를 로그로 확인.

### Step 2: Ragas 4대 메트릭 + Answer Correctness 측정

#### 2-1. 메트릭 실행

`evaluate()`에 `dataset`, `metrics`, `llm`, `embeddings` 주입 → Basic/Advanced 각각 실행. `to_pandas()` 변환 후 `basic_ragas_scores.csv`, `advanced_ragas_scores.csv`로 저장.

메트릭 선택 이유:

- `ContextRecall()`: `reference` 기준 검색 재현율
- `LLMContextPrecisionWithReference()`: `reference` 있는 경우 표준 Precision (v0.2+ 권장)
- `Faithfulness()`: 환각 체크 (`reference` 불필요)
- `ResponseRelevancy()`: 구 `answer_relevancy`
- `AnswerCorrectness()`: End-to-end 정확도

> 소문자 함수형은 deprecation 대상. 클래스형 사용.

비용: 20문항 × 5메트릭 × 2파이프라인 = 200~300회 LLM 호출. 대략 3~8 USD. 파일럿 5문항으로 먼저 검증 권장.

#### 2-2. 결과 기록

전체 평균

| 메트릭 | Basic | Advanced | 변화 |
|--------|------|---------|-----|
| Context Recall | | | |
| Context Precision | | | |
| Faithfulness | | | |
| Answer Relevancy | | | |
| Answer Correctness | | | |

대표 문항별

| 질문 ID | 난이도 | source_year | Ctx Recall (B/A) | Ctx Precision (B/A) | Faithfulness (B/A) | Ans Relevancy (B/A) | Ans Correctness (B/A) |
|---------|--------|------------|-----------------|-------------------|------------------|-------------------|---------------------|
| q01 | easy | 2025 | / | / | / | / | / |
| q02 | medium | 2026 | / | / | / | / | / |
| q03 | cross-year | 2025+2026 | / | / | / | / | / |
| ... | | | | | | | |

> B/A = Basic/Advanced.

#### 2-3. 4주차 수동 채점 vs Ragas 비교

| 질문 ID | 4주차 판정 | Ragas Ans Correctness | 일치 | 불일치 원인 |
|---------|-----------|----------------------|-----|-----------|
| q01 | 정답 | 0.87 | 일치 | |
| q02 | 오답 | 0.42 | 일치 | |
| q03 | 정답 | 0.58 | 불일치 | 표현 차이 |
| ... | | | | |

### Step 3: Basic RAG vs Advanced RAG 비교 분석

#### 3-1. 다차원 비교

| 구분 | Ragas 5메트릭에서 개선/악화된 차원 |
|------|-----------------------------|
| 개선 | |
| 악화 | |

#### 3-2. 년도 혼동 재진단

- 년도 혼동 문항에서 어느 메트릭이 가장 민감한가?
- Ragas 기본 메트릭으로 충분한가, Year Accuracy 커스텀 메트릭이 필요한가? (→ 심화 A)

#### 3-3. 인사이트 정리

2~3문단 작성:

- 4주차 "Advanced가 낫다" 결론은 어디까지 유효한가?
- 도메인 임계값(예: Faithfulness ≥ 0.9)으로 보면 프로덕션 가능한가?
- 개선 우선순위 메트릭과 근거는?

### Step 4: 실패 케이스 Deep Dive

메트릭 점수 뒤의 실제 현상 분석. 문항 2개 필수 (Case C 선택).

#### 4-1. 문항 선별

| 케이스 | 선별 기준 | 필수 | 예시 |
|-------|----------|-----|------|
| Case A | Advanced가 Basic보다 악화된 문항 | 필수 | Re-ranker가 정답 청크를 뒤로 미룸 |
| Case B | 년도 혼동 발생 (Context 맞는데 답변 년도 틀림) | 필수 | 2025/2026 청크 혼합 |
| Case C | Ragas 메트릭 간 충돌 (예: Faithfulness 높은데 Answer Correctness 낮음) | 선택 | 환각 없지만 ground_truth와 의미 차이 |

> Case A가 없으면 "Basic·Advanced 모두 오답인 가장 어려운 문항"으로 대체. 선별 기준 명시.

#### 4-2. 케이스별 분석 양식

```
### Case [A/B/C]: q[XX]

질문: {question}
참고 정답: {ground_truth}

[검색된 청크 — Basic RAG]
1. (청크 전문, source_year 포함)
2. ...

[검색된 청크 — Advanced RAG]
1. ...
2. ...

[생성된 답변]
- Basic: {답변}
- Advanced: {답변}

[메트릭 점수는 Step 2 결과표에서 해당 행 인용]

[원인 분석]
- 검색/생성/둘 다 중 어디가 문제?
- 어느 메트릭이 가장 잘 드러냈나?
- 조치 (청킹/프롬프트/Re-ranker/메타데이터 필터)?
```

#### 4-3. 공통 교훈

3~5개 bullet:

- 어떤 질문 유형에서 메트릭이 실제 품질과 어긋났나
- Ragas가 놓치는 실패 유형
- 4주차 수동과 5주차 자동 중 어느 쪽이 더 엄격/관대한가

---

## 선택 심화 과제

필수 아님. 하나 이상 수행 시 제출.

### 심화 A: Custom Metric — YearAccuracy

Ragas 기본 메트릭은 년도 혼동을 직접 포착하지 못함. `YearAccuracy` 커스텀 메트릭을 Ragas에 통합.

요구사항

- `MetricWithLLM` 등 베이스 클래스 상속
- 입력: `question`, `response`, `source_year`
- 출력: 0~1 점수
- `evaluate()`에서 기존 메트릭과 함께 사용 가능
- Basic/Advanced 측정 후 년도 혼동 진단 기여도 평가

참고: [Ragas — Write your own Metric](https://docs.ragas.io/en/stable/howtos/customizations/metrics/write_your_own_metric/)

### 심화 B: 평가 → 개선 → 재평가

Step 3-3의 "개선 우선순위 메트릭"을 한 번 개선하고 재평가.

절차

1. 개선할 메트릭 1개 선택 (예: Faithfulness)
2. 가설 수립 (예: "컨텍스트 외 정보 금지 지시 강화 → Faithfulness 0.72 → 0.85")
3. 변경 적용 (프롬프트·청킹·Re-ranker 중 하나만, 범위 최소)
4. Ragas 재실행
5. 변경 전/후 전체 메트릭 표 비교

| 메트릭 | 변경 전 | 변경 후 | 변화 |
|--------|--------|--------|------|
| Context Recall | | | |
| Context Precision | | | |
| Faithfulness | | | |
| Answer Relevancy | | | |
| Answer Correctness | | | |

- 가설 vs 결과 일치 여부
- 부작용 발생 시 원인

### 심화 C: 2주차~5주차 누적 개선 비교

"프롬프트만 → RAG → Advanced RAG → 평가 체계" 누적 가치 시각화.

조건

- 2주차 2024 PDF, 4/5주차 2025/2026 PDF라 정답값이 바뀔 수 있음. 교집합 문항 식별 선행
- 교집합 3문항 미만이면 평균 수준 비교로 대체 (같은 문항 비교가 아님을 명시)

| 방식 | 시점 | 비교 축 |
|------|-----|--------|
| 2주차 Zero-shot | 2주차 | 사람 정답률 |
| 2주차 최고 (CoT/Self-Consistency) | 2주차 | 사람 정답률 |
| 4주차 Basic RAG | 4주차 | 사람 정답률 |
| 4주차 Advanced RAG | 4주차 | 사람 정답률 |
| 5주차 Advanced RAG | 이번 주 | Ragas Ans Correctness |

- 척도 차이 주석 명시
- 누적 개선 곡선 bullet/표, 가장 큰 도약 구간 해석

### 심화 D: 건강보험심사평가원 데이터 확장 평가

4주차에서 쓴 의료급여 PDF 외에 건강보험심사평가원 e-book(https://www.hira.or.kr/ra/ebook/list.do?pgmid=HIRAA030402000000)에서 관련 자료를 추가 인덱싱하고 Golden Dataset을 확장해 평가 범위를 넓힙니다.

요구사항

- 건강보험심사평가원 e-book에서 1~2개 PDF 선택 (의료급여·건강보험 관련 주제)
- 4주차 벡터 저장소에 추가 인덱싱 (`source_document` 등 메타데이터 포함)
- Golden Dataset에 새 자료 기반 문제 5개 이상 추가 (`ground_truth`, `ground_truth_contexts` 포함)
- 확장 전/후 Ragas 5메트릭 비교
  - Context Recall/Precision 변화
  - Faithfulness·Answer Correctness 유지 여부
  - 자료 간 혼동·충돌 발생 여부

기록

- 추가한 자료 목록과 선택 이유
- 확장 전/후 메트릭 비교 표
- 새로 발견된 실패 유형 (있다면)

## 구현 요구사항

### 필수

1. Step 1~4 모두 구현 및 결과 기록
2. `golden_dataset_v2.jsonl` 작성 (`ground_truth_contexts` 포함, 최소 10문제 / 권장 15문제)
3. Ragas 5개 메트릭 측정 (Basic/Advanced)
4. 4주차 수동 채점과 Ragas Ans Correctness 일치도 분석
5. Basic vs Advanced 비교, 년도 혼동 재진단, 인사이트 정리
6. 실패 케이스 2~3건 분석 + 공통 교훈

### 권장

- 4주차 파이프라인 재사용
- 평가용 LLM: GPT-4o 또는 Claude Sonnet 이상 (한국어 품질)
- 평가용 LLM은 생성용과 다른 모델 패밀리

### 선택 심화

- 심화 A: YearAccuracy Custom Metric
- 심화 B: 평가 → 개선 → 재평가
- 심화 C: 2주차~5주차 누적 비교
- 심화 D: 건강보험심사평가원 데이터 확장 평가

### 금지

- ChatGPT/Claude 웹 UI 사용
- `ground_truth_contexts`를 LLM으로 자동 생성 (수동 어노테이션이 학습 포인트)

## 제출물

- 브랜치 `week5/<GithubID>` 생성 후 PR

필수 파일 (`week-5/<GithubID>/`):
- `golden_dataset_v2.jsonl`
- `README.md` (이론 + 실습 결과)
- 관련 코드 (Ragas 평가 스크립트)
- 평가 결과 로그 (JSON/CSV)

## README.md 필수 포함 항목

1. 프레임워크·모델(생성용 LLM, 평가용 LLM, 임베딩)·Ragas 버전·실행 환경
2. Golden Dataset 확장 전략 (`ground_truth_contexts` 발췌 원칙, `ground_truth` 정제 원칙, cross-year 처리)
3. Ragas 평가 파이프라인 (평가용 LLM 선택 이유, 데이터셋 구성)
4. Step 2 결과 (5메트릭 표, 4주차 수동 채점 비교)
5. Step 3 결과 (Basic vs Advanced 비교, 년도 혼동 재진단, 인사이트)
6. Step 4 결과 (실패 케이스, 공통 교훈)
7. 이론 과제 답변
8. 가설 vs 실제 결과 비교
9. (선택) 심화 과제 결과

## 참고 자료

Ragas
- [Ragas 공식 문서](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/explodinggradients/ragas)
- [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)
- [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)
- [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217)

평가 프레임워크
- [DeepEval](https://docs.confident-ai.com/)
- [Promptfoo](https://www.promptfoo.dev/docs/intro/)
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation)

## 힌트

- Ragas 첫 사용이라면 공식 Quickstart(https://docs.ragas.io/en/stable/getstarted/)로 스키마를 익힌 뒤 시작. 블로그 튜토리얼 대부분이 v0.1 기준이라 최신 버전과 맞지 않음.
- 4주차 RAG 체인이 답변 문자열만 반환한다면 `retrieved_contexts`도 함께 꺼내도록 수정 필요. LangChain LCEL이면 `RunnablePassthrough.assign()`으로 중간 검색 결과를 병행 전달. AI 도움으로 10~20줄 수준.
- `ground_truth_contexts` 품질이 Context Recall 신뢰도를 결정. PDF에서 2~5문장 단위로 발췌.
- 파일럿 5~10문항으로 스키마·비용 먼저 검증.
- `pip show ragas`로 버전 확인 (v0.1/v0.2+ API 차이).
- Faithfulness·Answer Relevancy 높은데 Answer Correctness만 낮으면 `ground_truth` 품질 의심.
- Context Recall 높고 Answer Correctness 낮음 → 생성 단계. Faithfulness 낮으면 환각, 높으면 프롬프트/포맷 문제.
- Context Precision은 순서 민감. Recall과 함께 봐야 "검색이 뭘 못했나" 드러남.
- 년도 혼동은 Ragas 기본 메트릭으로 직접 잡히지 않음 (Year Accuracy 커스텀 메트릭이 필요한 이유, 심화 A 참고).
- Step 1~2로 숫자 확보 후 Step 3~4 진행. 심화는 마지막에.