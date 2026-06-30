# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Hoàng Thanh Chiến  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    |
    v (~16.18ms P95)
[Presidio PII Scan]
    | block if: VN_CCCD / VN_PHONE / EMAIL / PHONE_NUMBER detected
    | action:   return safe refusal + anonymize detected PII
    v (~0.28ms P95)
[NeMo Input Rail + deterministic local guard]
    | block if: off-topic / jailbreak / prompt injection / PII request
    | action:   return refusal message
    v
[RAG Pipeline (Day 18)]
    | M1 Chunk -> M2 Search -> M3 Rerank -> GPT-4o-mini
    v
[NeMo Output Rail]
    | flag if: PII in response / sensitive content
    | action:  replace with safe response
    v
User Response
```

---

## Latency Budget

Kết quả lấy từ `reports/guard_results.json`, sinh bởi `measure_p95_latency()`.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---|
| Presidio PII | 8.49 | 16.18 | 16.18 | <20ms |
| NeMo Input Rail | 0.03 | 0.28 | 0.28 | <300ms |
| RAG Pipeline | N/A | N/A | N/A | <2000ms |
| NeMo Output Rail | N/A | N/A | N/A | <300ms |
| **Total Guard** | 8.53 | **16.21** | 16.21 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Tổng guard P95 đạt 16.21ms, thấp hơn nhiều so với budget 500ms. Presidio là phần tốn thời gian chính trong guard stack hiện tại; đã cache `setup_presidio()` để tránh khởi tạo analyzer lặp lại. Trong production, nếu bật NeMo với LLM thật cho mọi request thì NeMo sẽ trở thành bottleneck, nên nên giữ rule-based precheck cho các pattern rõ ràng và chỉ gọi LLM guard khi cần.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Judge Consistency Gate
  run: python src/phase_b_judge.py
  env:
    MAX_POSITION_BIAS_RATE: 0.30

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải >= 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; print(measure_p95_latency(['nghỉ phép năm 2024'], n_runs=1))"
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, inspect bottom failing samples |
| RAGAS answer relevancy | < 0.65 | Review query rewriting, retrieval prompts, and answer grounding |
| Adversarial block rate | < 80% | Review new attack patterns and update input rails |
| Guard P95 latency | > 600ms | Profile Presidio/NeMo, cache reusable engines, scale guard service |
| PII detected count | spike >10/hour | Security alert and audit affected sessions |
| Position bias rate | > 30% | Enforce swap-and-average or calibrate judge prompt |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---:|
| RAGAS avg_score (50q) | 0.751 |
| Worst metric | answer_relevancy (~0.641 overall) |
| Dominant failure distribution | factual |
| Dominant failure metric | faithfulness |
| Cohen's kappa | 1.000 |
| Position bias rate | 0.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 16.21 ms |
| Guard latency budget | Passed (<500ms) |

---

## Nhận xét & Cải tiến

Guardrail stack hoạt động tốt trong lab: Presidio bắt được CCCD, số điện thoại và email; input guard chặn đủ 20/20 adversarial cases; latency của guard stack vẫn nằm rất xa dưới ngưỡng 500ms. Điểm yếu chính của RAG nằm ở chất lượng câu trả lời, đặc biệt là `answer_relevancy` và các lỗi `faithfulness` trong nhóm factual/multi-hop, nghĩa là retrieval có thể tìm được context nhưng câu trả lời vẫn chưa bám sát hoặc chưa tổng hợp đúng. Nếu deploy production thật, tôi sẽ thêm regression set cố định cho các câu multi-hop khó, log top-k context để debug recall, và dùng judge nhiều mẫu hơn thay vì một sample để Cohen's kappa có ý nghĩa hơn. Với guardrails, tôi sẽ giữ lớp rule-based nhanh cho pattern rõ ràng, sau đó mới gọi NeMo/LLM guard cho case mơ hồ để cân bằng giữa an toàn và latency.
