# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Do Viet Hoang  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A (no API key) | N/A (no API key) | N/A |
| Answer Relevancy | N/A (no API key) | N/A (no API key) | N/A |
| Context Precision | N/A (no API key) | N/A (no API key) | N/A |
| Context Recall | N/A (no API key) | N/A (no API key) | N/A |

> **Note:** Scores are N/A because RAGAS evaluation requires an LLM API key (OpenAI or Gemini). The pipeline ran without API key using extractive fallbacks for enrichment and returned empty answers for LLM generation. Once you obtain an API key, run `python main.py` to generate valid scores and update this table.

---

## Bottom-5 Failures (Template — Fill After Running With API Key)

### #1
- **Question:** [Copy from test_set.json]
- **Expected:** [ground_truth from test_set.json]
- **Got:** [answer from pipeline]
- **Worst metric:** [faithfulness / answer_relevancy / context_precision / context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? → 
- **Root cause:** [Based on Diagnostic Tree mapping]
- **Suggested fix:** [From failure_analysis() suggested_fix]

### #2
- **Question:** [Copy from test_set.json]
- **Expected:** [ground_truth from test_set.json]
- **Got:** [answer from pipeline]
- **Worst metric:** [faithfulness / answer_relevancy / context_precision / context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? → 
- **Root cause:** [Based on Diagnostic Tree mapping]
- **Suggested fix:** [From failure_analysis() suggested_fix]

### #3
- **Question:** [Copy from test_set.json]
- **Expected:** [ground_truth from test_set.json]
- **Got:** [answer from pipeline]
- **Worst metric:** [faithfulness / answer_relevancy / context_precision / context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? → 
- **Root cause:** [Based on Diagnostic Tree mapping]
- **Suggested fix:** [From failure_analysis() suggested_fix]

### #4
- **Question:** [Copy from test_set.json]
- **Expected:** [ground_truth from test_set.json]
- **Got:** [answer from pipeline]
- **Worst metric:** [faithfulness / answer_relevancy / context_precision / context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? → 
- **Root cause:** [Based on Diagnostic Tree mapping]
- **Suggested fix:** [From failure_analysis() suggested_fix]

### #5
- **Question:** [Copy from test_set.json]
- **Expected:** [ground_truth from test_set.json]
- **Got:** [answer from pipeline]
- **Worst metric:** [faithfulness / answer_relevancy / context_precision / context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? → 
- **Root cause:** [Based on Diagnostic Tree mapping]
- **Suggested fix:** [From failure_analysis() suggested_fix]

---

## Test Set Reference (20 Questions)

| # | Question | Ground Truth |
|---|----------|--------------|
| 1 | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | 3 ngày làm việc có lương |
| 2 | Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên? | 200.000.000 VNĐ/năm |
| 3 | Phụ cấp ăn trưa hàng tháng là bao nhiêu? | 1.000.000 VNĐ/tháng |
| 4 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 15 ngày (v2024), cũ 12 ngày (v2023) |
| 5 | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | Từ 3 năm, mỗi 3 năm +1 ngày (v2024) |
| 6 | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 12 ký tự (v2.0), cũ 8 ký tự (v1.0) |
| 7 | Bao lâu phải đổi mật khẩu một lần? | 120 ngày (v2.0), cũ 90 ngày (v1.0) |
| 8 | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | Bắt buộc (v2.0), cũ không yêu cầu (v1.0) |
| 9 | Nhân viên thử việc có được nghỉ phép năm không? | KHÔNG |
| 10 | Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không? | KHÔNG, báo cáo 1h |
| 11 | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | KHÔNG |
| 12 | Senior 9 năm thâm niên: phép năm bao nhiêu & lương khoảng nào? | 18 ngày, 20-35 triệu |
| 13 | Mua laptop 30 triệu cho nhân viên mới: ai phê duyệt, cần gì CNTT? | Director phê duyệt, xác nhận cấu hình CNTT, 3 báo giá |
| 14 | Tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng: hoàn trả bao nhiêu? | 100% = 25.000.000 VNĐ |
| 15 | Mentor và buddy có thể là cùng người? Quản lý trực tiếp làm mentor được không? | KHÔNG cho cả hai |
| 16 | Mua thiết bị 55 triệu cần ai phê duyệt? | CEO |
| 17 | Tạm ứng 15 triệu, thanh toán sau 20 ngày: phạt bao nhiêu? | ~50.000 VNĐ cho 5 ngày quá hạn |
| 18 | Lương thử việc Junior cao nhất bao nhiêu? | 17.000.000 VNĐ/tháng (85% x 20M) |
| 19 | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | Dữ liệu Bí mật (cấp 3) |
| 20 | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | CEO |

---

## Case Study (cho presentation)

**Question chọn phân tích:** [Chọn 1 câu hỏi từ bottom-5 để phân tích sâu]

**Error Tree walkthrough:**
1. Output đúng? → [Yes/No + evidence]
2. Context đúng? → [Yes/No + retrieved chunks]  
3. Query rewrite OK? → [Yes/No + any query expansion]
4. Fix ở bước: [Chunking / Search / Rerank / Prompt / LLM]

**Nếu có thêm 1 giờ, sẽ optimize:**
- [ ] Tune chunk size / overlap cho domain HR policies
- [ ] Thêm keyword filter (category: leave/salary/it) vào hybrid search
- [ ] Điều chỉnh RRF parameter k hoặc weight BM25 vs Dense
- [ ] Thử reranker model khác (FlashRank cho latency thấp)
- [ ] Cải thiện prompt template: few-shot, chain-of-thought, citation requirement
- [ ] Thêm query rewriting / Hypothetical Document Embeddings (HyDE)