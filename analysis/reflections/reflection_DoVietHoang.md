# Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Đỗ Việt Hoàng  
**MSSV:** 2A202602882  
**Khóa:** K4 - Track 3B  
**Ngày:** 05/10/2026  

---

## 1. Lecture Mapping (Ánh xạ bài giảng)

| Khái niệm/Kỹ thuật trong bài giảng | Module liên quan | Cách áp dụng trong lab |
|-----------------------------------|------------------|------------------------|
| **Semantic Chunking** | M1 | Gom câu theo ngưỡng cosine similarity (0.85) thay vì cắt cứng theo số token. Giúp giữ nguyên ý nghĩa ngữ nghĩa, tránh tách gãy giữa các ý liên quan. |
| **Hierarchical Chunking (Parent-Child)** | M1 | Parent chunk 2048 tokens để bảo toàn ngữ cảnh rộng, child chunk 256 tokens cho tìm kiếm chính xác. Truy vấn bốc child → trả về parent cho LLM. |
| **Structure-Aware Chunking** | M1 | Parse header Markdown (#, ##, ###), giữ nguyên header trong chunk và lưu `section` vào metadata. Giúp bot biết được chunk thuộc mục nào. |
| **BM25 + Vietnamese Segmentation** | M2 | Dùng `underthesea` tách từ tiếng Việt trước khi cho vào `BM25Okapi`. Xử lý tốt hơn cho từ ghép tiếng Việt so với tách theo khoảng trắng. |
| **Dense Retrieval với BGE-M3** | M2 | Embedding 1024 chiều, đa ngôn ngữ, hỗ trợ cả dense/sparse/colbert. Index vào Qdrant dùng HNSW. |
| **Reciprocal Rank Fusion (RRF)** | M2 | Gộp điểm BM25 và Dense: `score = Σ 1/(k + rank + 1)`. Không cần tune trọng số, ổn định cross-domain. |
| **Cross-Encoder Reranking** | M3 | `BAAI/bge-reranker-v2-m3` chấm điểm (query, doc) cặp. Rerank top-20 → top-3. Đắt nhưng chính xác hơn bi-encoder. |
| **RAGAS Evaluation (4 metrics)** | M4 | Faithfulness, Answer Relevancy, Context Precision, Context Recall. Đánh giá toàn diện: retrieval + generation. |
| **Failure Analysis & Error Tree** | M4 | Phân loại lỗi theo root cause (retrieval vs generation), gợi ý fix cụ thể theo từng module. |
| **Chunk Enrichment (HyQA + Contextual Prepend)** | M5 | Sinh câu hỏi giả định, tóm tắt, prepend ngữ cảnh, trích metadata — gộp 1 lần gọi LLM để tiết kiệm chi phí. |

**Điểm mấu chốt:** Pipeline production không chỉ "ghép các module" mà là thiết kế sao cho output của module trước là input tối ưu cho module sau (ví dụ: chunking có header → enrichment thấy header → reranker ưu tiên chunk có header phiên bản mới).

---

## 2. Difficulties Encountered (Khó khăn gặp phải)

### 2.1. Xử lý văn bản tiếng Việt & PDF scan
- **Vấn đề:** 2/3 file PDF là scan ảnh (không có text layer). `pymupdf`/`pdfplumber` trả về chuỗi rỗng.
- **Ảnh hưởng:** Mất dữ liệu từ 2 tài liệu quan trọng, giảm Context Recall.
- **Giải pháp tạm thời:** Skip có warning, chỉ index 25 file `.md` + 1 PDF có text. Cần OCR (Tesseract/PaddleOCR) cho production thực tế.

### 2.2. Version Conflict trong Retrieval (Vấn đề lớn nhất)
- **Vấn đề:** Corpus chứa cả chính sách cũ (2023) và mới (2024) dùng từ khóa giống nhau. Query ngắn ("nghỉ bao nhiêu ngày phép") bốc cả 2 phiên bản → LLM hallucinate hoặc trả lời sai.
- **Thử fix:** Thêm metadata `version`, `status` nhưng reranker vẫn chưa đủ mạnh để phân biệt.
- **Bài học:** Cần **Query Rewrite** (bổ sung "chính sách mới nhất") hoặc **Temporal Reranking** (ưu tiên doc mới) ở tầng retrieval, không chỉ dựa vào reranker.

### 2.3. Tối ưu Latency vs Quality Trade-off
- **Cross-Encoder reranking:** 145ms p50 (35% tổng latency). Nếu bỏ qua → giảm 1/3 thời gian nhưng Faithfulness giảm ~0.15.
- **Enrichment M5:** Combined single-call tốn ~2-3s/indexing (chạy offline). Separate mode tốn gấp 4 lần.
- **Qdrant HNSW params:** `ef_search=128` cho recall cao nhưng chậm hơn. Cần benchmark thực tế hơn.

### 2.4. RAGAS Evaluation Stability
- **Vấn đề:** Chạy 2 lần cùng config có khi ra điểm chênh 0.02-0.03 do LLM judge (GPT-4o-mini) non-deterministic.
- **Workaround:** Set `temperature=0`, seed cố định, chạy 3 lần lấy trung bình. Nhưng tốn token và thời gian.

### 2.5. M5 Enrichment - LLM Fallback
- **Vấn đề:** Không có `OPENAI_API_KEY` → enrichment fallback về chunk gốc. Production pipeline chạy được nhưng quality thấp.
- **Ảnh hưởng:** HyQA questions không sinh ra → multi-hop query (câu hỏi cần ghép 2 doc) bị miss.

---

## 3. Action Plan (Kế hoạch cải tiến)

### 3.1. Ngắn hạn (1-2 tuần) — Fix các lỗi Bottom-5
| Hành động | Module | Expected Impact |
|-----------|--------|-----------------|
| Thêm metadata `doc_status: "active"/"deprecated"` + `effective_date` | M1 + M5 | Fix #1, #2, #3 (version conflict) |
| Implement Query Rewrite: prepend "theo chính sách mới nhất hiện hành" cho query không có năm | M2 (pre-retrieval) | Fix #1, #2, #3 |
| Thêm Chain-of-Thought prompt cho LLM: "Hãy tính toán từng bước trước khi trả lời con số" | Pipeline (generation) | Fix #5 (numeric hallucination) |
| Sinh HyQA questions cho cross-doc queries (thử việc + phép năm) | M5 | Fix #4 (multi-hop recall) |

### 3.2. Trung hạn (1 tháng) — Architecture Improvements
- **Parent-Child Retrieval hoàn chỉnh:** Hiện tại chỉ retrieve child → trả parent. Cần implement: retrieve child (top-20) → group by parent → rerank parent → top-3 parent cho LLM. Giảm noise, tăng recall.
- **Temporal Reranking:** Thêm feature `recency_score = 1 / (1 + days_since_effective)` vào RRF score.
- **Hybrid Search v2:** Thử BGE-M3 sparse vector (built-in) thay BM25, so sánh latency/quality.

### 3.3. Dài hạn (Production-ready) — Robustness & Observability
- **OCR Pipeline:** Tích hợp PaddleOCR cho PDF scan → không mất data.
- **Evaluation Automation:** CI/CD chạy RAGAS nightly, alert nếu metric giảm > 0.05.
- **A/B Test Framework:** So sánh production vs baseline trên traffic thật, không chỉ test set 20 câu.
- **Guardrails:** Output validator (regex cho số ngày phép, version) trước khi trả user.
- **Cost Optimization:** Cache embedding cho query lặp, batch reranker inference, quantization (int8) cho cross-encoder.

---

## Kết luận cá nhân

Lab 18 giúp hiểu rõ **RAG production không chỉ là "có retriever + có LLM"** mà là chuỗi các quyết định thiết kế có tính trade-off rõ ràng:

1. **Chunking strategy** quyết định ceiling của recall — chunk sai thì search tốt cũng vô ích.
2. **Hybrid Search + RRF** là baseline vững chắc, nhưng **reranker** mới là chìa khóa precision.
3. **Enrichment (M5)** trông "phụ" nhưng thực tế là force multiplier cho retrieval — HyQA questions biến multi-hop từ "may mắn" thành "systematic".
4. **Evaluation (RAGAS + Failure Analysis)** bắt buộc phải có từ đầu — không đo lường được thì không cải thiện được.

Điều gây bất ngờ nhất: **Cross-Encoder reranking chiếm 35% latency nhưng contribution to quality cực lớn** (+0.23 Faithfulness). Đây là trade-off đáng giá cho use-case HR policy (accuracy > speed), nhưng nếu làm chatbot real-time cần cân nhắc distillation hoặc bi-encoder reranker.

**Next step:** Sẽ implement Query Rewrite + Temporal Reranking trước, sau đó benchmark Parent-Child retrieval hoàn chỉnh. Mục tiêu: đẩy Context Recall > 0.95 và Faithfulness > 0.90 trên test set.