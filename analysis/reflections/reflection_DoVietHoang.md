# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Do Viet Hoang  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 2026-10-05

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 nhóm câu theo cosine similarity, tạo ít chunks hơn basic chunking; bảo toàn ngữ nghĩa câu liên quan thay vì cắt ngang theo ký tự" |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | "Parent 2048 chars + Child 256 chars; child có `parent_id` link đến parent; retrieve child (precision) → return parent (context) — best practice cho production RAG" |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | "Parse markdown headers (#, ##, ###); giữ header trong chunk text + `section` trong metadata; không cắt giữa table/code block" |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF: score = Σ 1/(k + rank + 1); kết hợp điểm xếp hạng lexical (BM25 với Vietnamese segmentation underthesea) và semantic (bge-m3 dense vectors); `segment_vietnamese()` replace '_' bằng space để token khớp" |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Load bge-reranker-v2-m3 via sentence_transformers.CrossEncoder; rerank top-20 → top-3; sorted by `rerank_score` descending; doc 'nghỉ phép' ranked cao hơn 'VPN' cho query về nghỉ phép" |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "4 metrics: faithfulness, answer_relevancy, context_precision, context_recall; dùng HuggingfaceEmbeddings (bge-m3) + LLM (Gemini/OpenAI); wrap try/except trả về 0 nếu fail; `failure_analysis()` map worst metric → Diagnostic Tree (diagnosis + suggested_fix)" |
| Contextual embeddings / Enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | "Combined mode: 1 LLM call/chunk → summary + questions + context + metadata; fallback extractive khi không có API key; `enrich_chunks()` trả về `EnrichedChunk` với `enriched_text` khác `original_text`" |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `Ignoring wrong pointing object 11 0 (offset 0)` từ pypdf khi đọc PDF scan (BCTC.pdf, Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf) — PDF không có text layer, cần OCR
  - `NaN` trong `reports/ragas_report.json` aggregate scores — do RAGAS evaluation fail khi không có LLM API key (OPENAI_API_KEY/GEMINI_API_KEY)
  - Rate limiting 4.5s/call trong `_rate_limit()` của M5 enrichment khiến pipeline chậm (~8 phút cho 107 chunks) ngay cả khi dùng fallback

- **Nguyên nhân gốc rễ & Cách debug:**
  - **PDF scan:** `pypdf.PdfReader.extract_text()` trả về rỗng → code đã handle bằng cách skip và print warning (đúng thiết kế)
  - **NaN scores:** `evaluate_ragas()` cần LLM để compute metrics; không có API key → try/except bắt exception → trả về dict với 0.0; nhưng JSON serialize `NaN` từ numpy float → cần `_to_serializable()` convert
  - **Rate limit fallback:** `_call_llm()` vẫn gọi `_rate_limit()` trước khi check API key; fix: check API key trước khi rate limit, hoặc bỏ rate limit cho fallback path

- **Kiến thức còn thiếu & Cách khắc phục:**
  - **RAGAS internals:** Cách 4 metrics tính toán chi tiết (faithfulness dùng NLI, context_recall cần ground truth context) → đọc docs RAGAS + paper
  - **Vietnamese BM25:** underthesea tokenize dùng "_" cho từ ghép → phải replace "_" → " " trước khi BM25Okapi tokenize bằng split(" ") → hiểu rõ pipeline tokenization
  - **Cross-encoder vs Bi-encoder:** Cross-encoder (reranker) chậm hơn nhưng chính xác hơn vì attention đầy đủ giữa query-doc pair → benchmark latency vs accuracy tradeoff
  - **Production deployment:** Qdrant persistence, batch indexing, caching embeddings, async pipeline → tìm hiểu Qdrant production config + monitoring

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Hệ thống Q&A chính sách nhân sự nội bộ (HR Policy RAG)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Basic RAG — paragraph chunking (500 chars) + dense search (bge-m3) + top-3 context → LLM answer; không rerank, không evaluation
- **Vấn đề / Bottlenecks đang gặp:** Retrieval miss context khi query dùng từ đồng nghĩa; hallucination khi context không đủ; không có metrics định lượng để cải thiện; latency cao do không cache embedding

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** **Hierarchical** (parent 2048, child 256) — vì tài liệu HR có cấu trúc phân cấp rõ (chương → điều → khoản); retrieve child tìm chính xác đoạn → return parent cho context đầy đủ
2. **Search retrieval:** **Hybrid BM25 + Dense + RRF** — BM25 bắt từ khóa chính xác (mã policy, số điều); Dense bắt ý nghĩa (semantic); RRF fuse không cần tune weight
3. **Reranking:** **Có dùng bge-reranker-v2-m3** — rerank top-20 → top-3; latency ~200ms acceptable cho HR internal tool; tăng precision đáng kể cho câu hỏi tra cứu cụ thể
4. **Evaluation:** **RAGAS 4 metrics + custom metrics** — faithfulness (giảm hallucination), context_recall (đảm bảo retrieval đủ), thêm metric custom: "policy_accuracy" (check số ngày, điều khoản đúng không)
5. **Enrichment:** **Combined mode (_enrich_single_call)** — 1 call/chunk: summary (giảm noise embedding), HyQA questions (bridge vocabulary gap), contextual prepend (giảm retrieval failure 49%), auto metadata (filter category: leave/salary/it/safety)

#### 3. Timeline triển khai
- **Tuần 1:** Implement hierarchical chunking + hybrid search + reranker; index 26 documents HR policies; test retrieval trên 20 test questions
- **Tuần 2:** Setup RAGAS evaluation pipeline; chạy baseline vs production; điền failure analysis; optimize chunk size / top-k / rerank threshold
- **Tuần 3:** Thêm enrichment combined mode; so sánh retrieval quality trước/sau enrichment; setup monitoring (latency, token usage, retrieval metrics)
- **Tuần 4:** Deploy staging; A/B test với users thật; collect feedback; iterate chunking/search params; document runbook