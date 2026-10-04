from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY, GEMINI_API_KEY, USE_GEMINI, GEMINI_MODEL, OPENAI_MODEL


# ─── Rate Limiter for Free Tier (shared with m5_enrichment) ─────────────────

_last_call_time = 0.0
_min_interval = 4.5  # seconds between calls (15 RPM = 4 sec, add buffer)


def _rate_limit():
    """Enforce minimum interval between API calls."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_call_time = time.time()


def _parse_retry_delay(error_msg: str) -> float:
    """Extract retry delay from error message (e.g., 'Please retry in 22.36s')."""
    match = re.search(r'retry in\s+([\d.]+)\s*s', error_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1  # add 1 second buffer
    return 0.0


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    """Call LLM (Gemini or OpenAI) with rate limiting and retry logic."""
    max_retries = 5
    base_delay = 5.0
    
    for attempt in range(max_retries):
        _rate_limit()
        
        if USE_GEMINI and GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    )
                )
                return response.text.strip()
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    retry_delay = _parse_retry_delay(error_msg)
                    if retry_delay == 0:
                        retry_delay = base_delay * (2 ** attempt)
                    print(f"  ⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    continue
                print(f"  ⚠️  Gemini call failed: {e}")
        
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower():
                    retry_delay = base_delay * (2 ** attempt)
                    print(f"  ⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    continue
                print(f"  ⚠️  OpenAI call failed: {e}")
        
        break
    
    return ""


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    system_prompt = "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"
    if contexts:
        context_str = "\n\n".join(contexts)
        user_prompt = f"Context:\n{context_str}\n\nCâu hỏi: {query}"
        answer = _call_llm(system_prompt, user_prompt)
        if not answer:
            answer = contexts[0]
    else:
        answer = "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
