"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION, OPENAI_API_KEY, GEMINI_API_KEY, USE_GEMINI, GEMINI_MODEL, OPENAI_MODEL


# ─── Rate Limiter for Free Tier ──────────────────────────

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
        return float(match.group(1)) + 1
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


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    system_prompt = "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if contexts:
            context_str = "\n\n".join(contexts)
            user_prompt = f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"
            answer = _call_llm(system_prompt, user_prompt)
            if not answer:
                answer = contexts[0]
        else:
            answer = "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if all(results.get(m, 0) == 0 for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        print("\n💡 Lưu ý: Điểm baseline hiển thị 0.00 là bình thường khi chưa hoàn thiện M2 (Dense Search) và M4 (Eval).")
        print("   Sau khi bạn implement xong các module, hãy chạy 'python main.py' để tự động cập nhật baseline thật và so sánh.")
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
