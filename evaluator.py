"""
Evaluator Agent: menilai jawaban dari sisi accuracy, efficiency, dan hallucination.
Versi sederhana (heuristik) -- untuk versi lengkap bisa diganti library RAGAS/DeepEval.
"""
import time


def evaluate_response(query: str, result: dict, elapsed_seconds: float) -> dict:
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Heuristik sederhana: seberapa banyak kata jawaban yang juga muncul di sumber (proxy faithfulness)
    source_text = " ".join([str(s) for s in sources]).lower()
    answer_words = set(answer.lower().split())
    overlap = len(answer_words & set(source_text.split())) if answer_words else 0
    faithfulness_score = round(overlap / max(len(answer_words), 1), 2)

    return {
        "latency_seconds": round(elapsed_seconds, 2),
        "num_sources_used": len(sources),
        "faithfulness_score": faithfulness_score,  # mendekati 1 = lebih grounded ke sumber
        "answer_length_words": len(answer.split()),
    }


def timed_route(route_fn, query: str):
    """Wrapper untuk mengukur latency sekaligus memanggil orchestrator."""
    start = time.time()
    result = route_fn(query)
    elapsed = time.time() - start
    scores = evaluate_response(query, result, elapsed)
    return result, scores
