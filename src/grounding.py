REFUSAL_ANSWER = "I cannot answer based on the provided document."


def retrieval_is_sufficient(top_score: float, threshold: float) -> bool:
    return top_score >= threshold


def combine_confidence(retrieval_score: float, grounding_score: float) -> float:
    score = min(retrieval_score, grounding_score)
    return round(max(0.0, min(1.0, score)), 2)