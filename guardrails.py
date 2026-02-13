def classify_query(query: str):
    query_lower = query.lower()

    if "compare" in query_lower or "better than" in query_lower:
        return "comparison"

    if "bad" in query_lower or "worst" in query_lower or "fraud" in query_lower:
        return "negative"

    if "lpu" in query_lower:
        return "lpu"

    return "irrelevant"
