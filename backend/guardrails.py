def classify_query(q: str):
    q = q.lower()

    # -----------------------------
    # Explicit comparisons / other universities
    # -----------------------------
    if any(x in q for x in [
        "cu",
        "chandigarh university",
        "amity",
        "vit",
        "iit",
        "nit",
        "better than",
        "vs",
        "compare"
    ]):
        return "comparison"

    # -----------------------------
    # Abusive / offensive / hostile language
    # -----------------------------
    if any(x in q for x in [
        "fuck",
        "shit",
        "idiot",
        "stupid",
        "worst university",
        "fraud",
        "scam",
        "fake",
        "useless",
        "waste",
        "bad university",
        "lpu is bad",
        "lpu worst"
    ]):
        return "negative"

    # -----------------------------
    # Admission / course related (implicitly LPU)
    # -----------------------------
    if any(x in q for x in [
        "admission",
        "apply",
        "join",
        "course",
        "department",
        "program",
        "eligibility",
        "fee",
        "fees",
        "scholarship",
        "placement",
        "hostel",
        "campus",
        "agriculture",
        "engineering",
        "management",
        "mba",
        "btech",
        "law",
        "pharmacy",
        "design"
    ]):
        return "normal"

    # -----------------------------
    # Mixed / unclear / broken language
    # (treat politely as normal)
    # -----------------------------
    if len(q.split()) < 6:
        return "normal"

    return "unrelated"
