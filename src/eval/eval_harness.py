"""
Evaluation harness for multi-agent system.
"""
def record_ungrounded(query: str, filename: str = "human_review.txt"):
    """Record ungrounded queries for expert review."""
    with open(filename, "a") as f:
        f.write(f"Ungrounded: {query}\n")
