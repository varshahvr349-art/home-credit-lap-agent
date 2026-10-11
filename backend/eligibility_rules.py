"""Deterministic checks for the assignment's explicit eligibility rules."""

MAX_LOAN_AMOUNT = 7_500_000


def disqualification_reason(state: dict) -> str | None:
    """Return the first failed rule, or None while the customer still passes."""
    if state.get("property_type") == "agricultural":
        return "Agricultural property is not eligible for this LAP offer."
    if state.get("original_documents") is False:
        return "Original property documents must be available for verification."
    if state.get("income_mode") == "cash":
        return "Income must be received through a bank for this offer."
    tenure = state.get("tenure")
    if tenure is not None and not 3 <= tenure <= 15:
        return "The requested tenure must be between 3 and 15 years, inclusive."
    return None
