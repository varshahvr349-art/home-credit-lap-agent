"""Small deterministic conversation flow used by typed and spoken replies."""

import re
import copy
import logging

from backend.eligibility_rules import MAX_LOAN_AMOUNT, disqualification_reason
from backend.llm_service import ERROR_MESSAGES, LLMServiceError

logger = logging.getLogger(__name__)

CHECKLIST = [
    ("property_type", "What type of property is it: residential, commercial, or agricultural?"),
    ("ownership", "Is the property individually owned or jointly owned?"),
    ("original_documents", "Are the original property documents available for verification, even if safely stored at home?"),
    ("loan_amount", "How much would you like to borrow? The maximum under this offer is ₹75 lakh."),
    ("occupation_income", "What is your occupation—salaried or self-employed—and is your income received through a bank or in cash?"),
    ("market_value", "What is the approximate current market value of the property?"),
    ("tenure", "What loan tenure would you like, in years? The allowed range is 3 to 15 years."),
]


def new_conversation() -> tuple[dict, str]:
    state = {
        "customer_name": "Ravi",
        "stage": "identity",
        "identity_verified": False,
        "availability": None,
        "driving_status": None,
        "callback_status": "not_requested",
        "answers": {},
        "status": "active",
        "outcome": None,
        "reason": None,
    }
    return state, "Hello, may I speak with Ravi? This is a simulated call."


def _finish(state: dict, outcome: str, reason: str, reply: str) -> tuple[str, dict]:
    state.update(status="completed", outcome=outcome, reason=reason)
    return reply, state


def _amount_rupees(text: str) -> int | None:
    cleaned = text.lower().replace(",", "").replace("₹", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac| lakhs| lacs)?", cleaned)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "").strip()
    factor = 10_000_000 if unit in {"crore", "cr"} else 100_000 if unit in {"lakh", "lac", "lakhs", "lacs"} else 1
    amount = round(number * factor)
    # A bare number such as 50 in this question is naturally expressed in lakh.
    if not unit and amount <= 1000:
        amount *= 100_000
    return amount


def _parse(field: str, text: str):
    value = text.lower().strip()
    if field == "property_type":
        for kind in ("agricultural", "commercial", "residential"):
            if kind in value:
                return kind
    elif field == "ownership":
        if "joint" in value or "shared" in value:
            return "joint"
        if "individual" in value or "single" in value or "only me" in value:
            return "individual"
    elif field == "original_documents":
        if any(word in value for word in ("photocopy", "photocopies", "photo copy", "copies only", "no original")):
            return False
        if any(word in value for word in ("yes", "available", "at home", "safe", "with me")):
            return True
    elif field == "loan_amount":
        return _amount_rupees(value)
    elif field == "occupation":
        if "salar" in value:
            return "salaried"
        if "self" in value or "business" in value or "entrepreneur" in value:
            return "self-employed"
    elif field == "income_mode":
        if "cash" in value:
            return "cash"
        if any(word in value for word in ("bank", "account", "salary transfer", "deposited")):
            return "bank"
    elif field == "market_value":
        return _amount_rupees(value)
    elif field == "tenure":
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group())
    return None


def _answered(field: str, answers: dict) -> bool:
    if field == "occupation_income":
        return "occupation" in answers and "income_mode" in answers
    return field in answers


def _extract_answers(text: str, pending: str | None) -> dict:
    """Extract clear facts from a reply, even when several arrive together."""
    lower = text.lower()
    found = {}
    for field in ("property_type", "ownership", "original_documents", "occupation", "income_mode"):
        parsed = _parse(field, text)
        if parsed is not None:
            found[field] = parsed

    loan_match = re.search(
        r"(?:loan(?: amount)?|borrow(?:ing)?|need|want|require|request(?:ed|ing)?)"
        r"[^\d]{0,24}(\d+(?:\.\d+)?)\s*(crore|cr|lakhs?|lacs?)?", lower
    )
    if loan_match:
        found["loan_amount"] = _amount_rupees(loan_match.group(0))
    elif pending == "loan_amount" and not re.search(r"market value|worth|valued|\d+\s*(?:years?|yrs?)", lower):
        found["loan_amount"] = _amount_rupees(text)

    market_match = re.search(
        r"(?:market value|property value|worth|valued(?: at)?)\D{0,24}"
        r"(\d+(?:\.\d+)?)\s*(crore|cr|lakhs?|lacs?)?", lower
    )
    if market_match:
        found["market_value"] = _amount_rupees(market_match.group(0))
    elif pending == "market_value" and not re.search(r"\d+\s*(?:years?|yrs?)", lower):
        found["market_value"] = _amount_rupees(text)

    tenure_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", lower)
    if tenure_match and ("tenure" in lower or pending == "tenure"):
        found["tenure"] = float(tenure_match.group(1))
    elif pending == "tenure" and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", lower):
        found["tenure"] = float(lower)
    return {key: value for key, value in found.items() if value is not None}


def _loan_transfer_intent(text: str) -> bool:
    lower = text.lower()
    negates_existing_loan = any(phrase in lower for phrase in ("no existing loan", "don't have a loan", "do not have a loan", "no loan"))
    asks_emi_reduction = any(phrase in lower for phrase in ("reduce my emi", "reduce emi", "lower my emi", "emi reduction"))
    if negates_existing_loan and not asks_emi_reduction:
        return False
    return asks_emi_reduction or any(phrase in lower for phrase in (
        "already have a loan", "already have loan", "existing loan on",
        "loan against this property already", "have a loan against this property",
        "current loan against this property", "loan on this property",
    ))


def _callback_time_from_text(text: str) -> str | None:
    """Return a clear relative or named callback time from customer wording."""
    patterns = (
        r"\b(?:in|after)\s+\d+\s*(?:minutes?|mins?|hours?|hrs?)\b",
        r"\b(?:tomorrow|today|tonight|this\s+(?:morning|afternoon|evening))"
        r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))?\b",
        r"\bafter\s+(?:lunch|dinner|work|the\s+meeting)\b",
        r"\b(?:at\s+)?\d{1,2}(?::\d{2})\s*(?:a\.?m\.?|p\.?m\.?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0).strip(" .,!?").lower())
    return None


def _requests_callback(text: str) -> bool:
    lower = text.lower()
    if any(phrase in lower for phrase in (
        "busy", "not available", "not convenient", "call back", "callback",
        "call me later", "contact me later", "reach me later", "speak later",
    )):
        return True
    return bool(re.search(r"\bcall\s+(?:me\s+)?(?:back|after|in|tomorrow|later|at)\b", lower))


def _explicit_callback_request(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in ("call back", "callback", "call me later")) or bool(
        re.search(r"\bcall\s+(?:me\s+)?(?:back|after|in|tomorrow|later|at)\b", lower)
    )


def _is_call_purpose_question(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(
        r"\bwhy\s+(?:did|are|have)\s+you\s+call(?:ed|ing)?\b|"
        r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:purpose|reason)\s+(?:for|of)\s+(?:your\s+)?call\b|"
        r"\bwhat\s+is\s+this\s+call\s+about\b",
        lower,
    ))


def _callback_takes_priority(text: str, analysis: dict | None) -> bool:
    analysis = analysis or {}
    explicit_request = _explicit_callback_request(text)
    if _is_call_purpose_question(text) and not explicit_request:
        return False
    if analysis.get("availability") == "available" and not explicit_request and not analysis.get("callback_time"):
        return False
    llm_callback = analysis.get("intent") == "callback" and (
        analysis.get("availability") == "unavailable"
        or bool(analysis.get("callback_time"))
        or _requests_callback(text)
    )
    return _requests_callback(text) or explicit_request or llm_callback


def _looks_like_question(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.endswith("?") or bool(re.match(
        r"^(?:who|what|why|when|where|which|how|can|could|would|do|does|is|are|will|may)\b",
        normalized,
    ))


def _fallback_analysis(state: dict, text: str) -> dict | None:
    """Recognize only clear, rule-backed text when Groq's analysis is invalid."""
    lower = text.lower().strip()
    if _callback_takes_priority(text, None):
        return {
            "intent": "callback",
            "availability": "unavailable" if any(word in lower for word in ("busy", "driving", "not available", "not convenient")) else None,
            "callback_time": _callback_time_from_text(text),
        }
    if _looks_like_question(text):
        return {
            "intent": "question",
            "identity_confirmation": True if any(phrase in lower for phrase in ("i am ravi", "i'm ravi", "this is ravi", "yes, speaking")) else None,
        }

    stage = state.get("stage")
    if stage == "identity":
        if any(phrase in lower for phrase in ("wrong person", "not ravi", "not me", "no, this isn't")):
            return {"intent": "identity", "identity_confirmation": False}
        if any(phrase in lower for phrase in ("yes", "speaking", "that's me", "this is ravi", "i am ravi", "i'm ravi", "correct")):
            return {"intent": "identity", "identity_confirmation": True}
    if stage == "availability":
        if any(phrase in lower for phrase in ("you can talk now", "can talk now", "talk now", "free now", "available", "convenient", "go ahead", "yes")):
            return {"intent": "answer", "availability": "available"}
    if stage == "interest":
        if lower in {"yes", "yes please", "interested", "tell me more"}:
            return {"intent": "answer", "interest_status": "interested"}
        if lower in {"no", "not interested", "no thanks", "decline"}:
            return {"intent": "answer", "interest_status": "not_interested"}
    if stage == "checklist" and _extract_answers(text, _next_checklist_field(state)):
        return {"intent": "answer"}
    if stage == "existing_loan" and (_loan_transfer_intent(text) or lower in {"no", "no existing loan", "none"}):
        return {"intent": "existing_property_loan" if _loan_transfer_intent(text) else "answer"}
    if stage == "amount_limit" and lower in {"yes", "yes please", "proceed", "okay", "ok", "no", "no thanks"}:
        return {"intent": "answer"}
    return None


def _callback_ack_time(callback_time: str) -> str:
    """Make a stored relative time sound natural in the acknowledgment."""
    after_minutes = re.fullmatch(r"after\s+(\d+)\s+(minutes?|mins?|hours?|hrs?)", callback_time)
    if after_minutes:
        return f"in {after_minutes.group(1)} {after_minutes.group(2)}"
    return callback_time


def _finish_callback(state: dict, callback_time: str) -> tuple[str, dict]:
    state["answers"]["callback_time"] = callback_time
    state["callback_status"] = "requested"
    state.pop("callback_return_stage", None)
    natural_time = _callback_ack_time(callback_time)
    reply = f"Certainly. I'll note your request for a callback {natural_time}. Thank you for your time. Goodbye."
    return _finish(state, "callback_requested", "Customer requested a callback.", reply)


def _process_deterministic_response(state: dict, text: str, analysis: dict | None = None) -> tuple[str, dict]:
    """Update conversation state once and return the next spoken agent turn."""
    if state.get("status") != "active":
        return "This simulated conversation has ended. Start a new session to continue.", state

    answer = text.lower().strip()
    stage = state.get("stage")

    # Callback intent is a diversion from every active step, including identity,
    # offer-interest, and checklist questions. No LAP details are disclosed here.
    if stage != "callback_time" and _callback_takes_priority(text, analysis):
        callback_time = _callback_time_from_text(text) or (analysis or {}).get("callback_time")
        if callback_time:
            return _finish_callback(state, callback_time)
        state["callback_return_stage"] = stage
        state["stage"] = "callback_time"
        state["callback_status"] = "pending_time"
        return "Certainly. What time would be convenient for a callback?", state

    # Driving is a safety interruption, not a callback request. A customer must
    # explicitly ask for a callback (or provide a time) before one is recorded.
    parked_now = bool(re.search(r"\bparked\b", answer))
    hands_free = bool(re.search(r"\bhands?[- ]?free\b", answer))
    driving_now = bool(re.search(r"\bdriving\b|\bbehind the wheel\b", answer))
    if stage == "driving_check" and _callback_time_from_text(text):
        return _finish_callback(state, _callback_time_from_text(text))
    if stage == "driving_check" and answer in {"yes", "yes please", "that would be good", "please do"}:
        state["callback_status"] = "pending_time"
        state["stage"] = "callback_time"
        state["callback_return_stage"] = state.pop("driving_return_stage", "availability")
        return "Certainly. What time would be convenient for a callback?", state
    if stage == "driving_check" and answer in {"no", "no thanks", "not now"}:
        return "Understood. Please let me know when you're parked or able to speak safely.", state
    if stage == "driving_check" and parked_now:
        state["driving_status"] = "parked"
        state["availability"] = "available"
        resume_stage = state.pop("driving_return_stage", "availability")
        if state.get("identity_verified"):
            state["stage"] = "interest"
            return "Thank you for letting me know. You may be eligible for a Loan Against Property offer of up to â‚¹75 lakh, subject to checks and final approval. Would you be interested in learning more?", state
        state["stage"] = "identity" if resume_stage in {"identity", "greeting"} else resume_stage
        return "Thank you. May I confirm I’m speaking with Ravi?", state
    if driving_now and not parked_now and not hands_free:
        if stage == "identity" and re.search(r"\b(yes|yeah|yep|speaking|this is ravi|i am ravi|i'm ravi)\b", answer):
            state["identity_verified"] = True
        if stage != "driving_check":
            state["driving_return_stage"] = stage
        state["driving_status"] = "driving"
        state["stage"] = "driving_check"
        state["callback_status"] = "not_requested"
        return "I understand, Ravi. Since you're driving, I don't want to distract you. Would you prefer a callback later?", state
    if stage == "driving_check" and hands_free:
        state["driving_status"] = "hands_free"
        resume_stage = state.pop("driving_return_stage", "availability")
        if state.get("identity_verified"):
            state["availability"] = "available"
            state["stage"] = "interest"
            return "Thank you for confirming you can talk hands-free. You may be eligible for a Loan Against Property offer of up to â‚¹75 lakh, subject to checks and final approval. Would you be interested in learning more?", state
        state["stage"] = "identity" if resume_stage in {"identity", "greeting"} else resume_stage
        return "Thank you. May I confirm I’m speaking with Ravi?", state
    if driving_now and hands_free:
        state["driving_status"] = "hands_free"
        if stage == "identity" and re.search(r"\b(yes|yeah|yep|speaking|this is ravi|i am ravi|i'm ravi)\b", answer):
            state["identity_verified"] = True
        if state.get("identity_verified"):
            state["availability"] = "available"
            state["stage"] = "interest"
            return "Thank you for confirming you can talk hands-free. You may be eligible for a Loan Against Property offer of up to â‚¹75 lakh, subject to checks and final approval. Would you be interested in learning more?", state
        return "Thank you. May I confirm I’m speaking with Ravi?", state

    if stage == "identity":
        confirmation = (analysis or {}).get("identity_confirmation")
        if confirmation is False or (confirmation is None and any(word in answer for word in ("no", "wrong person", "not ravi", "not me"))):
            return _finish(state, "IDENTITY_NOT_VERIFIED", "Intended customer not verified.", "I understand. I won't share any offer details. Thank you, and goodbye.")
        confirmed_by_words = any(phrase in answer for phrase in (
            "yes", "speaking", "that's me", "this is ravi", "i am ravi", "i'm ravi", "correct",
        ))
        if confirmation is True or (confirmation is None and confirmed_by_words):
            state["identity_verified"] = True
            state["stage"] = "availability"
            return "Hello, Ravi. I'm calling on behalf of Home Credit about a Loan Against Property offer. Is this a convenient time to talk?", state
        if (analysis or {}).get("intent") == "question":
            return "I'm a simulated Home Credit representative calling about a Loan Against Property offer. May I speak with Ravi?", state
        if not state.get("identity_verified"):
            return "Sorry, I couldn't confirm that. Are you Ravi?", state
        state["stage"] = "availability"
        return "Is this a convenient time to talk?", state

    if stage == "availability":
        if not state.get("identity_verified"):
            state["stage"] = "identity"
            return "Before I continue, may I confirm that you are Ravi?", state
        availability = (analysis or {}).get("availability")
        if availability == "unavailable":
            state["availability"] = "unavailable"
            state["callback_status"] = "pending_time"
            state["callback_return_stage"] = "availability"
            state["stage"] = "callback_time"
            return "Certainly. What time would be convenient for a callback?", state
        if availability != "available" and not any(word in answer for word in (
            "yes", "sure", "convenient", "go ahead", "okay", "ok", "free now", "available",
            "you can talk now", "can talk now", "talk now",
        )):
            return "Would you like to continue now, or should we arrange a callback?", state
        state["availability"] = "available"
        state["stage"] = "interest"
        return "You may be eligible for a Loan Against Property offer of up to ₹75 lakh, subject to checks and final approval. Would you be interested in learning more?", state

    if stage == "callback_time":
        callback_time = _callback_time_from_text(text) or (analysis or {}).get("callback_time")
        if callback_time:
            return _finish_callback(state, callback_time)
        is_available = (analysis or {}).get("availability") == "available" or any(
            phrase in answer for phrase in ("free now", "available now", "convenient to talk", "ready to talk", "you can talk now", "can talk now")
        )
        if is_available:
            state["callback_status"] = "not_requested"
            state["availability"] = "available"
            state["stage"] = state.pop("callback_return_stage", "availability")
            return _process_deterministic_response(state, text, analysis)
        return "Please share a clearer callback time, such as in 10 minutes or tomorrow at 11 AM.", state

    if stage == "interest":
        interest = (analysis or {}).get("interest_status")
        if interest == "not_interested" or (interest is None and any(word in answer for word in ("no", "not interested", "decline"))):
            return _finish(state, "NOT_INTERESTED", "Customer declined the offer.", "Of course. Thank you for your time. Goodbye.")
        if interest == "interested" or (interest is None and any(word in answer for word in ("yes", "interested", "tell me", "more"))):
            state["stage"] = "existing_loan"
            return "Do you already have a loan against this property, or are you looking to reduce its EMI?", state
        return "Would you like to learn more about this LAP offer?", state

    if stage == "existing_loan":
        if _loan_transfer_intent(text):
            state["answers"]["existing_property_loan"] = True
            state["answers"]["emi_reduction_request"] = any(word in answer for word in ("emi", "reduce", "lower"))
            return _finish(state, "LOAN_TRANSFER_REFERRAL", "Existing property loan or EMI reduction request.", "A loan-transfer specialist would be better placed to help. I'll record this for the appropriate team. Goodbye.")
        if any(word in answer for word in ("no", "don't", "do not", "neither", "none")):
            state["answers"]["existing_property_loan"] = False
            state["stage"] = "checklist"
            return CHECKLIST[0][1], state
        if any(word in answer for word in ("yes", "already", "reduce", "emi", "existing loan")):
            state["answers"]["existing_property_loan"] = True
            return _finish(state, "LOAN_TRANSFER_REFERRAL", "Existing property loan or EMI reduction request.", "A loan-transfer specialist would be better placed to help. I'll record this for the appropriate team. Goodbye.")
        return "Just to confirm, do you already have a loan against this property or want to reduce its EMI?", state

    if stage == "checklist":
        if _loan_transfer_intent(text):
            state["answers"]["existing_property_loan"] = True
            state["answers"]["emi_reduction_request"] = any(word in answer for word in ("emi", "reduce", "lower"))
            return _finish(state, "LOAN_TRANSFER_REFERRAL", "Existing property loan or EMI reduction request.", "A loan-transfer specialist would be better placed to help. I'll record this for the appropriate team. Goodbye.")
        # Answer the common information diversion without changing the pending field.
        if "maximum" in answer or "max loan" in answer or "how much" in answer:
            return "The maximum amount under this offer is ₹75 lakh, subject to eligibility checks and final approval. " + _next_question(state), state
        pending = next((field for field, _question in CHECKLIST if not _answered(field, state["answers"])), None)
        extracted = _extract_answers(text, pending)
        analysis = analysis or {}
        llm_to_state = {
            "property_type": "property_type", "ownership": "ownership",
            "original_documents_available": "original_documents",
            "requested_loan_amount_inr": "loan_amount", "occupation": "occupation",
            "income_mode": "income_mode", "property_market_value_inr": "market_value",
            "requested_tenure_years": "tenure",
        }
        ambiguous_sources = set(analysis.get("ambiguous_fields", [])) if analysis.get("clarification_needed") else set()
        # LLM interpretation wins when it returns a validated, non-ambiguous fact;
        # the text parser remains a fallback when the model leaves a field blank.
        for source, target in llm_to_state.items():
            value = analysis.get(source)
            if source in ambiguous_sources:
                extracted.pop(target, None)
                continue
            if value is not None:
                extracted[target] = value
        if not extracted:
            if analysis.get("clarification_needed") and analysis.get("clarification_question"):
                return analysis["clarification_question"], state
            return "I didn't quite understand that answer. Please answer the current question, or ask about the maximum loan amount.", state
        state["answers"].update(extracted)
        reason = disqualification_reason(state["answers"])
        if reason:
            return _finish(state, "DISQUALIFIED", reason, reason + " This preliminary qualification cannot continue. Thank you for your time.")

        if analysis.get("clarification_needed") and analysis.get("clarification_question"):
            return analysis["clarification_question"], state

        requested = state["answers"].get("loan_amount")
        if requested and requested > MAX_LOAN_AMOUNT:
            state["stage"] = "amount_limit"
            state["answers"]["amount_over_limit"] = requested
            return "The maximum under this offer is ₹75 lakh. Would you like to proceed with ₹75 lakh?", state

        for field, question in CHECKLIST:
            if not _answered(field, state["answers"]):
                return question, state
        return _finish(state, "QUALIFIED_FOR_EXPERT", "Passed the preliminary checks; final approval is not guaranteed.", "Thank you. Your initial details meet these preliminary checks. I'll pass your request to a senior loan expert. This is not a final approval.")

    if stage == "amount_limit":
        if any(word in answer for word in ("yes", "agree", "okay", "ok", "proceed")):
            state["answers"]["loan_amount"] = MAX_LOAN_AMOUNT
            del state["answers"]["amount_over_limit"]
            state["stage"] = "checklist"
            for field, question in CHECKLIST:
                if field not in state["answers"]:
                    return question, state
        if any(word in answer for word in ("no", "decline", "don't", "do not")):
            return _finish(state, "CUSTOMER_WITHDREW", "Customer declined the capped amount.", "Understood. Thank you for your time. Goodbye.")
        return "Would you like to continue with a loan amount of ₹75 lakh?", state

    return "I couldn't process that response. Please try again or start a new simulation.", state


def process_response(state: dict, text: str, llm_service=None) -> tuple[str, dict]:
    """Use Groq for extraction and original wording; Python owns all decisions."""
    if not llm_service:
        # Explicit service-less mode is used by deterministic unit tests only.
        return _process_deterministic_response(state, text)
    if not llm_service.enabled:
        error = LLMServiceError("Groq is configured but unavailable.", "not_configured")
        logger.error("Conversation recovery selected reason=groq_client_unavailable")
        return ERROR_MESSAGES[error.category], state

    if state.get("status") != "active":
        return _process_deterministic_response(state, text)

    pending = _next_question(state) if state.get("stage") == "checklist" else None

    def generate(updated: dict, analysis: dict, decision: dict | None = None) -> str:
        return llm_service.generate_reply(
            text, state, updated, analysis,
            decision or _response_decision(state, updated, analysis, text),
        )

    try:
        lower_text = text.lower()
        is_driving = bool(re.search(r"\bdriving\b|\bbehind the wheel\b", lower_text))
        is_hands_free = bool(re.search(r"\bhands?[- ]?free\b", lower_text))
        is_parked = bool(re.search(r"\bparked\b", lower_text))
        # Clear callback requests use the deterministic callback detector first,
        # so a transient analysis-output issue cannot turn one into a generic retry.
        if _callback_takes_priority(text, None):
            analysis = _fallback_analysis(state, text) or {"intent": "callback"}
            callback_state = copy.deepcopy(state)
            if is_driving and not is_parked:
                callback_state["driving_status"] = "driving"
            _reply, updated = _process_deterministic_response(callback_state, text, analysis)
            return generate(updated, analysis), updated

        # Safety status changes and parked/hands-free confirmations are handled
        # deterministically so Groq cannot continue qualification while driving.
        if (is_driving or (state.get("stage") == "driving_check" and
                           (is_parked or is_hands_free or _callback_time_from_text(text)))):
            working_state = copy.deepcopy(state)
            reply, updated = _process_deterministic_response(working_state, text)
            return reply, updated

        try:
            analysis = llm_service.analyze_response(text, state, pending)
        except LLMServiceError as exc:
            if getattr(exc, "category", None) != "invalid_output":
                raise
            analysis = _fallback_analysis(state, text)
            logger.warning("Groq analysis output unusable; deterministic recognition=%s",
                           "matched" if analysis else "unrecognized")
            if analysis is None:
                return ERROR_MESSAGES["invalid_output"], state
        # Callback intent is evaluated first, but only after Groq has interpreted
        # the complete utterance. Python records the time and controls the outcome.
        if _callback_takes_priority(text, analysis):
            callback_state = copy.deepcopy(state)
            _reply, updated = _process_deterministic_response(callback_state, text, analysis)
            return generate(updated, analysis), updated

        analysis_transfer = (
            analysis.get("intent") in {"existing_property_loan", "emi_reduction"}
            or analysis.get("existing_property_loan") is True
            or analysis.get("emi_reduction_request") is True
        )
        lower_text = text.lower()
        explicitly_denies_transfer = any(phrase in lower_text for phrase in (
            "no existing loan", "don't have a loan", "do not have a loan",
            "no loan", "no emi reduction", "don't want to reduce", "do not want to reduce",
        ))
        verified_stage = state.get("stage") not in {"identity", "availability"}
        if (analysis_transfer or _loan_transfer_intent(text)) and not explicitly_denies_transfer and verified_stage:
            routed_state = copy.deepcopy(state)
            routed_state.setdefault("answers", {})["existing_property_loan"] = True
            routed_state["answers"]["emi_reduction_request"] = analysis.get("emi_reduction_request") is True or analysis.get("intent") == "emi_reduction"
            _finish(routed_state, "LOAN_TRANSFER_REFERRAL", "Existing property loan or EMI reduction request.", "")
            return generate(routed_state, analysis), routed_state

        if analysis.get("intent") == "question" and state.get("stage") == "checklist":
            lower = text.lower()
            if "personal loan" in lower:
                approved_reply = "This simulation can only help with the Loan Against Property offer. " + _next_question(state)
            elif "maximum" in lower or "max loan" in lower or "how much" in lower:
                approved_reply = "The maximum amount under this offer is ₹75 lakh, subject to eligibility checks and final approval. " + _next_question(state)
            else:
                approved_reply = "I can help with the Loan Against Property offer. Let's continue. " + _next_question(state)
            decision = _response_decision(state, state, analysis, text)
            return generate(state, analysis, decision), state

        if (analysis.get("intent") == "question" and state.get("stage") == "identity"
                and analysis.get("identity_confirmation") is not True):
            approved_reply = "I'm a simulated Home Credit representative calling about a Loan Against Property offer. May I speak with Ravi?"
            decision = _response_decision(state, state, analysis, text)
            return generate(state, analysis, decision), state

        if (analysis.get("intent") == "question" and not (
            state.get("stage") == "identity" and analysis.get("identity_confirmation") is True
        )):
            decision = _response_decision(state, state, analysis, text)
            return generate(state, analysis, decision), state

        working_state = copy.deepcopy(state)
        _reply, updated_state = _process_deterministic_response(working_state, text, analysis)
        return generate(updated_state, analysis), updated_state
    except LLMServiceError as exc:
        category = getattr(exc, "category", "llm_error")
        logger.warning("Conversation fallback selected reason=%s", category)
        return ERROR_MESSAGES.get(category, ERROR_MESSAGES["llm_error"]), state


def _next_question(state: dict) -> str:
    for field, question in CHECKLIST:
        if not _answered(field, state["answers"]):
            return question
    return "Please continue with your current answer."


def _next_checklist_field(state: dict) -> str | None:
    for field, _question in CHECKLIST:
        if not _answered(field, state.get("answers", {})):
            return field
    return None


def _response_decision(previous: dict, current: dict, analysis: dict, text: str) -> dict:
    """Build reply instructions from state, without carrying a canned sentence."""
    outcome = current.get("outcome")
    if outcome == "callback_requested":
        return {"action": "acknowledge_callback_request_and_close",
                "callback_time": current.get("answers", {}).get("callback_time"),
                "must_not_claim_scheduled": True}
    terminal_actions = {
        "IDENTITY_NOT_VERIFIED": "protect_customer_privacy_and_close",
        "DISQUALIFIED": "explain_python_determined_ineligibility_and_close",
        "LOAN_TRANSFER_REFERRAL": "explain_specialist_referral_and_close",
        "QUALIFIED_FOR_EXPERT": "explain_qualified_for_expert_handoff_and_close_without_claiming_approval",
        "NOT_INTERESTED": "acknowledge_decline_and_close",
        "CUSTOMER_WITHDREW": "acknowledge_withdrawal_and_close",
    }
    if outcome in terminal_actions:
        return {"action": terminal_actions[outcome], "reason": current.get("reason"),
                "confirmed_facts": current.get("answers", {})}
    stage = current.get("stage")
    if previous.get("stage") == "identity" and current.get("identity_verified") is True:
        if analysis.get("intent") == "question":
            return {"action": "explain_call_purpose_as_lap_offer_then_check_availability",
                    "maximum_offer_inr": MAX_LOAN_AMOUNT, "final_approval_required": True}
        return {"action": "acknowledge_identity_and_check_availability"}
    if analysis.get("intent") == "question":
        if stage == "identity":
            return {"action": "identify_simulated_caller_and_check_intended_customer",
                    "customer_name": current.get("customer_name")}
        if stage == "availability":
            return {"action": "explain_call_purpose_as_lap_offer_then_check_availability",
                    "maximum_offer_inr": MAX_LOAN_AMOUNT, "final_approval_required": True}
        if stage == "interest":
            return {"action": "answer_lap_question_within_known_offer_facts_then_resume_interest_question",
                    "maximum_offer_inr": MAX_LOAN_AMOUNT, "final_approval_required": True}
        if stage == "existing_loan":
            return {"action": "answer_within_lap_scope_then_resume_existing_loan_question"}
        if stage == "callback_time":
            return {"action": "answer_briefly_then_request_callback_time_or_resume_if_available"}
        if stage == "amount_limit":
            return {"action": "clarify_maximum_offer_then_confirm_whether_customer_will_proceed",
                    "maximum_offer_inr": MAX_LOAN_AMOUNT}
        if "personal loan" in text.lower():
            return {"action": "explain_scope_is_lap_only_then_resume", "next_field": _next_checklist_field(current)}
        if any(token in text.lower() for token in ("maximum", "max loan", "how much")):
            return {"action": "answer_offer_limit_and_resume", "maximum_offer_inr": MAX_LOAN_AMOUNT,
                    "final_approval_required": True, "next_field": _next_checklist_field(current)}
        return {"action": "briefly_answer_within_lap_scope_then_resume", "next_field": _next_checklist_field(current)}
    stage_actions = {
        "identity": {"action": "verify_intended_customer_identity", "customer_name": current.get("customer_name")},
        "availability": {"action": "check_customer_availability"},
        "interest": {"action": "present_lap_offer_and_check_interest", "maximum_offer_inr": MAX_LOAN_AMOUNT,
                     "final_approval_required": True},
        "existing_loan": {"action": "check_existing_property_loan_or_emi_reduction"},
        "callback_time": {"action": "clarify_callback_time_or_resume_if_available",
                          "callback_return_stage": current.get("callback_return_stage")},
        "amount_limit": {"action": "confirm_customer_acceptance_of_maximum_offer", "maximum_offer_inr": MAX_LOAN_AMOUNT},
        "checklist": {"action": "ask_first_unanswered_eligibility_item", "next_field": _next_checklist_field(current),
                      "maximum_offer_inr": MAX_LOAN_AMOUNT, "confirmed_facts": current.get("answers", {})},
    }
    return stage_actions.get(stage, {"action": "continue_current_lap_conversation", "stage": stage})
