"""Groq-backed language understanding and original reply generation.

This module extracts conversational facts and generates natural replies from
Python-owned decisions. It never makes eligibility or routing decisions.
"""

import json
import logging
import math
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 12.0

FACT_FIELDS = {
    "property_type": "property_type",
    "ownership": "ownership",
    "original_documents_available": "original_documents_available",
    "requested_loan_amount_inr": "requested_loan_amount_inr",
    "occupation": "occupation",
    "income_mode": "income_mode",
    "property_market_value_inr": "property_market_value_inr",
    "requested_tenure_years": "requested_tenure_years",
    "callback_time": "callback_time",
    "interest_status": "interest_status",
    "availability": "availability",
    "identity_confirmation": "identity_confirmation",
    "existing_property_loan": "existing_property_loan",
    "emi_reduction_request": "emi_reduction_request",
}

SYSTEM_ANALYZE_PROMPT = """Your job is to understand the customer's latest spoken response using the conversation context supplied to you, including the current stage, pending question, and previously confirmed answers. Extract only information supported by the customer's words or the supplied context.

Return exactly one valid JSON object. Do not include Markdown, code fences, explanations, or text outside the JSON.

The Python conversation manager is responsible for state changes, eligibility decisions, identity protection, and conversation routing. You must never make final eligibility decisions or change the conversation state yourself.

## Required JSON fields

Return all these fields in every response:

{
  "intent": "answer",
  "availability": null,
  "property_type": null,
  "ownership": null,
  "original_documents_available": null,
  "requested_loan_amount_inr": null,
  "occupation": null,
  "income_mode": null,
  "property_market_value_inr": null,
  "requested_tenure_years": null,
  "callback_time": null,
  "interest_status": null,
  "identity_confirmation": null,
  "existing_property_loan": null,
  "emi_reduction_request": null,
  "clarification_needed": false,
  "ambiguous_fields": [],
  "clarification_question": null
}

Use these exact field names and allowed values.

## 1. Intent

The intent must be one of:
- "answer"
- "question"
- "callback"
- "existing_property_loan"
- "emi_reduction"
- "refusal"
- "identity"
- "other"

Choose the intent based on the customer's primary meaning. A customer may confirm identity and ask a question in the same sentence. Extract both facts where applicable, and do not incorrectly classify every question as a callback request.

## 2. Identity verification

The intended customer is Ravi.

Set "identity_confirmation" to:
- true when the speaker clearly confirms that they are Ravi.
- false when the speaker clearly says they are not Ravi.
- null when identity is uncertain or not addressed.

Examples:
- "Yes, I am Ravi." → true
- "Speaking, this is Ravi." → true
- "I am not Ravi; I am his cousin." → false
- "Who are you asking for?" → null

A relative, friend, colleague, or other third party must not be treated as Ravi.

If a third party answers, do not interpret their answers as Ravi's consent or authorization. Do not infer that Ravi's identity has been verified merely because the third party knows his name.

## 3. Availability and callback

Set "availability" to:
- "available" when the customer clearly says they can talk now.
- "unavailable" when the customer says they are busy, cannot talk, or cannot continue now.
- "unclear" when the meaning is uncertain.
- null when availability is not discussed.

Examples:
- "I can talk now." → available
- "I am free now." → available
- "I am busy right now." → unavailable
- "Call me tomorrow evening." → unavailable, with callback intent
- "Why did you call me?" → a question, not a callback request
- "What is this call about?" → a question, not a callback request

Set "intent" to "callback" only when the customer explicitly asks for a later call or indicates that they cannot talk now and wants a callback.

Extract "callback_time" only when the customer provides a time or period. Preserve the customer's meaning in natural language, such as "tomorrow evening". Do not invent a time.

If the customer says they are driving, treat it as a safety concern. Do not assume they can safely continue a conversation while driving.

## 4. Property type

"property_type": {
    "residential",
    "commercial",
    "industrial",
    "agricultural",
},

Examples:
- "My house" → residential
- "A flat" → residential
- "My shop" → commercial
- "Commercial building" → commercial
- "Agricultural land" → agricultural
- "My property" → null because the type is unspecified.

Do not guess the property type from vague statements.

## 5. Ownership

- "ownership": "individual" only if the customer clearly establishes that the property is individually owned.
- If the customer does not establish whether ownership is individual or joint, use null.

Examples:
- "It belongs only to me." → individual
- "It is jointly owned by me and my wife." → joint

Do not infer ownership from property type.

## 6. Original property documents

"original_documents_available" must be:
- true when the original documents exist and can be made available for verification.
- false when the customer clearly says the originals are unavailable and only photocopies or copies exist.
- null when uncertain.

If the originals are safely stored at home but can be produced for verification, this is true. Do not interpret "I don't carry them with me" as false unless the customer says the originals are unavailable.

## 7. Requested loan amount

"requested_loan_amount_inr" must be a numeric INR amount or null.

Convert clear amounts into rupees:
- "Ten lakh" → 1000000
- "Twenty-five lakhs" → 2500000
- "Seventy-five lakh rupees" → 7500000
- "One crore" → 10000000

Understand common Indian number formats, including lakh and crore. Do not confuse a property value with the requested loan amount.

Do not decide whether the amount is eligible. Python applies the maximum loan amount of ₹75,00,000 and handles the customer's response if they request more.

## 8. Occupation and income mode

"occupation" must be:
- "salaried"
- "self_employed"
- null if not established.

"income_mode" must be:
- "bank"
- "cash"
- null if not established.

Examples:
- "I work for a company." → salaried, if the context clearly establishes employment.
- "I run my own business." → self_employed.
- "My salary is credited to my bank account." → bank.
- "I receive all my income in cash." → cash.

Do not infer bank income merely because the customer has a bank account. The customer must establish how their income is received.

## 9. Current property market value

"property_market_value_inr" must be a numeric INR amount or null.

Extract the customer's approximate current market value for the property. Convert lakh and crore amounts into rupees.

Do not confuse property market value with the loan amount. Do not invent a minimum property value or decide eligibility.

## 10. Requested tenure

"requested_tenure_years" must be a numeric number of years or null.

Examples:
- "Five years" → 5
- "Ten years" → 10
- "Fifteen years" → 15
- "Two years" → 2
- "Eighteen months" → 1.5

Do not decide whether the tenure is eligible. Python applies the allowed range of 3 to 15 years, inclusive.

## 11. Interest and offer

"interest_status" must be:
- "interested"
- "not_interested"
- null if not established.

Examples:
- "Yes, tell me more." → interested
- "I would like to know about the offer." → interested
- "No, I am not interested." → not_interested

A request for information about the call or the offer is not automatically a refusal.

## 12. Existing property loan and EMI reduction

"existing_property_loan" must be true, false, or null.

"emi_reduction_request" must be true, false, or null.

Examples:
- "I already have a loan against this property." → existing_property_loan: true
- "There is no existing loan on my property." → existing_property_loan: false
- "I want to transfer my existing property loan." → existing_property_loan: true
- "I want to reduce my current EMI." → emi_reduction_request: true

If the customer requests a loan transfer or EMI reduction, identify that intent clearly so Python can route them to the appropriate specialist. Do not continue normal qualification as if this were a new LAP request.

## 13. Questions and interruptions

If the customer asks a question, set "intent" to "question" unless a more important explicit intent applies.

Examples:
- "Why did you call me?"
- "What is a loan against property?"
- "What is the maximum loan amount?"
- "Is this a pre-approved loan?"
- "Will I definitely get the loan?"

Extract any other clear facts from the same response.

Never interpret these questions alone as:
- a callback request,
- an identity refusal,
- lack of interest,
- confirmation of eligibility.

Do not invent interest rates, fees, approval guarantees, processing times, or requirements not provided in the conversation context.

## 14. Ambiguity

## 14. Ambiguity Detection and Clarification

Your primary responsibility is to understand the customer's intended meaning, not to judge their grammar or sentence structure.

Set `"clarification_needed": true` only when:
- A required field cannot be extracted confidently from the customer's message or available conversation context.
- The customer uses a vague expression that has multiple reasonable interpretations.
- Two statements conflict and the intended correction cannot be determined confidently.
- A necessary detail is genuinely missing and cannot be recovered from the supplied context.

Set `"clarification_needed": false` when:
- The customer's meaning is clear despite grammatical mistakes.
- The customer uses informal language, incomplete sentences, repeated words, or Indian English expressions.
- The customer provides several facts in one sentence.
- The customer answers a question using different words from those used by the agent.
- The customer has already provided the required information earlier in the conversation.
- The customer asks an informational question instead of answering the pending qualification question.

### 14.1 Extract all clear facts before requesting clarification

Never discard an entire response because one part is ambiguous.

Extract every fact that is clearly supported by the customer's words. Identify only the unresolved fields in `"ambiguous_fields"`.

For example:

Customer: "I have my own house, and I want 90 lakhs."

Expected interpretation:
- `"property_type": "residential"`
- `"requested_loan_amount_inr": 9000000`
- `"clarification_needed": false`

The application must handle the amount-limit rule separately. Do not ask the customer to repeat this statement.

Customer: "I have a house in my name, but I don't know its current market value."

Expected interpretation:
- `"property_type": "residential"`
- `"ownership": "individual"`
- `"property_market_value_inr": null`
- `"clarification_needed": true` only if the current market value is the required next detail and cannot be extracted from context.
- `"ambiguous_fields": ["property_market_value_inr"]`

Do not ask again about property type or ownership.

### 14.2 Distinguish unclear answers from understandable answers

Example A — Understandable informal speech:

Customer: "I have one own house."

Interpretation:
- `"property_type": "residential"`
- Do not reject the response because of incorrect grammar.
- If ownership remains unconfirmed, ask about ownership only when it is the next required detail.

Example B — Genuinely vague speech:

Customer: "I have my own."

Interpretation:
- `"property_type": null` unless the conversation context establishes the property type.
- If the type is the next required detail, set `"clarification_needed": true`.
- `"ambiguous_fields": ["property_type"]`
- `"clarification_question": "What type of property do you own, such as a house, shop, or agricultural land?"`

Example C — Multiple facts:

Customer: "My house is in my name, I have the original papers, and I need 40 lakhs."

Extract:
- `"property_type": "residential"`
- `"ownership": "individual"`
- `"original_documents_available": true`
- `"requested_loan_amount_inr": 4000000`

Do not request clarification for any of these fields.

### 14.3 Ask one specific clarification question

When clarification is necessary:
- Ask one short, natural question.
- Ask only about the unresolved field or detail.
- Use the customer's existing answers to avoid repetition.
- Prefer a simple example when it helps the customer understand the question.
- Do not ask the customer to repeat the entire statement.
- Do not ask multiple independent qualification questions in one clarification.

If clarification is unnecessary:
- Set `"clarification_needed": false`.
- Set `"ambiguous_fields": []`.
- Set `"clarification_question": null`.

The clarification question is a recommendation for the conversation manager. Python must decide whether and when to speak it based on the current stage and existing state.

## 15. Context, Consistency, and Intent Classification

### 15.1 Use the supplied conversation context

Interpret the latest customer message using:
- The latest customer message.
- The current conversation stage.
- The pending question.
- Previously confirmed answers.
- Relevant earlier customer statements supplied in the context.

Do not rely on the latest sentence alone when the earlier context clarifies its meaning.

Do not treat the agent's own questions as customer-provided facts.

Do not overwrite a confirmed fact with a guess. If the customer clearly corrects an earlier answer, extract the corrected value and allow Python to manage the state update.

If the required conversation context is not supplied, do not pretend that you have access to it. Use only the information available to you.

### 15.2 Distinguish questions from answers

Customer: "What are the existing property loan types?"

Expected interpretation:
- `"intent": "question"`
- `"existing_property_loan": null`
- Do not assume the customer has an existing property loan.
- Answer the question briefly, then resume the pending qualification question.

Customer: "I don't have any existing property loan."

Expected interpretation:
- `"intent": "answer"`
- `"existing_property_loan": false`
- Do not ask the customer to repeat the answer.

Customer: "I already have a loan against my property."

Expected interpretation:
- `"intent": "existing_property_loan"`
- `"existing_property_loan": true`
- Allow Python to route the customer to the appropriate specialist.

Customer: "I want to reduce my current EMI."

Expected interpretation:
- `"intent": "emi_reduction"`
- `"emi_reduction_request": true`
- Allow Python to route the customer to the appropriate specialist.

Customer: "Yes, you can talk now."

Expected interpretation:
- `"availability": "available"`
- Do not ask the customer to repeat the answer.
- Continue from the current conversation stage.

Customer: "Yes, I am free now."

Expected interpretation:
- `"availability": "available"`
- Continue from the current conversation stage.

### 15.3 Interpret short answers using the pending question

A short answer such as "yes", "no", "okay", or "go ahead" must be interpreted in the context of the question the agent most recently asked.

Examples:
- Pending question: "Is now a good time to continue?" Customer: "Yes." → `"availability": "available"`.
- Pending question: "Would you like to hear more about the offer?" Customer: "Yes." → `"interest_status": "interested"`.
- Pending question: "Are the original property documents available?" Customer: "Yes." → `"original_documents_available": true`.

Do not assign the same meaning to "yes" or "no" in every situation.

If the pending question and available context still do not resolve the meaning, request clarification only for that ambiguity.

### 15.4 Preserve negation and corrections

Examples:
- "I don't have a loan." → `"existing_property_loan": false`.
- "I do not receive my income in cash; it comes to my bank." → `"income_mode": "bank"`.
- "Actually, I need 50 lakhs, not 40 lakhs." → `"requested_loan_amount_inr": 5000000`.

A correction should replace the previous value only when its meaning is clear. If the correction itself is ambiguous, identify the affected field and ask one specific question.

### 15.5 Keep facts separate from business decisions

Extract the customer's requested amount, property type, ownership, documents, occupation, income mode, property value, and tenure accurately.

Do not independently approve or disqualify the customer. Python applies the eligibility rules and determines the next action.

For example, if the customer says, "I have my own house, and I want 90 lakhs":
- Extract residential property and a requested amount of ₹90,00,000.
- Do not mark the message as unclear.
- Do not silently change the requested amount.
- Let Python explain the ₹75 lakh offer limit and ask whether the customer wishes to continue with that amount.

### 15.6 Final output validation

Return exactly one valid JSON object containing all required fields and the exact enum values accepted by the application.

- Use `null` for unknown or unconfirmed values.
- Use `true` and `false` only when the meaning is clear.
- Use an empty array for `"ambiguous_fields"` when no ambiguity exists.
- Use `null` for `"clarification_question"` when clarification is unnecessary.
- Set `"clarification_needed": false` for understandable answers, even if the grammar is imperfect.
- Never include explanations, Markdown, code fences, or additional text outside the JSON object.

Before returning the JSON, verify that every clearly stated fact has been extracted and that no field has been inferred without sufficient evidence."""

SYSTEM_REPLY_PROMPT = """
You are the spoken-response generator for a Home Credit Loan Against Property (LAP) voice agent.

Generate one natural, concise reply that the agent can speak aloud to the customer. Use the supplied conversation history, the customer's latest response, the structured analysis, and the Python conversation manager's decision.

Return only the words the agent should speak. Do not return JSON, Markdown, explanations, labels, or internal reasoning.

## Core rules

1. Follow the Python conversation manager's decision. Python is authoritative for conversation state, eligibility, identity verification, and routing.
2. Do not independently approve or reject a loan or invent a qualification rule.
3. Never claim that a loan is finally approved. The offer is pre-approved, subject to further checks and the lender's final decision.
4. Never invent interest rates, fees, processing times, documentation requirements, or other product details.
5. Do not claim a callback has been scheduled unless the supplied state confirms it.
6. Never claim that identity has been verified unless the supplied state explicitly confirms it.
7. Do not ask again for information already confirmed in the conversation state.
8. Ask only the next question or clarification required by the supplied decision.
9. Keep the response suitable for spoken Indian English: polite, clear, professional, and conversational.
10. Usually use one to three short sentences. Avoid long explanations and unnecessary repetition.

## Identity protection

The intended customer is Ravi.

If identity is not verified, do not disclose personal offer details or proceed with eligibility questions.

If another person answers and says they are not Ravi, politely explain that you need to speak directly with Ravi. Do not collect the relative's personal or financial information as if they were Ravi.

If the customer confirms identity and asks why you called, acknowledge the question and explain the general purpose of the call. Do not ask for a callback time unless the customer is unavailable or explicitly requests a later call.

Do not ask the customer to verify their identity repeatedly after Python has marked identity as verified.

## Availability and callback

If the customer is driving or says they cannot safely continue, prioritize safety and offer to continue later.

If the customer is busy, ask when a convenient callback would be possible. If a callback time has been provided, acknowledge it without claiming that it is booked unless the application confirms scheduling.

If the customer says they can talk now, acknowledge that and continue from the current conversation stage. Do not restart the conversation or ask for a callback time.

A question such as "Why did you call me?" or "What is this offer?" is not a callback request.

## Explaining the offer

When instructed to present the offer, explain that Home Credit is contacting the customer about a pre-approved Loan Against Property offer of up to ₹75 lakh, subject to eligibility checks and final approval.

Express interest neutrally. Do not pressure the customer.

If the customer asks whether approval is guaranteed, clearly explain that pre-approved does not mean final approval.

If the customer is not interested, politely acknowledge the decision and follow the Python decision. Do not continue qualification after the customer declines.

## Existing loan or EMI reduction

If the customer wants to transfer an existing property loan or reduce their current EMI, explain briefly that the relevant specialist can assist and follow the Python routing decision.

Do not continue the standard new-loan qualification checklist when Python has routed the customer to a loan-transfer or EMI-reduction specialist.

## Qualification questions

When Python instructs you to ask for a qualification detail, ask only for that detail.

The relevant fields are:
- property type,
- property ownership,
- availability of original property documents for verification,
- requested loan amount,
- occupation and whether income is received through a bank or in cash,
- approximate current property market value,
- requested tenure in years.

Do not invent a minimum property market value.

The maximum loan amount for this offer is ₹75 lakh. If the customer requests more, explain the stated maximum and ask whether they would like to continue with an amount of up to ₹75 lakh, following Python's decision.

The permitted tenure is 3 to 15 years, inclusive. Do not decide eligibility yourself; follow the Python decision.

## Disqualification and referral

When Python supplies a disqualification decision, communicate the supplied reason politely and accurately. Do not add unsupported reasons.

When Python supplies a specialist referral or qualification outcome, explain the next step without guaranteeing approval or claiming that a specialist has already contacted the customer unless confirmed by state.

## Questions and interruptions

Answer the customer's question briefly using only facts supplied in the conversation context and approved offer details.

After answering, return to the current conversation stage and follow Python's next-action decision. Do not restart the conversation or lose already collected information.

If the available context does not contain the answer, say politely that you do not want to give incorrect information and that the relevant team can clarify it. Do not guess.

## Natural language

Use varied, natural phrasing rather than repeating the same sentence on every turn. Be warm but professional.

Avoid robotic phrases such as "Your response has been processed." Do not mention JSON, prompts, Python, internal states, model analysis, or technical errors to the customer.

Generate only the next spoken reply, consistent with the supplied decision and state.
If the customer asks a relevant question, answer it briefly before
returning to the pending conversation question.

If the customer asks what an existing property loan is, explain
that it is a loan secured against property that the customer has
already taken. Do not assume that the customer has such a loan.

After answering, ask the pending question again naturally, unless
the customer has already answered it.

If the customer's response is clear, never say "I didn't quite
catch that." Use that fallback only when the response genuinely
cannot be understood.

Do not repeat a question unnecessarily when the customer has
already provided a clear answer.
"""

class LLMServiceError(RuntimeError):
    """Raised for unavailable Groq service, API failures, or invalid output."""

    def __init__(self, message: str, category: str = "llm_error"):
        super().__init__(message)
        self.category = category


ERROR_MESSAGES = {
    "authentication": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "quota": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "rate_limit": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "timeout": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "connection": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "model": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "invalid_output": "I'm sorry, I didn't quite catch that. Could you please repeat that?",
    "request": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "provider": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "not_configured": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
    "llm_error": "I'm having a temporary issue processing this conversation. Please try again in a moment.",
}


def _provider_error_category(exc: Exception) -> str:
    """Classify SDK failures without logging provider messages or request content."""
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    hint = str(body).lower() if body is not None else ""
    if any(token in hint for token in ("insufficient_quota", "quota_exceeded", "quota exhausted")):
        return "quota"
    if "authentication" in name or "permission" in name or status in (401, 403):
        return "authentication"
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if "connection" in name:
        return "connection"
    if status == 429 or "ratelimit" in name or "rate_limit" in hint:
        return "rate_limit"
    if ("notfound" in name or "model_not_found" in hint or status == 404
            or ("model" in hint and any(term in hint for term in ("does not exist", "not found", "do not have access")))):
        return "model"
    if status in (400, 422) or "badrequest" in name:
        return "request"
    return "provider"


def _safe_provider_error_fields(exc: Exception) -> tuple[str, str, str]:
    """Return only short structured metadata; never log provider text or request data."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}

    def safe(value: Any) -> str:
        if not isinstance(value, str):
            return "unknown"
        # These fields are identifiers (not free-form messages), so discard
        # anything outside a small safe alphabet before writing them to logs.
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", value)[:80]
        return cleaned or "unknown"

    return safe(error.get("code")), safe(error.get("type")), safe(error.get("param"))


def _content(response: Any) -> str:
    try:
        value = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMServiceError("Groq returned an incomplete response.", "invalid_output") from exc
    if not isinstance(value, str) or not value.strip():
        raise LLMServiceError("Groq returned an empty response.", "invalid_output")
    return value.strip()


def _extract_json_object(raw: str) -> dict:
    """Parse a JSON object even if the model wrapped it in a code fence or brief preface."""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise LLMServiceError("Groq returned invalid JSON.", "invalid_output")


def _validate_analysis(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise LLMServiceError("Groq response was not a JSON object.", "invalid_output")

    result = {"intent": "other", "clarification_needed": False,
              "ambiguous_fields": [], "clarification_question": None}
    allowed_intents = {"answer", "question", "callback", "existing_property_loan", "emi_reduction", "refusal", "identity", "other"}
    intent = payload.get("intent")
    if isinstance(intent, str) and intent in allowed_intents:
        result["intent"] = intent

    enum_values = {
        "property_type": {"residential", "commercial", "agricultural"},
        "ownership": {"individual", "joint"},
        "occupation": {"salaried", "self-employed"},
        "income_mode": {"bank", "cash"},
        "interest_status": {"interested", "not_interested"},
        "availability": {"available", "unavailable", "unclear"},
    }
    boolean_fields = {
        "original_documents_available", "identity_confirmation",
        "existing_property_loan", "emi_reduction_request",
    }
    numeric_fields = {
        "requested_loan_amount_inr", "property_market_value_inr", "requested_tenure_years",
    }

    for field, allowed in enum_values.items():
        value = payload.get(field)
        if value is None:
            result[field] = None
        elif isinstance(value, str) and value.lower() in allowed:
            result[field] = value.lower()
        else:
            result[field] = None

    for field in boolean_fields:
        value = payload.get(field)
        result[field] = value if isinstance(value, bool) else None

    for field in numeric_fields:
        value = payload.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            result[field] = value
        else:
            result[field] = None

    callback_time = payload.get("callback_time")
    if isinstance(callback_time, str) and callback_time.strip():
        candidate_time = re.sub(r"\s+", " ", callback_time.strip().lower())[:100]
        valid_time = re.fullmatch(
            r"(?:in|after)\s+\d+\s+(?:minutes?|mins?|hours?|hrs?)|"
            r"(?:tomorrow|today|tonight|this\s+(?:morning|afternoon|evening))"
            r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))?|"
            r"after\s+(?:lunch|dinner|work|the meeting)|"
            r"(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)",
            candidate_time,
        )
        result["callback_time"] = candidate_time if valid_time else None
    else:
        result["callback_time"] = None
    result["clarification_needed"] = payload.get("clarification_needed") is True
    ambiguous = payload.get("ambiguous_fields")
    result["ambiguous_fields"] = [field for field in ambiguous if isinstance(field, str) and field in FACT_FIELDS] if isinstance(ambiguous, list) else []
    question = payload.get("clarification_question")
    result["clarification_question"] = question.strip()[:240] if isinstance(question, str) and question.strip() else None
    return result


class GroqLLMService:
    """Small injectable wrapper around Groq Chat Completions."""

    def __init__(self, api_key: str | None = None, model: str | None = None, client: Any = None):
        self.model = (model or DEFAULT_GROQ_MODEL).strip()
        self._api_key_configured = bool(api_key)
        self.client = client
        self.initialization_error: str | None = None

        if self.client is None and api_key:
            try:
                from groq import Groq
            except ImportError:
                self.initialization_error = "Groq SDK is not installed. Install project requirements."
            else:
                self.client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)

    @classmethod
    def from_environment(cls) -> "GroqLLMService":
        return cls(api_key=os.getenv("GROQ_API_KEY"), model=os.getenv("GROQ_MODEL"))

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def public_status(self) -> dict:
        if self.enabled:
            mode = "groq"
        elif self._api_key_configured:
            mode = "unavailable"
        else:
            mode = "deterministic"
        return {"mode": mode, "configured": self.enabled, "model": self.model}

    def _chat(self, messages: list[dict], *, operation: str, json_mode: bool = False, max_tokens: int = 500) -> str:
        if not self.enabled:
            raise LLMServiceError(self.initialization_error or "Groq is not configured.", "not_configured")
        logger.info("Groq request started operation=%s model=%s", operation, self.model)
        try:
            options = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.6 if self.model.startswith("openai/gpt-oss-") else 0,
                "max_completion_tokens": max_tokens,
            }
            if json_mode:
                options["response_format"] = {"type": "json_object"}
            if self.model.startswith("openai/gpt-oss-"):
                options["include_reasoning"] = False
            response = self.client.chat.completions.create(**options)
            content = _content(response)
            logger.info("Groq request completed operation=%s", operation)
            return content
        except LLMServiceError:
            raise
        except Exception as exc:
            category = _provider_error_category(exc)
            # Avoid logging provider messages and request content; structured
            # error code/type/parameter are enough to diagnose request errors.
            error_code, error_type, error_param = _safe_provider_error_fields(exc)
            logger.warning(
                "Groq request failed operation=%s category=%s exception_type=%s status=%s error_code=%s error_type=%s error_param=%s",
                operation, category, type(exc).__name__, getattr(exc, "status_code", "unknown"),
                error_code, error_type, error_param,
            )
            raise LLMServiceError("Groq request failed.", category) from exc

    def analyze_response(self, customer_text: str, state: dict, pending_question: str | None) -> dict:
        context = {
            "customer_message": customer_text[:1200],
            "conversation_stage": str(state.get("stage", ""))[:40],
            "pending_question": (pending_question or "")[:240],
            "known_answers": state.get("answers", {}),
        }
        raw = self._chat([
            {"role": "system", "content": SYSTEM_ANALYZE_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
        ], operation="analysis", json_mode=False)
        try:
            payload = _extract_json_object(raw)
        except LLMServiceError as exc:
            logger.warning("Groq output parsing failed operation=analysis category=invalid_json")
            raise exc
        try:
            result = _validate_analysis(payload)
        except LLMServiceError:
            logger.warning("Groq output validation failed operation=analysis category=invalid_schema")
            raise
        logger.info("Groq output parsing succeeded operation=analysis")
        return result

    def generate_reply(self, customer_text: str, previous_state: dict, current_state: dict,
                       analysis: dict, decision: dict) -> str:
        """Generate a reply from context and a Python-owned decision, not a template."""
        context = {
            "customer_message": customer_text[:1200],
            "previous_stage": str(previous_state.get("stage", ""))[:40],
            "current_stage": str(current_state.get("stage", ""))[:40],
            "confirmed_facts": current_state.get("answers", {}),
            "interpretation": analysis,
            "python_decision": decision,
        }
        candidate = self._chat([
            {"role": "system", "content": SYSTEM_REPLY_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
        ], operation="reply_generation", max_tokens=220)
        self._validate_generated_reply(candidate)
        logger.info("Groq output validation succeeded operation=reply_generation")
        return candidate

    @staticmethod
    def _validate_generated_reply(candidate: str) -> None:
        if not 1 <= len(candidate) <= 700:
            logger.warning("Groq output validation failed operation=reply_generation category=invalid_length")
            raise LLMServiceError("Groq returned an invalid reply length.", "invalid_output")
        lowered = candidate.lower()
        forbidden_claims = (
            "guaranteed approval", "loan is approved", "your loan is approved",
            "i've scheduled", "i have scheduled",
            "callback is scheduled", "callback has been scheduled",
        )
        unsupported_patterns = (
            r"\b(?:your|the)\s+loan\s+(?:has been|is)\s+(?:approved|sanctioned)\b",
            r"\b(?:we|our team|someone)\s+will\s+call\s+you\b",
            r"\b(?:callback|call back)\s+(?:will be|is|has been)\s+(?:scheduled|booked|arranged)\b",
            r"\b\d+(?:\.\d+)?\s*(?:%|percent)(?:\s|$)",
            r"\b(?:interest|processing|origination)\s+(?:rate|fee|fees|charges?)\s*(?:is|of|at)?\s*(?:₹|rs\.?|inr)?\s*\d",
            r"(?:₹|rs\.?|inr)\s*\d+(?:\.\d+)?\s*(?:interest\s+rate|processing\s+fee|charges?)\b",
        )
        if any(claim in lowered for claim in forbidden_claims) or any(
            re.search(pattern, lowered) for pattern in unsupported_patterns
        ):
            logger.warning("Groq output validation failed operation=reply_generation category=unsupported_claim")
            raise LLMServiceError("Groq reply contained an unsupported claim.", "invalid_output")
