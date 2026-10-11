"""Mocked Groq service and integration tests; never call the real API."""

import json
import logging
import unittest
from types import SimpleNamespace

from app import create_app
from backend.conversation_manager import _callback_takes_priority, new_conversation, process_response
from backend.llm_service import GroqLLMService, LLMServiceError


def response_with(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletions:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return response_with(answer)


def fake_client(*answers):
    completions = FakeCompletions(answers)
    return SimpleNamespace(chat=SimpleNamespace(
        complete=completions.create,
        completions=completions,  # Keep call inspection convenient in tests.
    )), completions


class GroqServiceTests(unittest.TestCase):
    def test_analyzes_structured_facts_and_validates_json_locally(self):
        payload = {"intent": "answer", "property_type": "residential", "ownership": "joint",
                   "property_market_value_inr": 8_000_000, "clarification_needed": False}
        client, calls = fake_client(json.dumps(payload))
        service = GroqLLMService(api_key="mock-secret", client=client)
        analysis = service.analyze_response("Residential, jointly owned, worth 80 lakh", {}, None)
        self.assertEqual(analysis["ownership"], "joint")
        self.assertEqual(analysis["property_market_value_inr"], 8_000_000)
        self.assertNotIn("response_format", calls.calls[0])
        self.assertIn("Return exactly one valid JSON object", calls.calls[0]["messages"][0]["content"])
        self.assertEqual(calls.calls[0]["temperature"], 0.6)
        self.assertIn("max_completion_tokens", calls.calls[0])
        self.assertNotIn("max_tokens", calls.calls[0])

    def test_default_model_is_gpt_oss_20b(self):
        client, calls = fake_client("ok")
        service = GroqLLMService(client=client)
        self.assertEqual(service.model, "openai/gpt-oss-20b")
        service._chat([], operation="model_check")
        self.assertEqual(calls.calls[0]["model"], "openai/gpt-oss-20b")
        self.assertTrue(hasattr(client.chat, "completions"))

    def test_analysis_accepts_json_wrapped_in_fence_or_short_preface(self):
        raw = 'Here is the analysis:\n```json\n{"intent":"question","clarification_needed":false}\n```'
        client, _ = fake_client(raw)
        analysis = GroqLLMService(client=client).analyze_response("What is the rate?", {}, None)
        self.assertEqual(analysis["intent"], "question")

    def test_provider_rejection_logs_safe_parameter_metadata_only(self):
        class BadRequestError(Exception):
            status_code = 400
            body = {"error": {
                "message": "bad field for customer speech do-not-log-this",
                "type": "invalid_request_error", "code": "bad_value", "param": "temperature",
            }}

        client, _ = fake_client(BadRequestError("secret message"))
        service = GroqLLMService(client=client)
        with self.assertLogs("backend.llm_service", level="WARNING") as logs:
            with self.assertRaises(LLMServiceError):
                service.analyze_response("customer private speech", {}, None)
        log_text = " ".join(logs.output)
        self.assertIn("error_param=temperature", log_text)
        self.assertIn("error_code=bad_value", log_text)
        self.assertNotIn("customer private speech", log_text)
        self.assertNotIn("do-not-log-this", log_text)
        self.assertNotIn("bad field", log_text)

    def test_invalid_json_and_provider_errors_are_normalized(self):
        client, _ = fake_client("not json")
        with self.assertRaises(LLMServiceError) as invalid_json:
            GroqLLMService(client=client).analyze_response("hello", {}, None)
        self.assertEqual(invalid_json.exception.category, "invalid_output")
        client, _ = fake_client(TimeoutError("sensitive provider details"))
        with self.assertRaises(LLMServiceError) as timeout:
            GroqLLMService(client=client).analyze_response("hello", {}, None)
        self.assertEqual(timeout.exception.category, "timeout")

    def test_authentication_quota_rate_limit_and_model_errors_are_distinguished(self):
        cases = (
            (type("AuthenticationError", (Exception,), {})(), "authentication"),
            (type("RateLimitError", (Exception,), {"status_code": 429, "body": {"code": "insufficient_quota"}})(), "quota"),
            (type("RateLimitError", (Exception,), {"status_code": 429})(), "rate_limit"),
            (type("NotFoundError", (Exception,), {"status_code": 404})(), "model"),
        )
        for error, expected in cases:
            with self.subTest(expected=expected):
                client, _ = fake_client(error)
                with self.assertRaises(LLMServiceError) as raised:
                    GroqLLMService(client=client).analyze_response("hello", {}, None)
                self.assertEqual(raised.exception.category, expected)

    def test_generated_reply_rejects_unsupported_approval_claim(self):
        client, _ = fake_client("Your loan is approved for 90 lakh.")
        with self.assertRaises(LLMServiceError):
            GroqLLMService(client=client).generate_reply("yes", {}, {}, {}, {"action": "continue"})

    def test_safe_rate_disclaimer_passes_but_invented_rate_fails(self):
        GroqLLMService._validate_generated_reply(
            "I don't have verified rate details to share. We can continue with your LAP questions."
        )
        with self.assertRaises(LLMServiceError):
            GroqLLMService._validate_generated_reply("The interest rate is 12 percent per year.")

    def test_unclear_speech_gets_natural_repeat_and_state_is_unchanged(self):
        client, _ = fake_client("not valid JSON")
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        original = json.loads(json.dumps(state))
        reply, updated = process_response(state, "blargh what this", service)
        self.assertEqual(reply, "I'm sorry, I didn't quite catch that. Could you please repeat that?")
        self.assertEqual(updated, original)

    def test_clear_question_survives_invalid_analysis_output(self):
        client, _ = fake_client(
            "not valid JSON",
            "I’m a simulated Home Credit representative calling about a Loan Against Property offer. May I speak with Ravi?",
        )
        state, _ = new_conversation()
        reply, updated = process_response(state, "Who are you?", GroqLLMService(client=client))
        self.assertEqual(updated["stage"], "identity")
        self.assertFalse(updated["identity_verified"])
        self.assertIn("simulated Home Credit representative", reply)

    def test_explicit_callback_is_handled_even_if_analysis_would_be_invalid(self):
        client, calls = fake_client(
            "Certainly, I’ll note your request for a callback in 10 minutes. Goodbye."
        )
        state, _ = new_conversation()
        reply, updated = process_response(
            state, "I'm driving; can you call me after 10 minutes?", GroqLLMService(client=client)
        )
        self.assertEqual(len(calls.calls), 1)  # Direct callback recognition skips analysis.
        self.assertEqual(updated["outcome"], "callback_requested")
        self.assertEqual(updated["answers"]["callback_time"], "after 10 minutes")
        self.assertNotIn("scheduled", reply.lower())

    def test_extracts_facts_but_python_rules_disqualify_agricultural_property(self):
        payload = {"intent": "answer", "property_type": "agricultural"}
        client, _ = fake_client(json.dumps(payload), "I’m sorry, agricultural property does not meet this offer’s criteria. Thank you for your time.")
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        state["stage"] = "checklist"
        reply, updated = process_response(state, "My land is farmland", service)
        self.assertEqual(updated["outcome"], "DISQUALIFIED")
        self.assertIn("agricultural", updated["reason"].lower())
        self.assertEqual(len(client.chat.completions.calls), 2)

    def test_llm_amount_above_cap_is_handled_by_python_confirmation_branch(self):
        payload = {"intent": "answer", "requested_loan_amount_inr": 9_000_000}
        client, _ = fake_client(json.dumps(payload), "The maximum under this offer is 75 lakh. Would you like to proceed with 75 lakh?")
        state, _ = new_conversation()
        state.update(stage="checklist", answers={
            "existing_property_loan": False,
            "property_type": "residential",
            "ownership": "individual",
            "original_documents": True,
        })
        reply, updated = process_response(state, "Could I borrow 90 lakh?", GroqLLMService(client=client))
        self.assertEqual(updated["stage"], "amount_limit")
        self.assertEqual(updated["answers"]["amount_over_limit"], 9_000_000)
        self.assertIn("proceed", reply.lower())

    def test_llm_facts_are_saved_out_of_order_and_next_missing_item_is_asked(self):
        payload = {"intent": "answer", "property_type": "residential", "ownership": "joint",
                   "property_market_value_inr": 8_000_000}
        client, _ = fake_client(json.dumps(payload), "Thanks. Are the original property documents available for verification?")
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        state.update(stage="checklist", answers={"existing_property_loan": False})
        reply, updated = process_response(state, "It's a house owned jointly; value is 80 lakh", service)
        self.assertEqual(updated["answers"]["property_type"], "residential")
        self.assertEqual(updated["answers"]["ownership"], "joint")
        self.assertEqual(updated["answers"]["market_value"], 8_000_000)
        self.assertIn("original property documents", reply.lower())

    def test_ambiguous_fact_is_not_applied_and_existing_confirmed_value_is_preserved(self):
        payload = {"intent": "answer", "property_type": "residential", "clarification_needed": True,
                   "ambiguous_fields": ["property_type"], "clarification_question": "Could you confirm the property type?"}
        client, _ = fake_client(json.dumps(payload), "Could you confirm the property type?")
        state, _ = new_conversation()
        state.update(stage="checklist", answers={"existing_property_loan": False, "property_type": "commercial"})
        reply, updated = process_response(state, "Maybe residential, I'm not sure", GroqLLMService(client=client))
        self.assertEqual(updated["answers"]["property_type"], "commercial")
        self.assertEqual(reply, "Could you confirm the property type?")

    def test_explicit_correction_updates_a_previously_confirmed_fact(self):
        payload = {"intent": "answer", "property_type": "residential", "clarification_needed": False}
        client, _ = fake_client(json.dumps(payload), "Is the property individually or jointly owned?")
        state, _ = new_conversation()
        state.update(stage="checklist", answers={"existing_property_loan": False, "property_type": "commercial"})
        _reply, updated = process_response(state, "Actually, it is residential", GroqLLMService(client=client))
        self.assertEqual(updated["answers"]["property_type"], "residential")
        self.assertEqual(updated["stage"], "checklist")

    def test_public_status_never_contains_api_key(self):
        client, _ = fake_client()
        service = GroqLLMService(api_key="do-not-expose-this", client=client, model="test-model")
        self.assertEqual(service.public_status(), {"mode": "groq", "configured": True, "model": "test-model"})
        self.assertNotIn("do-not-expose-this", str(service.public_status()))

    def test_callback_has_priority_after_llm_analysis_and_uses_python_outcome(self):
        client, calls = fake_client(
            "Certainly, I’ll note your request for a callback in 10 minutes. Thank you, goodbye.",
        )
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        reply, updated = process_response(state, "Can you call me after 10 minutes?", service)
        self.assertEqual(updated["outcome"], "callback_requested")
        self.assertEqual(updated["answers"]["callback_time"], "after 10 minutes")
        self.assertNotIn("scheduled", reply.lower())
        self.assertEqual(len(calls.calls), 1)
        self.assertNotIn("response_format", calls.calls[0])
        self.assertIn("python_decision", calls.calls[0]["messages"][1]["content"])
        self.assertNotIn("python_approved_response", calls.calls[0]["messages"][1]["content"])

    def test_flask_uses_injected_service_and_keeps_response_shape(self):
        analysis = {"intent": "identity", "identity_confirmation": True}
        client, calls = fake_client(
            "Hello, may I speak with Ravi? This is a simulated call.",
            json.dumps(analysis), "Hello Ravi, is this a convenient time to talk?",
        )
        flask_app = create_app(GroqLLMService(client=client))
        http = flask_app.test_client()
        start = http.post("/api/conversation/start").get_json()
        result = http.post("/api/conversation/respond", json={"text": "Yes, speaking"}).get_json()
        self.assertIn("reply", result)
        self.assertIn("state", result)
        self.assertEqual(result["state"]["stage"], "availability")
        self.assertIn("convenient time", result["reply"])
        self.assertIn("reply", start)
        self.assertEqual(len(calls.calls), 3)

    def test_unconfigured_runtime_refuses_start_instead_of_using_scripted_replies(self):
        flask_app = create_app(GroqLLMService())
        http = flask_app.test_client()
        response = http.post("/api/conversation/start")
        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.get_json()["error"])

    def test_who_are_you_gets_safe_identity_explanation(self):
        client, _ = fake_client(
            json.dumps({"intent": "question"}),
            "I am a simulated Home Credit representative calling about a Loan Against Property offer. May I speak with Ravi?",
        )
        state, _ = new_conversation()
        reply, updated = process_response(state, "Who are you?", GroqLLMService(client=client))
        self.assertEqual(updated["stage"], "identity")
        self.assertIn("simulated Home Credit representative", reply)

    def test_identity_confirmation_with_call_purpose_question_then_availability(self):
        client, calls = fake_client(
            json.dumps({"intent": "question", "identity_confirmation": True}),
            "I’m calling about a Loan Against Property offer of up to 75 lakh, subject to checks and final approval. Is now a convenient time to talk?",
            json.dumps({"intent": "answer", "availability": "available", "identity_confirmation": False}),
            "Thank you. Would you like to learn more about the Loan Against Property offer of up to 75 lakh, subject to eligibility and final approval?",
        )
        service = GroqLLMService(client=client)
        state, _ = new_conversation()

        reply, state = process_response(state, "Yes, I am Ravi. Why did you call me?", service)
        self.assertEqual(state["stage"], "availability")
        self.assertTrue(state["identity_verified"])
        self.assertNotEqual(state["outcome"], "callback_requested")
        self.assertIn("Loan Against Property", reply)
        first_decision = json.loads(calls.calls[1]["messages"][1]["content"])["python_decision"]
        self.assertEqual(first_decision["action"], "explain_call_purpose_as_lap_offer_then_check_availability")

        reply, state = process_response(state, "Yeah, you can talk now", service)
        self.assertEqual(state["stage"], "interest")
        self.assertTrue(state["identity_verified"])
        self.assertEqual(state["availability"], "available")
        self.assertEqual(reply, "Thank you. Would you like to learn more about the Loan Against Property offer of up to 75 lakh, subject to eligibility and final approval?")

    def test_call_purpose_question_does_not_trigger_callback_even_if_llm_mislabels_intent(self):
        state, _ = new_conversation()
        self.assertFalse(_callback_takes_priority(
            "Yes, I am Ravi. Why did you call me?",
            {"intent": "callback", "identity_confirmation": True},
        ))

    def test_busy_then_free_resumes_availability_question(self):
        client, _ = fake_client(
            "What time would be convenient for a callback?",
            json.dumps({"intent": "answer", "availability": "available"}),
            "You may be eligible for up to 75 lakh, subject to checks and final approval. Are you interested?",
        )
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        state["stage"] = "availability"
        state["identity_verified"] = True
        reply, state = process_response(state, "I'm busy right now", service)
        self.assertEqual(state["stage"], "callback_time")
        self.assertIn("what time", reply.lower())
        reply, state = process_response(state, "I'm free now", service)
        self.assertEqual(state["stage"], "interest")
        self.assertIn("interested", reply.lower())
        self.assertNotIn("callback_return_stage", state)

    def test_model_error_does_not_mutate_conversation(self):
        class Broken:
            enabled = True

            def analyze_response(self, *_args):
                raise LLMServiceError("failure")

        state, _ = new_conversation()
        original = json.loads(json.dumps(state))
        reply, updated = process_response(state, "Yes, speaking", Broken())
        self.assertEqual(updated, original)
        self.assertIn("try again", reply)

    def test_generation_failure_does_not_commit_updated_conversation_state(self):
        class GenerationFailure:
            enabled = True

            def analyze_response(self, *_args):
                return {"intent": "identity", "identity_confirmation": True}

            def generate_reply(self, *_args):
                raise LLMServiceError("invalid draft", "invalid_output")

        state, _ = new_conversation()
        reply, updated = process_response(state, "yes I am Ravi was this", GenerationFailure())
        self.assertEqual(updated, state)
        self.assertIn("didn't quite catch that", reply)

    def test_generated_reply_is_used_and_not_given_a_prewritten_sentence(self):
        analysis = {"intent": "identity", "identity_confirmation": True}
        generated = "Thanks for confirming, Ravi. Is now a good time to speak?"
        client, calls = fake_client(json.dumps(analysis), generated)
        state, _ = new_conversation()
        reply, updated = process_response(state, "Yes, this is Ravi", GroqLLMService(client=client))
        self.assertEqual(reply, generated)
        self.assertEqual(updated["stage"], "availability")
        generation_context = json.loads(calls.calls[1]["messages"][1]["content"])
        self.assertIn("python_decision", generation_context)
        self.assertNotIn("approved_response", generation_context)

    def test_quota_error_is_explained_and_state_stays_unchanged(self):
        class QuotaFailure:
            enabled = True

            def analyze_response(self, *_args):
                raise LLMServiceError("provider detail omitted", "quota")

        state, _ = new_conversation()
        original = json.loads(json.dumps(state))
        reply, updated = process_response(state, "Yes, I am Ravi", QuotaFailure())
        self.assertIn("temporary issue", reply.lower())
        self.assertNotIn("quota", reply.lower())
        self.assertEqual(updated, original)

    def test_groq_provider_quota_failure_uses_temporary_message_and_logs_safe_category(self):
        class RateLimitError(Exception):
            status_code = 429
            body = {"error": {"code": "insufficient_quota", "message": "private provider detail"}}

        client, _ = fake_client(RateLimitError("private provider detail"))
        service = GroqLLMService(client=client)
        state, _ = new_conversation()
        original = json.loads(json.dumps(state))

        with self.assertLogs("backend.llm_service", level="WARNING") as logs:
            reply, updated = process_response(state, "Yes, I am Ravi", service)

        self.assertIn("temporary issue", reply.lower())
        self.assertNotIn("quota", reply.lower())
        self.assertEqual(updated, original)
        log_text = " ".join(logs.output)
        self.assertIn("category=quota", log_text)
        self.assertNotIn("private provider detail", log_text)

    def test_busy_without_time_uses_llm_and_stays_in_callback_time_stage(self):
        client, calls = fake_client(
            "Certainly. What time would be convenient for a callback?",
        )
        state, _ = new_conversation()
        state["stage"] = "availability"
        reply, updated = process_response(state, "I'm busy right now", GroqLLMService(client=client))
        self.assertEqual(updated["stage"], "callback_time")
        self.assertEqual(updated["callback_return_stage"], "availability")
        self.assertIn("what time", reply.lower())
        self.assertEqual(len(calls.calls), 1)


if __name__ == "__main__":
    unittest.main()
