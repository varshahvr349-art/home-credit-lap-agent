"""Smoke tests for the API used by both typed and spoken responses."""

import unittest

from app import app
from backend.llm_service import GroqLLMService


class VoiceConversationApiTests(unittest.TestCase):
    def setUp(self):
        # Existing conversation tests exercise deterministic rules and must never
        # send live requests just because a developer has GROQ_API_KEY configured.
        app.extensions["llm_service"] = GroqLLMService()
        app.testing = True
        self.client = app.test_client()

    def start(self):
        return self.client.post("/api/conversation/start")

    def reply(self, text):
        return self.client.post("/api/conversation/respond", json={"text": text})

    def enter_checklist(self):
        self.start()
        for text in ("Yes, speaking", "Yes, convenient", "Yes, interested", "No existing loan"):
            response = self.reply(text)
            self.assertEqual(response.status_code, 200)
        return response.get_json()

    def enter_availability(self):
        self.start()
        return self.reply("Yes, speaking")

    def force_stage(self, stage):
        with self.client.session_transaction() as flask_session:
            state = flask_session["conversation"]
            state["stage"] = stage
            flask_session["conversation"] = state

    def test_homepage_includes_voice_controls_and_script(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"microphone-button", response.data)
        self.assertIn(b"static/js/app.js", response.data)
        self.assertIn(b'id="start-button" type="button">Start simulated conversation</button>', response.data)

    def test_start_button_starts_simulation(self):
        response = self.start()
        self.assertEqual(response.status_code, 200)
        self.assertIn("may I speak with Ravi", response.get_json()["reply"])

    def test_busy_and_call_after_ten_minutes_finishes_without_reasking(self):
        self.enter_availability()
        response = self.reply("No, I am busy. Can you call after 10 minutes?")
        result = response.get_json()
        self.assertEqual(result["state"]["outcome"], "callback_requested")
        self.assertEqual(result["state"]["answers"]["callback_time"], "after 10 minutes")
        self.assertNotIn("what time", result["reply"].lower())
        self.assertEqual(result["reply"], "Certainly. I'll note your request for a callback in 10 minutes. Thank you for your time. Goodbye.")
        self.assertNotIn("scheduled", result["reply"].lower())
        self.assertEqual(self.reply("Thanks").get_json()["state"]["outcome"], "callback_requested")

    def test_busy_and_tomorrow_at_eleven_saves_time_and_finishes(self):
        self.enter_availability()
        result = self.reply("I'm busy, call me tomorrow at 11 AM").get_json()
        self.assertEqual(result["state"]["answers"]["callback_time"], "tomorrow at 11 am")
        self.assertEqual(result["state"]["outcome"], "callback_requested")
        self.assertNotIn("scheduled", result["reply"].lower())

    def test_driving_after_identity_confirmation_pauses_and_asks_callback_preference(self):
        self.start()
        result = self.reply("Yes, I am driving").get_json()
        state = result["state"]
        self.assertEqual(result["reply"], "I understand, Ravi. Since you're driving, I don't want to distract you. Would you prefer a callback later?")
        self.assertTrue(state["identity_verified"])
        self.assertEqual(state["driving_status"], "driving")
        self.assertIsNone(state["availability"])
        self.assertEqual(state["callback_status"], "not_requested")
        self.assertEqual(state["stage"], "driving_check")
        self.assertIsNone(state["outcome"])

    def test_driving_then_provided_callback_time_records_request_and_ends(self):
        self.start()
        self.reply("Yes, I am driving")
        result = self.reply("Call me in 20 minutes").get_json()
        self.assertEqual(result["state"]["driving_status"], "driving")
        self.assertEqual(result["state"]["answers"]["callback_time"], "in 20 minutes")
        self.assertEqual(result["state"]["callback_status"], "requested")
        self.assertEqual(result["state"]["outcome"], "callback_requested")
        self.assertNotIn("scheduled", result["reply"].lower())

    def test_driving_with_hands_free_is_not_misclassified_as_callback(self):
        self.enter_availability()
        result = self.reply("I'm driving but can talk using hands-free").get_json()
        self.assertEqual(result["state"]["driving_status"], "hands_free")
        self.assertEqual(result["state"]["availability"], "available")
        self.assertEqual(result["state"]["stage"], "interest")
        self.assertNotEqual(result["state"]["outcome"], "callback_requested")

    def test_parked_customer_can_resume_after_driving_safety_pause(self):
        self.enter_availability()
        self.reply("I'm driving")
        result = self.reply("I'm parked now").get_json()
        self.assertEqual(result["state"]["driving_status"], "parked")
        self.assertEqual(result["state"]["availability"], "available")
        self.assertEqual(result["state"]["stage"], "interest")
        self.assertNotEqual(result["state"]["outcome"], "callback_requested")

    def test_busy_without_time_asks_once_and_follow_up_time_finishes(self):
        self.enter_availability()
        first = self.reply("I'm busy; call me later").get_json()
        self.assertEqual(first["state"]["stage"], "callback_time")
        self.assertIsNone(first["state"]["outcome"])
        self.assertIn("what time", first["reply"].lower())

        second = self.reply("You can call back in 15 minutes").get_json()
        self.assertEqual(second["state"]["answers"]["callback_time"], "in 15 minutes")
        self.assertEqual(second["state"]["outcome"], "callback_requested")
        self.assertNotIn("what time", second["reply"].lower())

    def test_callback_intent_takes_priority_across_conversation_stages(self):
        for stage in ("greeting", "identity", "interest", "checklist"):
            with self.subTest(stage=stage):
                self.client = app.test_client()
                self.start()
                self.force_stage(stage)
                result = self.reply("Can you call me after 10 minutes?").get_json()
                self.assertEqual(result["state"]["outcome"], "callback_requested")
                self.assertEqual(result["state"]["answers"]["callback_time"], "after 10 minutes")
                self.assertNotIn("interested", result["reply"].lower())
                self.assertNotIn("what time", result["reply"].lower())

    def test_call_purpose_question_is_not_misread_as_callback_and_identity_persists(self):
        self.start()
        first = self.reply("Yes, I am Ravi. Why did you call me?").get_json()
        self.assertEqual(first["state"]["stage"], "availability")
        self.assertTrue(first["state"]["identity_verified"])
        self.assertIsNone(first["state"]["answers"].get("callback_time"))
        self.assertNotEqual(first["state"]["outcome"], "callback_requested")

        second = self.reply("Yeah, you can talk now").get_json()
        self.assertTrue(second["state"]["identity_verified"])
        self.assertEqual(second["state"]["availability"], "available")
        self.assertEqual(second["state"]["stage"], "interest")
        self.assertNotEqual(second["state"]["outcome"], "IDENTITY_NOT_VERIFIED")

    def test_callback_time_is_not_asked_again_after_follow_up(self):
        self.enter_availability()
        first = self.reply("I'm busy").get_json()
        self.assertIn("What time would be convenient", first["reply"])
        second = self.reply("After lunch").get_json()
        self.assertEqual(second["state"]["answers"]["callback_time"], "after lunch")
        third = self.reply("Is it noted?").get_json()
        self.assertNotIn("callback time", third["reply"].lower())
        self.assertEqual(third["state"]["outcome"], "callback_requested")

    def test_respond_uses_flow_and_completes_eligible_simulation(self):
        self.assertEqual(self.start().status_code, 200)
        for answer in (
            "Yes, speaking", "Yes", "Yes, interested", "No existing loan",
            "Residential", "Joint ownership", "Originals are safely at home",
            "50 lakh", "Salaried", "Bank", "80 lakh", "15 years",
        ):
            response = self.reply(answer)
            self.assertEqual(response.status_code, 200, answer)
        result = response.get_json()
        self.assertEqual(result["state"]["outcome"], "QUALIFIED_FOR_EXPERT")
        self.assertEqual(result["state"]["answers"]["ownership"], "joint")
        self.assertEqual(result["state"]["answers"]["market_value"], 8_000_000)

    def test_agricultural_property_disqualifies(self):
        self.start()
        for answer in ("Yes", "Yes", "Yes", "No", "Agricultural"):
            response = self.reply(answer)
        result = response.get_json()
        self.assertEqual(result["state"]["outcome"], "DISQUALIFIED")

    def test_over_limit_amount_offers_cap_before_continuing(self):
        self.start()
        for answer in ("Yes", "Yes", "Yes", "No", "Residential", "Individual", "Yes", "90 lakh"):
            response = self.reply(answer)
        self.assertIn("proceed with ₹75 lakh", response.get_json()["reply"])
        response = self.reply("Yes, proceed")
        self.assertEqual(response.get_json()["state"]["answers"]["loan_amount"], 7_500_000)

    def test_empty_or_unstarted_response_is_rejected(self):
        self.assertEqual(self.reply("hello").status_code, 409)
        self.start()
        self.assertEqual(self.client.post("/api/conversation/respond", json={"text": "  "}).status_code, 400)

    def test_flask_session_remembers_state_across_requests(self):
        self.start()
        self.reply("Yes, speaking")
        response = self.reply("Yes, it is convenient")
        self.assertIn("up to", response.get_json()["reply"].lower())
        response = self.reply("Yes, I'm interested")
        self.assertEqual(response.get_json()["state"]["stage"], "existing_loan")

    def test_out_of_order_answers_are_remembered_and_skipped(self):
        self.enter_checklist()
        response = self.reply("It's a residential house, jointly owned with my brother, market value around 80 lakh")
        state = response.get_json()["state"]
        self.assertEqual(state["answers"]["property_type"], "residential")
        self.assertEqual(state["answers"]["ownership"], "joint")
        self.assertEqual(state["answers"]["market_value"], 8_000_000)
        self.assertIn("original property documents", response.get_json()["reply"].lower())
        self.assertNotIn("What type of property", response.get_json()["reply"])

    def test_informational_interruption_preserves_answers_and_next_question(self):
        self.enter_checklist()
        self.reply("Residential")
        before = self.reply("Joint ownership").get_json()["state"]
        response = self.reply("What is the maximum loan amount?")
        after = response.get_json()["state"]
        self.assertEqual(after["answers"], before["answers"])
        self.assertIn("maximum", response.get_json()["reply"].lower())
        self.assertIn("original property documents", response.get_json()["reply"].lower())

    def test_agricultural_property_stops_flow_and_later_reply_cannot_continue(self):
        self.enter_checklist()
        response = self.reply("Agricultural")
        state = response.get_json()["state"]
        self.assertEqual(state["outcome"], "DISQUALIFIED")
        stopped = self.reply("Actually, residential")
        self.assertEqual(stopped.get_json()["state"], state)

    def test_photocopies_cash_income_and_invalid_tenure_disqualify(self):
        self.enter_checklist()
        self.reply("Residential")
        self.reply("Individual")
        response = self.reply("Only photocopies")
        self.assertEqual(response.get_json()["state"]["outcome"], "DISQUALIFIED")

        self.client = app.test_client()
        self.enter_checklist()
        for answer in ("Residential", "Individual", "Originals safely at home", "50 lakh"):
            response = self.reply(answer)
        response = self.reply("Salaried, cash income")
        self.assertEqual(response.get_json()["state"]["outcome"], "DISQUALIFIED")

        self.client = app.test_client()
        self.enter_checklist()
        for answer in ("Residential", "Individual", "Originals at home", "50 lakh", "Salaried", "Bank", "Market value 1 lakh"):
            response = self.reply(answer)
        response = self.reply("2 years")
        self.assertEqual(response.get_json()["state"]["outcome"], "DISQUALIFIED")

    def test_market_value_has_no_invented_minimum(self):
        self.enter_checklist()
        for answer in (
            "Residential", "Individual", "Originals at home", "50 lakh",
            "Salaried", "Bank", "Market value 1 lakh", "15 years",
        ):
            response = self.reply(answer)
        self.assertEqual(response.get_json()["state"]["outcome"], "QUALIFIED_FOR_EXPERT")

    def test_over_limit_offer_asks_before_accepting_or_declining(self):
        self.enter_checklist()
        for answer in ("Residential", "Individual", "Originals at home", "I need 90 lakh"):
            response = self.reply(answer)
        state = response.get_json()["state"]
        self.assertEqual(state["stage"], "amount_limit")
        self.assertEqual(state["status"], "active")
        self.assertIn("proceed with", response.get_json()["reply"])
        response = self.reply("No")
        self.assertEqual(response.get_json()["state"]["outcome"], "CUSTOMER_WITHDREW")

    def test_existing_loan_and_emi_reduction_route_before_checklist(self):
        self.start()
        for answer in ("Yes", "Yes", "Yes", "I already have a loan against the property"):
            response = self.reply(answer)
        self.assertEqual(response.get_json()["state"]["outcome"], "LOAN_TRANSFER_REFERRAL")

        self.client = app.test_client()
        self.start()
        for answer in ("Yes", "Yes", "Yes", "No existing loan"):
            self.reply(answer)
        response = self.reply("I want to reduce my EMI")
        self.assertEqual(response.get_json()["state"]["outcome"], "LOAN_TRANSFER_REFERRAL")

    def test_unsupported_browser_fallback_is_defined_in_client_script(self):
        script = self.client.get("/static/js/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"Speech recognition is unsupported in this browser", script.data)
        self.assertIn(b"responseInput.disabled = !canRespond", script.data)
        script.close()


if __name__ == "__main__":
    unittest.main()
