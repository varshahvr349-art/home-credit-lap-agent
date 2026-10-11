"""Small Flask entry point for the Home Credit LAP voice-agent prototype."""

import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

from backend.conversation_manager import new_conversation, process_response
from backend.llm_service import GroqLLMService

logger = logging.getLogger(__name__)

load_dotenv()




def create_app(llm_service: GroqLLMService | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "local-simulation-only-change-before-deployment"
    active_llm_service = llm_service or GroqLLMService.from_environment()
    app.extensions["llm_service"] = active_llm_service
    
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            llm_status=app.extensions["llm_service"].public_status()
        )

    
    @app.get("/api/llm/status")
    def llm_status():
        """Report configured mode without exposing credentials or claiming connectivity."""
        return jsonify(app.extensions["llm_service"].public_status())

    @app.post("/api/conversation/start")
    def start_conversation():
        """Start a fictional, local conversation. This never places a call."""
        state, reply = new_conversation()
        service = app.extensions["llm_service"]
        if service.enabled:
            try:
                reply = service.generate_reply(
                    "", state, state, {},
                    {"action": "open_simulated_call_and_verify_intended_customer",
                     "customer_name": state.get("customer_name")},
                )
            except Exception as exc:
                # Do not start or store a conversation when the configured LLM
                # could not generate its opening turn.
                logger.warning("Conversation opening generation failed category=%s",
                               getattr(exc, "category", type(exc).__name__))
                return jsonify({"error": "The AI reply could not be generated. Please try starting again."}), 503
        elif not app.testing:
            return jsonify({"error": "Groq is not configured. Add GROQ_API_KEY to .env before starting a conversation."}), 503
        session["conversation"] = state
        return jsonify({"reply": reply, "state": state})

    @app.post("/api/conversation/respond")
    def respond():
        """Send typed or browser-recognized speech through the same flow."""
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Please provide a response."}), 400
        state = session.get("conversation")
        if not state:
            return jsonify({"error": "Start a simulated conversation first."}), 409

        service = app.extensions["llm_service"]
        # Automated deterministic API tests deliberately inject no LLM; runtime
        # requests never silently fall back to the fixed-rule dialogue.
        turn_service = None if app.testing and not service.enabled else service
        reply, updated_state = process_response(state, text.strip(), llm_service=turn_service)
        session["conversation"] = updated_state
        return jsonify({"reply": reply, "state": updated_state})

    return app


app = create_app()


if __name__ == "__main__":
    # Debug mode is useful locally; do not use it for a public deployment.
    app.run(debug=True)
