# Home Credit LAP Voice Agent

A local, simulated prototype for a Loan Against Property qualification voice agent. The browser dashboard supports speech input through the Web Speech API where available, with typing as a fallback, and reads replies aloud with browser speech synthesis. Python owns conversation state, routing, and eligibility decisions. In Groq mode, Groq extracts customer facts and generates the actual spoken reply from the latest utterance, confirmed state, and a structured Python decision. It never places a telephone call.

## Requirements

- Python 3.10 or later
- Windows PowerShell or the VS Code terminal

## Run locally

From this project folder, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env locally and set GROQ_API_KEY. Never share or commit the key.
py app.py
```

Open <http://127.0.0.1:5000/> in your browser. Stop the local server with `Ctrl+C` in the terminal.

Set `GROQ_API_KEY` in `.env` to run the conversation. Without it, the app refuses to start a customer simulation instead of presenting fixed Python dialogue as AI-generated. Deterministic conversation rules remain available to the automated tests. Set `GROQ_MODEL` to a model enabled for your Groq account; the default is `openai/gpt-oss-20b`. `/api/llm/status` reports configuration only and does not verify network connectivity. The key is read from `.env` and is never sent to the browser. When Groq is unavailable or its output fails validation, the app returns an honest recovery response and does not commit the current turn's state changes.

## Try the voice simulation

1. Start Flask and open the URL above. Modern browsers treat `localhost` as a secure context for microphone permissions.
2. Press **Start simulated conversation**. The browser reads the agent greeting aloud if speech synthesis is available.
3. Press the microphone button, allow microphone access when prompted, and speak a response. Recognized words appear in the transcript and are submitted through the same Flask route used by typed text.
4. Wait for processing and listen to the next agent prompt. You can also type a reply and press Enter or the arrow button.

Speech recognition support varies by browser, operating system, and language settings. Current Chrome and Edge commonly expose `SpeechRecognition`/`webkitSpeechRecognition`; Firefox and some configurations may not. Recognition may rely on a browser vendor's online speech service, so offline recognition is not guaranteed. If the microphone is denied, unavailable, or unsupported, the page explains the issue and typed replies remain available. Speech output uses installed browser voices and varies by device. Recognition is turn-based; it does not provide real-time interruption handling.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Groq integration boundaries

The service in `backend/llm_service.py` calls Groq Chat Completions for structured extraction (`analyze_response`) and original reply generation (`generate_reply`). `backend/conversation_manager.py` validates and merges extracted facts, then the existing Python eligibility rules, callback handling, and transfer routing produce a structured next-action decision. That decision, rather than a prewritten reply sentence, is sent to Groq. Terminal outcomes are determined by Python and worded by Groq. Callback requests are recorded only; this prototype does not schedule callbacks. Groq failures return a category-specific recovery message and leave the current conversation state unchanged. Logs show request stage, result parsing, and safe error categories without message text or credentials.

To verify live Groq generation, set `GROQ_API_KEY` in your private `.env`, restart Flask, and start a conversation. Speak or type **“Yes, I am Ravi”** and check the terminal: it should show both `operation=analysis` and `operation=reply_generation`. The reply should be natural wording from that call and will also be spoken by the browser. Mocked tests prove the generation function is invoked and its reply is returned; they do not prove live account access or network connectivity.

## Scope and limitations

Conversation state uses a small Flask session and is not persisted in SQLite. Browser speech support varies by browser and operating system. Groq use requires an account/API key and may incur provider charges. No live Groq request is needed for automated tests. Never put credentials in source control or paste them into chat.
