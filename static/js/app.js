(() => {
  const startButton = document.getElementById('start-button');
  const responseInput = document.getElementById('customer-response');
  const sendButton = document.getElementById('send-button');
  const microphoneButton = document.getElementById('microphone-button');
  const doneSpeakingButton = document.getElementById('done-speaking-button');
  const transcript = document.getElementById('transcript');
  const emptyState = document.getElementById('transcript-empty');
  const statusPill = document.getElementById('conversation-status');
  const voiceStatus = document.getElementById('voice-status');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  // Override window.LAP_SILENCE_TIMEOUT_MS before this script loads to tune the pause cutoff.
  const SILENCE_TIMEOUT_MS = Number(window.LAP_SILENCE_TIMEOUT_MS) || 1800;
  let recognition = null;
  let active = false;
  let busy = false;
  let agentSpeaking = false;
  let voiceSession = false;
  let voiceSubmitted = false;
  let finalSegments = [];
  let silenceTimer = null;
  let recognitionFailed = false;
  let speechGeneration = 0;

  function setVoiceStatus(message, state = 'ready') {
    voiceStatus.textContent = message;
    voiceStatus.dataset.state = state;
  }

  function addMessage(who, message) {
    emptyState.hidden = true;
    const bubble = document.createElement('div');
    bubble.className = `transcript-message ${who}`;
    bubble.textContent = message;
    bubble.setAttribute('aria-label', `${who === 'agent' ? 'Agent' : 'Customer'} said: ${message}`);
    transcript.appendChild(bubble);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function updateControls() {
    const canRespond = active && !busy;
    responseInput.disabled = !canRespond || Boolean(voiceSession);
    sendButton.disabled = !canRespond || Boolean(voiceSession);
    microphoneButton.disabled = !canRespond || !recognition || agentSpeaking || Boolean(voiceSession);
    doneSpeakingButton.disabled = !canRespond || !voiceSession;
  }

  function speak(message) {
    if (!('speechSynthesis' in window) || !window.SpeechSynthesisUtterance) {
      agentSpeaking = false;
      setVoiceStatus('Text-to-speech is unavailable in this browser. You can still read the transcript.');
      updateControls();
      return;
    }
    const speechId = ++speechGeneration;
    const synthesizer = window.speechSynthesis;
    try {
      // Do not clear an empty queue before the greeting; some browsers can
      // suppress the first utterance after an unnecessary cancel call.
      if (synthesizer.speaking || synthesizer.pending) synthesizer.cancel();
      // A canceled utterance can leave the browser queue paused, especially
      // after a completed conversation is restarted.
      synthesizer.resume();
    } catch (_error) {
      // Continue with the browser's regular speak path when resume is unavailable.
    }
    agentSpeaking = true;
    updateControls();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'en-IN';
    utterance.onstart = () => {
      if (speechId !== speechGeneration) return;
      setVoiceStatus('Speaking response…', 'speaking');
    };
    utterance.onend = () => {
      if (speechId !== speechGeneration) return;
      agentSpeaking = false;
      setVoiceStatus('Ready');
      updateControls();
    };
    utterance.onerror = event => {
      if (speechId !== speechGeneration) return;
      if (event?.error === 'canceled' || event?.error === 'interrupted') {
        agentSpeaking = false;
        setVoiceStatus('Ready');
        updateControls();
        return;
      }
      agentSpeaking = false;
      setVoiceStatus('Could not play speech. The response is still available in the transcript.');
      updateControls();
    };
    try {
      synthesizer.speak(utterance);
      synthesizer.resume();
    } catch (_error) {
      if (speechId !== speechGeneration) return;
      agentSpeaking = false;
      setVoiceStatus('Could not play speech. The response is still available in the transcript.');
      updateControls();
    }
  }

  async function submitResponse(text) {
    const cleanText = text.trim();
    if (voiceSession && !voiceSubmitted) {
      setVoiceStatus('Finish the current speech turn before sending another response.', 'listening');
      return false;
    }
    if (!cleanText || busy || !active) {
      if (!cleanText && active) setVoiceStatus('No final speech was detected. Try again or type your response.');
      return false;
    }
    busy = true;
    updateControls();
    setVoiceStatus('Processing your response…', 'processing');
    addMessage('customer', cleanText);
    responseInput.value = '';
    try {
      const result = await fetch('/api/conversation/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: cleanText })
      });
      const data = await result.json();
      if (!result.ok) throw new Error(data.error || 'The response could not be processed.');
      addMessage('agent', data.reply);
      if (data.state.status !== 'active') {
        active = false;
        statusPill.textContent = data.state.outcome.replaceAll('_', ' ');
      }
      speak(data.reply);
      return true;
    } catch (error) {
      setVoiceStatus(error.message || 'A connection error occurred. Please try again.');
      return false;
    } finally {
      busy = false;
      updateControls();
      if (active && !agentSpeaking) responseInput.focus();
    }
  }

  function clearSilenceTimer() {
    if (silenceTimer !== null) window.clearTimeout(silenceTimer);
    silenceTimer = null;
  }

  function bufferedFinalTranscript() {
    return finalSegments.filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
  }

  function showRecognitionTranscript(interim = '') {
    responseInput.value = [bufferedFinalTranscript(), interim.trim()].filter(Boolean).join(' ');
  }

  function finishVoiceTurn(reason) {
    if (!voiceSession || voiceSubmitted) return;
    voiceSubmitted = true;
    voiceSession = false;
    clearSilenceTimer();
    microphoneButton.classList.remove('listening');
    microphoneButton.setAttribute('aria-label', 'Speak response');
    const utterance = bufferedFinalTranscript();
    responseInput.value = '';
    updateControls();
    if (!utterance) {
      if (!(reason === 'natural-end' && recognitionFailed)) {
        setVoiceStatus('No final speech was detected. Try speaking again or type your response.');
      }
      return;
    }
    // Mark submitted before stopping; onend may fire synchronously in browsers and test doubles.
    if (reason !== 'natural-end' && recognition) {
      try { recognition.stop(); } catch (_error) { /* Already stopped by the browser. */ }
    }
    submitResponse(utterance);
  }

  function scheduleSilenceSubmit() {
    clearSilenceTimer();
    silenceTimer = window.setTimeout(() => finishVoiceTurn('silence'), SILENCE_TIMEOUT_MS);
  }

  startButton.addEventListener('click', async () => {
    startButton.disabled = true;
    setVoiceStatus('Starting simulated conversation…', 'processing');
    try {
      const result = await fetch('/api/conversation/start', { method: 'POST' });
      const data = await result.json();
      if (!result.ok) throw new Error(data.error || 'Could not start the conversation.');
      transcript.replaceChildren();
      active = true;
      statusPill.textContent = 'Active simulation';
      updateControls();
      addMessage('agent', data.reply);
      speak(data.reply);
      if (!recognition) setVoiceStatus('Speech recognition is unsupported here. Type a reply instead.');
    } catch (error) {
      setVoiceStatus(error.message || 'Could not start the conversation.');
    } finally {
      startButton.disabled = false;
    }
  });

  sendButton.addEventListener('click', () => submitResponse(responseInput.value));
  responseInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') submitResponse(responseInput.value);
  });

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      voiceSession = true;
      voiceSubmitted = false;
      recognitionFailed = false;
      finalSegments = [];
      clearSilenceTimer();
      responseInput.value = '';
      microphoneButton.classList.add('listening');
      microphoneButton.setAttribute('aria-label', 'Listening');
      setVoiceStatus('Listening… speak now.', 'listening');
      updateControls();
    };
    recognition.onresult = event => {
      if (!voiceSession || voiceSubmitted) return;
      let interim = '';
      let gotSpeech = false;
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const phrase = result[0]?.transcript || '';
        if (phrase.trim()) gotSpeech = true;
        if (result.isFinal && phrase.trim()) finalSegments[i] = phrase.trim();
        else if (!result.isFinal) interim += phrase;
      }
      showRecognitionTranscript(interim);
      if (gotSpeech) {
        clearSilenceTimer();
        if (bufferedFinalTranscript()) scheduleSilenceSubmit();
      }
    };
    recognition.onerror = event => {
      recognitionFailed = true;
      voiceSession = false;
      voiceSubmitted = true;
      clearSilenceTimer();
      microphoneButton.classList.remove('listening');
      microphoneButton.setAttribute('aria-label', 'Speak response');
      responseInput.value = '';
      const explanations = {
        'not-allowed': 'Microphone access was blocked. Allow microphone permission in your browser settings, or type your response.',
        'service-not-allowed': 'Speech recognition is blocked by the browser. You can type your response instead.',
        'audio-capture': 'No microphone was found. Connect a microphone or type your response.',
        'no-speech': 'No speech was detected. Try speaking a little louder, or type your response.',
        'network': 'The browser speech service had a network error. Try again or type your response.'
      };
      setVoiceStatus(explanations[event.error] || `Speech recognition failed (${event.error}). You can type your response.`);
      updateControls();
    };
    recognition.onend = () => {
      microphoneButton.classList.remove('listening');
      microphoneButton.setAttribute('aria-label', 'Speak response');
      // Browsers can end capture unexpectedly. Submit only accumulated final results;
      // a preceding timer or Done Speaking action is deduplicated by voiceSubmitted.
      if (voiceSession && !voiceSubmitted) finishVoiceTurn('natural-end');
      else updateControls();
    };
    microphoneButton.addEventListener('click', () => {
      if (busy || agentSpeaking || voiceSession) return;
      try {
        recognition.start();
      } catch (_error) {
        setVoiceStatus('The microphone could not start. Check permission and try again, or type your response.');
      }
    });
    doneSpeakingButton.addEventListener('click', () => finishVoiceTurn('done'));
  } else {
    microphoneButton.disabled = true;
    doneSpeakingButton.disabled = true;
    microphoneButton.title = 'Speech recognition is not supported by this browser';
    setVoiceStatus('Speech recognition is unsupported in this browser. Typed responses are available.');
  }
})();
