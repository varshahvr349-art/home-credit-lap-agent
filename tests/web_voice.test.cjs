const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'static', 'js', 'app.js'), 'utf8');

class FakeElement {
  constructor() {
    this.listeners = {};
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.classes = new Set();
    this.classList = {
      add: name => this.classes.add(name),
      remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
    };
    this.disabled = false;
    this.value = '';
    this.textContent = '';
    this.hidden = false;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  async trigger(name, event = {}) { return this.listeners[name]?.(event); }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute(name, value) { this.attributes[name] = value; }
  focus() {}
}

function makeHarness({ recognitionSupported = true, respondState = 'active', silenceTimeout = 1750 } = {}) {
  const ids = [
    'start-button', 'customer-response', 'send-button', 'microphone-button', 'done-speaking-button',
    'transcript', 'transcript-empty', 'conversation-status', 'voice-status',
  ];
  const elements = Object.fromEntries(ids.map(id => [id, new FakeElement()]));
  const calls = [];
  const utterances = [];
  const timers = new Map();
  let nextTimer = 1;
  let recognition;
  const speechSynthesis = {
    speaking: false,
    paused: false,
    resumeCalls: 0,
    cancelCalls: 0,
    cancel() {
      this.cancelCalls += 1;
      utterances.at(-1)?.onerror?.({ error: 'canceled' });
      this.speaking = false;
      this.paused = false;
    },
    resume() { this.paused = false; this.resumeCalls += 1; },
    speak(utterance) {
      utterances.push(utterance);
      this.speaking = true;
      utterance.onstart?.();
    },
  };
  const setTimeout = (callback, delay) => {
    const id = nextTimer++;
    timers.set(id, { callback, delay });
    return id;
  };
  const clearTimeout = id => timers.delete(id);

  class MockRecognition {
    constructor() { recognition = this; }
    start() { this.onstart?.(); }
    stop() { this.onend?.(); }
  }
  class MockUtterance { constructor(text) { this.text = text; } }

  const window = {
    ...(recognitionSupported ? { SpeechRecognition: MockRecognition } : {}),
    SpeechSynthesisUtterance: MockUtterance,
    LAP_SILENCE_TIMEOUT_MS: silenceTimeout,
    setTimeout,
    clearTimeout,
    speechSynthesis,
  };
  const document = {
    getElementById: id => elements[id],
    createElement: () => new FakeElement(),
  };
  const fetch = async (url, options = {}) => {
    calls.push({ url, options, body: options.body ? JSON.parse(options.body) : null });
    const isStart = url.endsWith('/start');
    return {
      ok: true,
      async json() {
        return isStart
          ? { reply: 'Hello, may I speak with Ravi?', state: { status: 'active' } }
          : { reply: 'Is this a convenient time to talk?', state: { status: respondState, outcome: 'TEST' } };
      },
    };
  };
  vm.runInNewContext(source, { window, document, fetch, SpeechSynthesisUtterance: MockUtterance });
  return {
    elements, calls, utterances, speechSynthesis,
    get recognition() { return recognition; },
    get timers() { return [...timers.values()]; },
    get timerIds() { return [...timers.keys()]; },
    fireLatestTimer() {
      const last = [...timers.entries()].at(-1);
      if (!last) return false;
      timers.delete(last[0]);
      last[1].callback();
      return true;
    },
    fireStaleTimer(timer) { timer.callback(); },
  };
}

const startAndListen = async harness => {
  await harness.elements['start-button'].trigger('click');
  // End the greeting utterance so microphone capture can begin.
  harness.utterances.at(-1).onend?.();
  harness.elements['microphone-button'].trigger('click');
};
const waitForRequest = async () => new Promise(resolve => setImmediate(resolve));
const finalResult = transcript => Object.assign([{ transcript }], { isFinal: true });
const interimResult = transcript => Object.assign([{ transcript }], { isFinal: false });

test('interim speech is displayed but never submitted', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onresult({ resultIndex: 0, results: [interimResult('I am busy right now')] });
  assert.equal(harness.elements['customer-response'].value, 'I am busy right now');
  assert.equal(harness.calls.filter(call => call.url.endsWith('/respond')).length, 0);
  assert.equal(harness.timers.length, 0);
});

test('multiple final segments accumulate and submit as one complete utterance after silence', async () => {
  const harness = makeHarness({ silenceTimeout: 1800 });
  await startAndListen(harness);
  const first = [finalResult('I am busy right now,')];
  harness.recognition.onresult({ resultIndex: 0, results: first });
  const timerBeforePause = harness.timers[0];
  const timerIdBeforePause = harness.timerIds[0];
  assert.equal(timerBeforePause.delay, 1800);
  const withInterim = [...first, interimResult('can you call me after ten minutes')];
  harness.recognition.onresult({ resultIndex: 1, results: withInterim });
  assert.equal(harness.timers.length, 1);
  assert.notEqual(harness.timerIds[0], timerIdBeforePause);
  const complete = [...first, finalResult('can you call me after 10 minutes?')];
  harness.recognition.onresult({ resultIndex: 1, results: complete });
  harness.fireLatestTimer();
  await waitForRequest();
  const requests = harness.calls.filter(call => call.url.endsWith('/respond'));
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0].body, { text: 'I am busy right now, can you call me after 10 minutes?' });
});

test('silence timer resets after later speech and onend cannot duplicate it', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onresult({ resultIndex: 0, results: [finalResult('I am busy')] });
  const oldTimer = harness.timers[0];
  harness.recognition.onresult({ resultIndex: 1, results: [finalResult('I am busy'), interimResult('call me after 10 minutes')] });
  harness.recognition.onresult({ resultIndex: 1, results: [finalResult('I am busy'), finalResult('call me after 10 minutes')] });
  harness.recognition.onend();
  harness.fireStaleTimer(oldTimer);
  await waitForRequest();
  assert.equal(harness.calls.filter(call => call.url.endsWith('/respond')).length, 1);
  assert.deepEqual(harness.calls.find(call => call.url.endsWith('/respond')).body,
    { text: 'I am busy call me after 10 minutes' });
});

test('empty result and silence do not submit an empty request', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onend();
  await waitForRequest();
  assert.equal(harness.calls.filter(call => call.url.endsWith('/respond')).length, 0);
  assert.match(harness.elements['voice-status'].textContent, /No final speech/);
});

test('Done Speaking submits buffered finals once and onend cannot duplicate', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onresult({ resultIndex: 0, results: [finalResult('Yes, speaking')] });
  await harness.elements['done-speaking-button'].trigger('click');
  await waitForRequest();
  assert.equal(harness.calls.filter(call => call.url.endsWith('/respond')).length, 1);
  assert.deepEqual(harness.calls.find(call => call.url.endsWith('/respond')).body, { text: 'Yes, speaking' });
});

test('agent reply finishes before microphone can start the next listening turn', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onresult({ resultIndex: 0, results: [finalResult('Yes, speaking')] });
  harness.elements['done-speaking-button'].trigger('click');
  await waitForRequest();
  assert.equal(harness.elements['microphone-button'].disabled, true);
  assert.equal(harness.elements['voice-status'].dataset.state, 'speaking');
  harness.utterances.at(-1).onend();
  assert.equal(harness.elements['microphone-button'].disabled, false);
  harness.elements['microphone-button'].trigger('click');
  assert.equal(harness.elements['voice-status'].dataset.state, 'listening');
});

test('recognition errors stay visible and typed input remains available', async () => {
  const harness = makeHarness();
  await harness.elements['start-button'].trigger('click');
  harness.recognition.onerror({ error: 'not-allowed' });
  harness.recognition.onend();
  assert.match(harness.elements['voice-status'].textContent, /Microphone access was blocked/);
  assert.equal(harness.elements['customer-response'].disabled, false);
});

test('unsupported recognition disables only voice controls and typed fallback still posts', async () => {
  const harness = makeHarness({ recognitionSupported: false });
  await harness.elements['start-button'].trigger('click');
  assert.equal(harness.elements['microphone-button'].disabled, true);
  assert.equal(harness.elements['done-speaking-button'].disabled, true);
  assert.equal(harness.elements['customer-response'].disabled, false);
  harness.elements['customer-response'].value = 'Yes, speaking';
  await harness.elements['send-button'].trigger('click');
  assert.deepEqual(harness.calls.find(call => call.url.endsWith('/respond')).body, { text: 'Yes, speaking' });
});

test('recognized customer speech posts once and returned reply is spoken', async () => {
  const harness = makeHarness();
  await startAndListen(harness);
  harness.recognition.onresult({ resultIndex: 0, results: [finalResult('Yes, speaking')] });
  harness.recognition.onend();
  await waitForRequest();
  const responseCall = harness.calls.find(call => call.url.endsWith('/respond'));
  assert.deepEqual(responseCall.body, { text: 'Yes, speaking' });
  assert.equal(harness.utterances.at(-1).text, 'Is this a convenient time to talk?');
  assert.deepEqual(harness.elements.transcript.children.map(bubble => bubble.textContent), [
    'Hello, may I speak with Ravi?', 'Yes, speaking', 'Is this a convenient time to talk?',
  ]);
});

test('speech playback starts on the greeting and survives a second conversation', async () => {
  const harness = makeHarness();
  await harness.elements['start-button'].trigger('click');
  assert.equal(harness.utterances.at(-1).text, 'Hello, may I speak with Ravi?');
  assert.equal(harness.elements['voice-status'].dataset.state, 'speaking');
  assert.equal(harness.speechSynthesis.cancelCalls, 0);
  assert.ok(harness.speechSynthesis.resumeCalls > 0);

  harness.utterances.at(-1).onend();
  harness.elements['customer-response'].value = 'Yes, speaking';
  await harness.elements['send-button'].trigger('click');
  await waitForRequest();
  assert.equal(harness.utterances.at(-1).text, 'Is this a convenient time to talk?');

  await harness.elements['start-button'].trigger('click');
  assert.equal(harness.utterances.at(-1).text, 'Hello, may I speak with Ravi?');
  assert.equal(harness.elements['voice-status'].dataset.state, 'speaking');
  assert.equal(harness.elements['microphone-button'].disabled, true);
  assert.equal(harness.elements['voice-status'].textContent, 'Speaking response…');
});

