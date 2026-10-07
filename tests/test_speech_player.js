/* Exercise the real player against browser policy and a fake Web Audio clock. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {test} = require("node:test");

function player({blocked = false} = {}) {
    const events = {}, parentEvents = {}, messages = [], sources = [], contexts = [];
    const elements = Object.fromEntries(["play", "stop", "status", "model"].map(id => [id, {}]));
    let allowed = !blocked, now = 0, timer = null;
    const resumes = [];
    class AudioContext {
        constructor() { this.currentTime = 0; this.state = "suspended"; contexts.push(this); }
        async resume() {
            if (!allowed) return new Promise(resolve => resumes.push(resolve));
            this.state = "running";
            this.onstatechange?.();
            resumes.splice(0).forEach(resolve => resolve());
        }
        async suspend() { this.state = "suspended"; this.onstatechange?.(); }
        async close() { this.state = "closed"; this.onstatechange?.(); }
        createBuffer(channels, length, rate) {
            assert.equal(channels, 1);
            const samples = new Float32Array(length);
            return {duration: length / rate, getChannelData: () => samples, samples};
        }
        createBufferSource() {
            const source = {
                connect() {}, disconnect() {},
                start(time) { this.time = time; },
                stop() { this.stopped = true; },
                finish() { this.onended?.(); }
            };
            sources.push(source);
            return source;
        }
    }
    const parent = {
        postMessage: message => messages.push(message),
        document: {
            addEventListener: (name, callback) => { parentEvents[name] = callback; },
            removeEventListener: name => { delete parentEvents[name]; }
        }
    };
    const sandbox = {
        document: {getElementById: id => elements[id], addEventListener() {}, removeEventListener() {}},
        AudioContext, performance: {now: () => now},
        setInterval: callback => { timer = callback; return 1; },
        clearInterval: () => { timer = null; },
        window: {parent, addEventListener: (name, callback) => { events[name] = callback; }},
        atob: text => Buffer.from(text, "base64").toString("binary")
    };
    vm.runInNewContext(fs.readFileSync("src/speech_player/player.js", "utf8"), sandbox);
    const chunk = (id, seconds = null) => ({id, pcm: (seconds ? Buffer.alloc(seconds * 48000) :
        Buffer.from([0, 128, 0, 0, 255, 127])).toString("base64")});
    const render = async (state = {}, source = parent) => {
        events.message({source, data: {type: "streamlit:render", args: {state: {
            session_id: "one", armed: true, acked: 0, chunks: [], sample_rate: 24000,
            closed: false, complete: false, playback_delay_ms: 0, ...state
        }}}});
        await Promise.resolve();
    };
    const gesture = async (trusted = true) => {
        if (trusted) allowed = true;
        parentEvents.pointerdown({isTrusted: trusted});
        await Promise.resolve();
    };
    const advance = ms => {
        now += ms;
        contexts.forEach(context => { if (context.state === "running") context.currentTime += ms / 1000; });
        timer?.();
    };
    return {events, messages, sources, contexts, elements, render, chunk, gesture, parentEvents, advance};
}

test("automatically plays PCM once across rerenders without a Play click", async () => {
    const p = player();
    await p.render({chunks: [p.chunk(1)]});
    assert.equal(p.sources.length, 1);
    assert.equal(p.elements.play.textContent, "Pause voice");
    await p.render({chunks: [p.chunk(1), p.chunk(2)]});
    await p.render({chunks: [p.chunk(1), p.chunk(2)]});
    assert.equal(p.sources.length, 2);
    assert.equal(p.sources[1].time, p.sources[0].time + p.sources[0].buffer.duration);
    assert.equal(p.sources[0].buffer.samples[0], -1);
    assert.equal(p.sources[0].buffer.samples[2], 32767 / 32768);
    p.sources[0].finish();
    const ack = p.messages.filter(m => m.type === "streamlit:setComponentValue").at(-1);
    assert.equal(ack.value.played, 1);
    assert.equal(ack.value.session_id, "one");
});

test("Start/Transcribe gesture unlocks the pre-mounted player under autoplay restrictions", async () => {
    const p = player({blocked: true});
    await p.render({session_id: "standby"});
    assert.equal(p.contexts[0].state, "suspended");
    await p.gesture(false);
    assert.equal(p.contexts[0].state, "suspended");
    await p.gesture();
    await p.render({chunks: [p.chunk(1)]});
    assert.equal(p.contexts.length, 1);
    assert.equal(p.sources.length, 1);
    assert.equal(p.elements.play.textContent, "Pause voice");
});

test("blocked playback exposes an Enable sound fallback", async () => {
    const p = player({blocked: true});
    await p.render({chunks: [p.chunk(1)]});
    assert.equal(p.sources.length, 0);
    assert.equal(p.elements.play.hidden, false);
    assert.equal(p.elements.play.textContent, "Enable sound");
    assert.match(p.elements.status.textContent, /browser has paused sound/);
});

test("buffers silently for 60 seconds, then starts automatically without a server render", async () => {
    const p = player();
    await p.render({playback_delay_ms: 60000, chunks: [p.chunk(1)]});
    assert.equal(p.sources.length, 0);
    p.advance(59000);
    assert.equal(p.sources.length, 0);
    assert.match(p.elements.status.textContent, /starts in 1s/);
    // Re-renders must not restart the countdown.
    await p.render({playback_delay_ms: 1000, chunks: [p.chunk(1)]});
    p.advance(1000);
    assert.equal(p.sources.length, 1);
    assert.match(p.elements.status.textContent, /Speaking English/);
});

test("the minute begins with accepted translation, not session creation or first PCM", async () => {
    const p = player();
    await p.render({playback_delay_ms: null});
    p.advance(120000);
    await p.render({playback_delay_ms: null});
    assert.match(p.elements.status.textContent, /Waiting for the first English translation/);
    assert.equal(p.sources.length, 0);
    // Translation is now accepted, but TTS has not produced audio yet.
    await p.render({playback_delay_ms: 60000});
    assert.match(p.elements.status.textContent, /starts in 60s/);
    p.advance(59000);
    await p.render({playback_delay_ms: 1000, chunks: [p.chunk(1)]});
    assert.equal(p.sources.length, 0);
    assert.match(p.elements.status.textContent, /starts in 1s/);
    p.advance(1000);
    assert.equal(p.sources.length, 1);
});

test("an empty completed conversation does not wait forever for a countdown", async () => {
    const p = player();
    await p.render({playback_delay_ms: null, complete: true, generation_complete: true});
    assert.equal(p.elements.status.textContent, "English playback complete.");
    assert.equal(p.sources.length, 0);
});

test("short completed clips still respect the lead-in and reconnect uses remaining delay", async () => {
    const p = player();
    await p.render({playback_delay_ms: 5000, chunks: [p.chunk(1)], complete: true});
    p.advance(4999);
    assert.equal(p.sources.length, 0);
    p.advance(1);
    assert.equal(p.sources.length, 1);
});

test("pause survives polls and unrelated clicks, then resume keeps ordering", async () => {
    const p = player();
    await p.render({chunks: [p.chunk(1)]});
    await p.elements.play.onclick();
    assert.equal(p.contexts[0].state, "suspended");
    await p.render({chunks: [p.chunk(1), p.chunk(2)]});
    await p.gesture();
    assert.equal(p.sources.length, 1);
    assert.equal(p.contexts[0].state, "suspended");
    await p.elements.play.onclick();
    assert.equal(p.sources.length, 2);
});

test("pause and stop cannot be overridden by the lead-in timer", async () => {
    const p = player();
    await p.render({playback_delay_ms: 60000, chunks: [p.chunk(1)]});
    await p.elements.play.onclick();
    p.advance(61000);
    assert.equal(p.sources.length, 0);
    await p.elements.play.onclick();
    assert.equal(p.sources.length, 1);
    p.elements.stop.onclick();
    assert.equal(p.sources[0].stopped, true);
    assert.equal(p.messages.at(-1).value.stopped, true);
    await p.render({chunks: [p.chunk(2)]});
    await p.gesture();
    assert.equal(p.sources.length, 1);
});

test("new session cancels old audio, resets its delay and ignores old callbacks", async () => {
    const p = player();
    await p.render({chunks: [p.chunk(1)]});
    const oldCallback = p.sources[0].onended;
    await p.render({session_id: "two", playback_delay_ms: 60000, chunks: [p.chunk(1)]});
    assert.equal(p.sources[0].stopped, true);
    assert.equal(p.sources.length, 1);
    const before = p.messages.filter(m => m.type === "streamlit:setComponentValue").length;
    oldCallback();
    assert.equal(p.messages.filter(m => m.type === "streamlit:setComponentValue").length, before);
    p.advance(60000);
    assert.equal(p.sources.length, 2);
});

test("restored player skips acknowledged audio and handles safe failure", async () => {
    const p = player();
    await p.render({acked: 3, chunks: [p.chunk(3), p.chunk(4)]});
    assert.equal(p.sources.length, 1);
    await p.render({closed: true, error: "Speech unavailable."});
    assert.equal(p.sources[0].stopped, true);
    assert.equal(p.elements.status.textContent, "Speech unavailable.");
});

test("schedules contiguous audio with bounded browser lookahead", async () => {
    const p = player();
    await p.render({chunks: Array.from({length: 30}, (_, i) => p.chunk(i + 1, 0.2))});
    assert(p.sources.length >= 19 && p.sources.length <= 21);
    for (let i = 1; i < p.sources.length; i++) {
        assert.equal(p.sources[i].time, p.sources[i-1].time + 0.2);
    }
    p.contexts[0].currentTime = 1;
    p.sources[0].finish();
    assert(p.sources.length > 21 && p.sources.length < 30);
});

test("disabled standby does not start audio; navigation cleans up gesture listeners", async () => {
    const p = player();
    await p.render({armed: false});
    await p.gesture();
    assert.equal(p.contexts.length, 0);
    await p.render({chunks: [p.chunk(1)]}, {});
    assert.equal(p.contexts.length, 0);
    await p.render({chunks: [p.chunk(1)]});
    assert.equal(p.sources.length, 1);
    p.events.pagehide();
    assert.equal(p.sources[0].stopped, true);
    assert.equal(p.contexts[0].state, "closed");
    assert.deepEqual(p.parentEvents, {});
});

test("a chunk arriving just before the previous one ends does not add a scheduling pause", async () => {
    const p = player();
    await p.render({chunks: [p.chunk(1, 0.2)]});
    const end = p.sources[0].time + p.sources[0].buffer.duration;
    p.contexts[0].currentTime = end - 0.03;
    await p.render({chunks: [p.chunk(1, 0.2), p.chunk(2, 0.2)]});
    assert.equal(p.sources[1].time, end);
});

test("bursty PCM builds a short buffer before playing and after an underrun", async () => {
    const p = player();
    const settings = {minimum_buffer_seconds: 2, generation_complete: false};
    await p.render({...settings, playback_delay_ms: 60000, chunks: [p.chunk(1, 0.2)]});
    p.advance(60000);
    assert.equal(p.sources.length, 0);
    await p.render({...settings, chunks: Array.from({length: 10}, (_, i) => p.chunk(i + 1, 0.2))});
    assert.equal(p.sources.length, 10);
    p.advance(2200);
    p.sources.forEach(s => s.finish());
    await p.render({...settings, chunks: [p.chunk(11, 0.2)]});
    assert.equal(p.sources.length, 10);
    p.advance(500);
    await p.render({...settings, chunks: Array.from({length: 10}, (_, i) => p.chunk(i + 11, 0.2))});
    assert.equal(p.sources.length, 20);
    for (let i = 11; i < 20; i++) {
        assert.equal(p.sources[i].time, p.sources[i - 1].time + 0.2);
    }
});

test("buffer wait is bounded and completed short audio flushes immediately", async () => {
    const p = player();
    const settings = {minimum_buffer_seconds: 2};
    await p.render({...settings, chunks: [p.chunk(1, 0.2)]});
    assert.equal(p.sources.length, 0);
    p.advance(1999);
    assert.equal(p.sources.length, 0);
    p.advance(1);
    assert.equal(p.sources.length, 1);
    await p.render({...settings, session_id: "two", generation_complete: true, chunks: [p.chunk(1, 0.2)]});
    assert.equal(p.sources.length, 2);
});

test("voice disclosure uses the active model and clears between sessions", async () => {
    const p = player();
    await p.render({model: "tts-1-hd"});
    assert.equal(p.elements.model.textContent, "AI-generated English voice · tts-1-hd");
    await p.render({session_id: "standby", armed: false});
    assert.equal(p.elements.model.textContent, "AI-generated English voice");
});
