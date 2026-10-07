/* Streamlit v1 component protocol. PCM never contains executable markup. */
const playButton = document.getElementById("play");
const stopButton = document.getElementById("stop");
const statusLabel = document.getElementById("status");
const modelLabel = document.getElementById("model");
let state = null, sessionId = null, context = null;
let scheduled = 0, played = 0, nextTime = 0, enabled = false, stopped = false;
let paused = false, blocked = false, resuming = false, disposed = false;
let playbackAt = null;
let buffering = true, bufferingSince = null;
const nodes = new Set();

function send(type, extra = {}) {
    window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, "*");
}
function acknowledge() {
    send("streamlit:setComponentValue", {
        value: {session_id: sessionId, played, stopped}, dataType: "json"
    });
}
function clearAudio() {
    for (const node of nodes) { node.onended = null; node.stop(); }
    nodes.clear();
    nextTime = 0;
}
function updateStatus() {
    modelLabel.textContent = "AI-generated English voice" + (state?.model ? ` · ${state.model}` : "");
    playButton.disabled = stopped;
    stopButton.disabled = stopped;
    playButton.hidden = !enabled && !paused && !blocked;
    playButton.textContent = enabled ? "Pause voice" : paused ? "Resume voice" : "Enable sound";
    statusLabel.textContent = state?.error || (stopped ? "Voice stopped. Translation continues." :
        paused ? "Voice paused." :
        state?.session_id === "standby" ? "English audio will start with your next recording or file." :
        playbackAt === null ? (state?.complete ? "English playback complete." :
            "Waiting for the first English translation to start the 1-minute audio lead…") :
        performance.now() < playbackAt ? `Building an audio lead · starts in ${Math.ceil((playbackAt - performance.now()) / 1000)}s` :
        !enabled ? "Your browser has paused sound. Click Enable sound if it does not start automatically." :
        nodes.size ? "Speaking English…" : state?.complete ? "English playback complete." :
        bufferingSince !== null ? "Buffering English audio for smoother playback…" :
        state?.pending ? "Generating English speech…" : "Waiting for English translation…");
}
function pump() {
    if (!enabled || stopped || !context || context.state !== "running" || !state ||
        playbackAt === null || performance.now() < playbackAt) return;
    const ready = state.chunks.filter(chunk => chunk.id > scheduled);
    if (nextTime <= context.currentTime) buffering = true;
    if (!ready.length) { bufferingSince = null; return; }
    if (buffering) {
        bufferingSince ??= performance.now();
        const seconds = ready.reduce((total, chunk) => total + atob(chunk.pcm).length / (state.sample_rate * 2), 0);
        // Wait for a small runway after a stall, instead of playing isolated
        // 200 ms scraps. Final clips flush immediately; live waits cap at 2 s.
        if (seconds + 1e-6 < (state.minimum_buffer_seconds || 0) && !state.generation_complete &&
            performance.now() - bufferingSince < 2000) { updateStatus(); return; }
        buffering = false;
        bufferingSince = null;
    }
    for (const chunk of ready) {
        // Retain excess audio on the server until the listener catches up.
        if (nextTime - context.currentTime > 4) break;
        const bytes = Uint8Array.from(atob(chunk.pcm), c => c.charCodeAt(0));
        const view = new DataView(bytes.buffer);
        const buffer = context.createBuffer(1, bytes.length / 2, state.sample_rate);
        const samples = buffer.getChannelData(0);
        for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
        const node = context.createBufferSource();
        node.buffer = buffer;
        node.connect(context.destination);
        const owner = sessionId;
        node.onended = () => {
            nodes.delete(node);
            node.disconnect();
            if (owner !== sessionId || stopped) return;
            played = Math.max(played, chunk.id);
            acknowledge();
            pump();
            updateStatus();
        };
        nodes.add(node);
        // A small lead absorbs render jitter; adjacent ready chunks are
        // scheduled directly against the same audio clock without extra gaps.
        if (nextTime <= context.currentTime) nextTime = context.currentTime + 0.08;
        node.start(nextTime);
        nextTime += buffer.duration;
        scheduled = chunk.id;
    }
    updateStatus();
}
async function activate(fromGesture = false) {
    if (disposed || stopped || paused || !state?.armed || (resuming && !fromGesture)) return;
    try {
        if (!context) {
            context = new AudioContext({sampleRate: 24000, latencyHint: "interactive"});
            context.onstatechange = () => {
                enabled = context.state === "running" && !paused && !stopped && !!state?.armed;
                if (enabled) blocked = false;
                pump();
                updateStatus();
            };
        }
        resuming = true;
        const resumed = context.resume();
        blocked = context.state !== "running";
        updateStatus();
        await resumed;
        if (disposed || stopped || paused || !state?.armed) return;
        enabled = context.state === "running";
        blocked = !enabled;
        pump();
        updateStatus();
    } catch (_) { blocked = true; enabled = false; updateStatus(); }
    finally { resuming = false; }
}
playButton.onclick = async () => {
    if (enabled) {
        paused = true;
        enabled = false;
        await context.suspend();
        updateStatus();
    } else {
        paused = false;
        buffering = true;
        bufferingSince = null;
        await activate(true);
    }
};
stopButton.onclick = () => {
    stopped = true;
    enabled = false;
    clearAudio();
    acknowledge();
    updateStatus();
};
window.addEventListener("message", event => {
    if (event.source !== window.parent || event.data.type !== "streamlit:render") return;
    state = event.data.args.state;
    if (sessionId !== state.session_id) {
        clearAudio();
        sessionId = state.session_id;
        scheduled = played = state.acked;
        stopped = false;
        paused = false;
        buffering = true;
        bufferingSince = null;
        playbackAt = null;
    }
    // The session mounts before translation is ready. Start this clock only
    // once the server supplies a deadline; subsequent polls never restart it.
    if (playbackAt === null && state.playback_delay_ms !== null) {
        playbackAt = performance.now() + Math.max(0, state.playback_delay_ms || 0);
    }
    if (state.closed) { stopped = true; enabled = false; clearAudio(); }
    if (!state.armed) { enabled = false; clearAudio(); }
    else if (!enabled) void activate();
    pump();
    updateStatus();
    send("streamlit:setFrameHeight", {height: state.armed ? 118 : 0});
});
// Start queued audio when the lead-in expires, even between server renders.
const playbackTimer = setInterval(() => { pump(); updateStatus(); }, 100);
// Listen only for a trusted activation gesture, without reading or modifying
// parent controls. Cross-origin hosts can still use the in-player fallback.
const gestureTargets = [document];
try { if (window.parent.document !== document) gestureTargets.push(window.parent.document); } catch (_) {}
const unlock = event => {
    if (event.target === playButton || event.target === stopButton) return;
    if (event.isTrusted && !enabled) void activate(true);
};
for (const target of gestureTargets) {
    target.addEventListener("pointerdown", unlock, true);
    target.addEventListener("keydown", unlock, true);
}
window.addEventListener("pagehide", () => {
    disposed = true;
    clearInterval(playbackTimer);
    for (const target of gestureTargets) {
        target.removeEventListener("pointerdown", unlock, true);
        target.removeEventListener("keydown", unlock, true);
    }
    clearAudio();
    if (context) context.close();
});
send("streamlit:componentReady", {apiVersion: 1});
send("streamlit:setFrameHeight", {height: 118});
