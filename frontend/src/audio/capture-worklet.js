/* AudioWorklet global scope: local asset, no network requests or microphone playback. */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batchSamples = Math.max(1, Math.round(sampleRate / 10));
    this.pending = new Int16Array(this.batchSamples);
    this.count = 0;
    this.finished = false;
    this.port.onmessage = event => {
      if (event.data?.type === 'flush') {
        this.finished = true;
        this.emit();
        // MessagePort preserves ordering: every PCM batch precedes this acknowledgement.
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  emit() {
    if (!this.count) return;
    const pcm = new ArrayBuffer(this.count * 2);
    const view = new DataView(pcm);
    for (let index = 0; index < this.count; index += 1) view.setInt16(index * 2, this.pending[index], true);
    this.port.postMessage({ type: 'pcm', pcm }, [pcm]);
    this.count = 0;
  }

  process(inputs, outputs) {
    for (const output of outputs) for (const channel of output) channel.fill(0);
    if (this.finished) return true;
    const channels = inputs[0];
    if (!channels?.length) return true;
    const frames = channels[0].length;
    // Native render block length may change; it is not assumed to be 128 samples.
    for (let frame = 0; frame < frames; frame += 1) {
      let mono = 0;
      for (const channel of channels) mono += channel[frame] ?? 0;
      mono = Math.max(-1, Math.min(1, mono / channels.length));
      this.pending[this.count++] = Math.round(mono < 0 ? mono * 32768 : mono * 32767);
      if (this.count === this.batchSamples) this.emit();
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
