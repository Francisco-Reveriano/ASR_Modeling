"""Remove near-silent TTS request padding without changing pauses inside speech."""

from array import array
import sys

SAMPLE_RATE = 24000
FRAME_BYTES = SAMPLE_RATE * 2 // 100  # 10 ms PCM16 frames.
LEADING_BYTES = SAMPLE_RATE * 2 * 80 // 1000
TRAILING_BYTES = SAMPLE_RATE * 2 * 120 // 1000
MAX_QUIET_BYTES = SAMPLE_RATE * 2 * 2
QUIET_PEAK = 64  # About -54 dBFS; keep even very soft speech above this level.


def _frames(chunks):
    remainder = b""
    for chunk in chunks:
        remainder += chunk
        end = len(remainder) // FRAME_BYTES * FRAME_BYTES
        for offset in range(0, end, FRAME_BYTES):
            yield remainder[offset:offset + FRAME_BYTES]
        remainder = remainder[end:]
    if len(remainder) % 2:
        raise ValueError("Incomplete PCM sample")
    if remainder:
        yield remainder


def trim_speech_padding(chunks):
    """Stream speech intact, keeping 80 ms before and 120 ms after each request.

    Only consecutive near-silent edge frames are shortened. Internal pauses are
    emitted unchanged when speech resumes. A two-second lookbehind bounds memory
    and latency even for an unusually long pause; excess silence streams through.
    """
    quiet = bytearray()
    started = False
    for frame in _frames(chunks):
        samples = array("h", frame)
        if sys.byteorder != "little":
            samples.byteswap()
        if max(abs(sample) for sample in samples) <= QUIET_PEAK:
            quiet.extend(frame)
            limit = MAX_QUIET_BYTES if started else LEADING_BYTES
            excess = len(quiet) - limit
            if excess > 0:
                if started:
                    yield bytes(quiet[:excess])
                del quiet[:excess]
        else:
            if quiet:
                yield bytes(quiet)
                quiet.clear()
            yield frame
            started = True
    if quiet:
        yield bytes(quiet[:TRAILING_BYTES])
