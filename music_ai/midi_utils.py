from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo


@dataclass
class MidiGridConfig:
    step_ms: int = 100  # fixed temporal grid
    ticks_per_step: int = 480  # computed after reading tempo/ticks


def tokens_to_midi(
    token_ids: List[int],
    inv_vocab: Dict[int, str],
    out_path: str,
    tempo_bpm: int = 120,
    step_ms: int = 100,
    program: int = 0,
):
    """Convert a token stream back to a single-track MIDI.

    Token format: "N:<midi_note>:<dur_steps>" or "REST:<dur_steps>" or "SOS"/"EOS".
    """

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))

    # mido uses delta-time ticks; choose a reasonable ticks_per_beat baseline.
    ticks_per_beat = mid.ticks_per_beat or 480
    step_ticks = int(ticks_per_beat * (step_ms / (60000 / tempo_bpm)))
    step_ticks = max(1, step_ticks)

    track.append(Message('program_change', program=program, time=0))

    current_time = 0
    pending_time = 0

    def dur_to_ticks(dur_steps: int) -> int:
        return max(0, dur_steps * step_ticks)

    for tid in token_ids:
        tok = inv_vocab.get(tid, None)
        if tok is None:
            continue
        if tok == 'EOS' or tok == 'SOS':
            continue
        if tok.startswith('REST:'):
            dur_steps = int(tok.split(':')[1])
            pending_time += dur_to_ticks(dur_steps)
            continue
        if tok.startswith('N:'):
            _, note_str, dur_str = tok.split(':')
            note = int(note_str)
            dur_steps = int(dur_str)

            # Note-on after any pending rest
            track.append(Message('note_on', note=note, velocity=64, time=pending_time))
            pending_time = 0
            track.append(Message('note_off', note=note, velocity=64, time=dur_to_ticks(dur_steps)))
            continue

    mid.save(out_path)


def midi_to_note_tokens(
    midi_path: str,
    step_ms: int = 100,
    max_notes: int | None = None,
):
    """Convert a MIDI file into a token list using a simple event quantization.

    Uses mido for robustness. We interpret note_on/note_off events and quantize
    their start times onto a fixed grid.

    Output tokens:
      - "SOS"
      - "N:<midi_note>:<dur_steps>"
      - "REST:<dur_steps>" (implicit waits between events)
      - "EOS"

    This baseline assumes monophonic-ish lines; for polyphony it will emit
    overlapping notes sequentially in event order.
    """

    from mido import MidiFile

    mid = MidiFile(midi_path)

    # Gather events with absolute time in ticks
    events = []  # (abs_ticks, type, note, velocity)
    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((abs_ticks, 'on', msg.note, msg.velocity))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append((abs_ticks, 'off', msg.note, msg.velocity))

    if not events:
        return []

    events.sort(key=lambda x: x[0])

    # Build note durations by pairing on/off per note instance (simple stack)
    # Use FIFO per note number.
    starts: Dict[int, List[Tuple[int, int]]] = {}
    intervals: List[Tuple[int, int, int]] = []  # (start_ticks, note, dur_ticks)

    for abs_ticks, typ, note, vel in events:
        if typ == 'on':
            starts.setdefault(note, []).append((abs_ticks, vel))
        else:  # off
            if note in starts and starts[note]:
                start_ticks, start_vel = starts[note].pop(0)
                dur_ticks = max(1, abs_ticks - start_ticks)
                intervals.append((start_ticks, note, dur_ticks))

    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    tokens: List[str] = ['SOS']

    # Convert step_ms to ticks using tempo. If multiple tempi exist, this is approximate.
    # Use default mido tempo from header if present; mido already converts message.time using ticks_per_beat.
    ticks_per_beat = mid.ticks_per_beat or 480
    tempo_bpm = 120
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                # msg.tempo is microseconds per beat
                tempo_bpm = int(60000000 / msg.tempo)
                break
        break

    step_ticks = int(ticks_per_beat * (step_ms / (60000 / tempo_bpm)))
    step_ticks = max(1, step_ticks)

    prev_end_ticks = None
    for i, (start_ticks, note, dur_ticks) in enumerate(intervals):
        if max_notes is not None and i >= max_notes:
            break

        # Rest from previous note end
        if prev_end_ticks is not None:
            gap_ticks = max(0, start_ticks - prev_end_ticks)
            gap_steps = int(round(gap_ticks / step_ticks))
            if gap_steps > 0:
                tokens.append(f"REST:{gap_steps}")

        dur_steps = int(max(1, round(dur_ticks / step_ticks)))
        tokens.append(f"N:{note}:{dur_steps}")

        prev_end_ticks = start_ticks + dur_ticks

    tokens.append('EOS')
    return tokens

