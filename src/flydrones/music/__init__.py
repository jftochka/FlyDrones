"""The fly brain drone compositor: spikes and flight -> Pd, Max, Tidal, Strudel.

    from flydrones.music import Compositor, make_sinks

    comp = Compositor(cfg, brain=brain)
    out = make_sinks("pd,strudel", cfg)
    ...
    out.frame(comp.tick(info))

See ``docs/MUSIC.md`` for the mapping, the OSC schema and the receiving patches.
"""

from .compositor import SECTIONS, Compositor, Voice, VoiceSpec
from .events import SCALES, ControlEvent, Frame, NoteEvent, Scale, event_from_dict, note_name
from .patterns import Score, strudel_live_snippet, tidal_live_file, to_strudel, to_tidal, transcribe
from .server import EventServer
from .sinks import (
    DEFAULT_PORTS,
    FanOut,
    FudiSink,
    JsonlSink,
    MaxSink,
    OscSink,
    PrintSink,
    PureDataSink,
    Sink,
    StrudelSink,
    SuperDirtSink,
    TidalSink,
    make_sink,
    make_sinks,
    read_jsonl,
)

__all__ = [
    "SCALES", "SECTIONS", "DEFAULT_PORTS",
    "Compositor", "ControlEvent", "EventServer", "FanOut", "Frame", "FudiSink", "JsonlSink", "MaxSink", "NoteEvent",
    "OscSink", "PrintSink", "PureDataSink", "Scale", "Score", "Sink", "StrudelSink", "SuperDirtSink", "TidalSink",
    "Voice", "VoiceSpec", "event_from_dict", "make_sink", "make_sinks", "note_name", "read_jsonl", "strudel_live_snippet",
    "tidal_live_file", "to_strudel", "to_tidal", "transcribe",
]
