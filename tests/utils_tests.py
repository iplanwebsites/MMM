"""Utils for tests."""

from pathlib import Path

HERE = Path(__file__).parent
MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))
MIDI_PATH = HERE / "midis" / "I Gotta Feeling.mid"
