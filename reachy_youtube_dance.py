"""
Reachy YouTube Dance Script
===========================
Makes Reachy Mini dance to YouTube videos with beat-synced movements.

Usage:
    # Basic beat-sync dance
    python reachy_youtube_dance.py [URL] [--volume normal|loud|max] [--time normal|half|double]

    # With config file (time switches)
    python reachy_youtube_dance.py --config configs/jailhouse_rock.json

    # With config file (choreographed triggers)
    python reachy_youtube_dance.py --config configs/jungle_book.json

    Interactive commands during session:
        - Enter URL to play new track
        - 'v' or 'volume' to change volume
        - 'q' or 'quit' to exit

Requirements:
    - Reachy Mini daemon must be running
    - yt-dlp, librosa, numpy, pydub installed in environment
"""

import sys
import os
import re
import json
import argparse
import tempfile
import threading
import time
import wave
import subprocess
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

VOLUME_LEVELS = {
    'normal': 0,
    'loud': 6,
    'max': 12,
    'quiet': -6,
}

# TIME_MULTIPLIERS: Adjusts the groove/movement timing relative to detected BPM.
# This is NOT the same as librosa's tempo detection - librosa detects the song's
# actual BPM, but many genres (blues, ballads, slow rock) have a "felt" pulse
# that differs from the notated tempo. Half-time means movements follow a pulse
# at half the detected BPM, matching how humans naturally sway to slower grooves.
TIME_MULTIPLIERS = {
    'normal': 1.0,    # Use detected BPM as-is
    'half': 0.5,      # Half-time feel (blues, ballads, slow rock)
    'double': 2.0,    # Double-time feel (punk, fast songs feeling sluggish)
    'quarter': 0.25,  # Quarter-time (very slow ambient grooves)
}

CURRENT_TIME = 'normal'

CURRENT_VOLUME = 'loud'

# Loaded config (from --config JSON file)
CURRENT_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a song config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_time_multiplier_at(t: float, config: Optional[Dict] = None) -> float:
    """
    Get the time multiplier at a given timestamp.
    Checks config time_switches first, falls back to CURRENT_TIME.
    """
    base_mult = TIME_MULTIPLIERS.get(CURRENT_TIME, 1.0)

    if config and 'time_switches' in config:
        # Find the most recent time switch before t
        for switch in reversed(config['time_switches']):
            if t >= switch['at']:
                return TIME_MULTIPLIERS.get(switch['time'], base_mult)

    return base_mult


print("=" * 60)
print("REACHY YOUTUBE DANCE")
print("=" * 60)
print()

print("[INIT] Loading dependencies...")

try:
    import numpy as np
    print("[INIT] numpy loaded successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import numpy: {e}")
    print("[ERROR] Please install numpy: pip install numpy")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    import librosa
    print("[INIT] librosa loaded successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import librosa: {e}")
    print("[ERROR] Please install librosa: pip install librosa")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    print("[INIT] reachy_mini SDK loaded successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import reachy_mini: {e}")
    print("[ERROR] Please install reachy-mini SDK")
    input("Press Enter to exit...")
    sys.exit(1)

print("[INIT] All dependencies loaded")
print()


# ============================================================================
# CHOREOGRAPHY SYSTEM
# ============================================================================
# Lyric-triggered choreography with multi-phase sequences.
# Movement triggers are loaded from config JSON files.
# See configs/jungle_book.json for an example.


@dataclass
class ChoreoLimits:
    """Hardware limits for Reachy Mini."""
    HEAD_PITCH_MIN: float = -30.0
    HEAD_PITCH_MAX: float = 20.0
    HEAD_YAW_MIN: float = -45.0
    HEAD_YAW_MAX: float = 45.0
    HEAD_ROLL_MIN: float = -20.0
    HEAD_ROLL_MAX: float = 20.0
    ANTENNA_MIN: float = 0.0
    ANTENNA_MAX: float = 1.0
    BODY_YAW_MIN: float = -15.0
    BODY_YAW_MAX: float = 15.0

    def clamp(self, pitch, yaw, roll, ant_l, ant_r, body_yaw):
        return (
            max(self.HEAD_PITCH_MIN, min(self.HEAD_PITCH_MAX, pitch)),
            max(self.HEAD_YAW_MIN, min(self.HEAD_YAW_MAX, yaw)),
            max(self.HEAD_ROLL_MIN, min(self.HEAD_ROLL_MAX, roll)),
            max(self.ANTENNA_MIN, min(self.ANTENNA_MAX, ant_l)),
            max(self.ANTENNA_MIN, min(self.ANTENNA_MAX, ant_r)),
            max(self.BODY_YAW_MIN, min(self.BODY_YAW_MAX, body_yaw)),
        )


CHOREO_LIMITS = ChoreoLimits()


@dataclass
class Pose:
    """A single pose with all joint values."""
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    body_yaw: float = 0.0

    def lerp(self, other: 'Pose', t: float) -> 'Pose':
        t = max(0, min(1, t))
        return Pose(
            self.head_pitch + (other.head_pitch - self.head_pitch) * t,
            self.head_yaw + (other.head_yaw - self.head_yaw) * t,
            self.head_roll + (other.head_roll - self.head_roll) * t,
            self.antenna_left + (other.antenna_left - self.antenna_left) * t,
            self.antenna_right + (other.antenna_right - self.antenna_right) * t,
            self.body_yaw + (other.body_yaw - self.body_yaw) * t,
        )

    def clamped(self) -> 'Pose':
        p, y, r, al, ar, by = CHOREO_LIMITS.clamp(
            self.head_pitch, self.head_yaw, self.head_roll,
            self.antenna_left, self.antenna_right, self.body_yaw
        )
        return Pose(p, y, r, al, ar, by)


REST_POSE = Pose(0, 0, 0, 0, 0, 0)


@dataclass
class Phase:
    """A phase within a movement sequence."""
    pose: Pose
    duration: float
    ease: str = "smooth"


@dataclass
class Sequence:
    """A multi-phase movement with anticipation and follow-through."""
    phases: List[Phase]
    blend_weight: float = 0.9

    def get_total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def get_pose_at(self, elapsed: float) -> Tuple[Pose, float]:
        if not self.phases:
            return REST_POSE, 0
        t = 0
        for i, phase in enumerate(self.phases):
            if elapsed < t + phase.duration:
                progress = (elapsed - t) / phase.duration if phase.duration > 0 else 1
                if phase.ease == "smooth":
                    progress = progress * progress * (3 - 2 * progress)
                elif phase.ease == "snap":
                    progress = 1 if progress > 0.3 else progress / 0.3
                elif phase.ease == "ease_in":
                    progress = progress * progress
                elif phase.ease == "ease_out":
                    progress = 1 - (1 - progress) ** 2
                prev = self.phases[i-1].pose if i > 0 else REST_POSE
                return prev.lerp(phase.pose, progress), self.blend_weight
            t += phase.duration
        final = self.phases[-1].pose
        overshoot = elapsed - t
        decay = max(0, 1 - overshoot / 0.3)
        return final.lerp(REST_POSE, 1 - decay), self.blend_weight * decay


# Movement parsing from config - no hardcoded movements
def parse_movement_from_config(move_def: dict) -> Sequence:
    """Parse a movement definition from config JSON into a Sequence."""
    phases = []
    for phase_def in move_def.get('phases', []):
        pose_vals = phase_def['pose']  # [pitch, yaw, roll, ant_l, ant_r, body_yaw]
        pose = Pose(
            pose_vals[0], pose_vals[1], pose_vals[2],
            pose_vals[3], pose_vals[4], pose_vals[5]
        )
        phases.append(Phase(pose, phase_def['duration'], phase_def.get('ease', 'smooth')))
    return Sequence(phases, move_def.get('blend_weight', 0.85))


class ChoreographyEngine:
    """Manages lyric-triggered choreography with multi-phase sequences.

    All movements are defined in config files, not hardcoded.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.active_sequence: Optional[Sequence] = None
        self.sequence_start_time: float = 0
        self.alt_counters: Dict[str, int] = {}  # For alternating moves
        self.triggers_fired: int = 0
        self.movements: Dict[str, dict] = {}  # Movement definitions from config
        self.triggers: List[dict] = []
        self.current_idx = 0

        if config:
            self._load_movements_from_config(config)
            self._build_triggers_from_config(config)

    def _load_movements_from_config(self, config: Dict):
        """Load movement definitions from config JSON."""
        for name, move_def in config.get('movements', {}).items():
            self.movements[name] = move_def

    def _build_triggers_from_config(self, config: Dict):
        """Build triggers list from config JSON."""
        triggers = []
        for trigger in config.get('triggers', []):
            time_val = trigger['at']
            move_name = trigger['move']

            # Check if movement exists directly
            if move_name in self.movements:
                def make_fn(name):
                    return lambda: parse_movement_from_config(self.movements[name])
                triggers.append({"time": time_val, "fn": make_fn(move_name)})
            # Check for alternating pattern (name_left, name_right, etc.)
            elif f"{move_name}_left" in self.movements:
                def make_alt_fn(base_name):
                    def alt_fn():
                        if base_name not in self.alt_counters:
                            self.alt_counters[base_name] = 0
                        self.alt_counters[base_name] += 1
                        variants = [k for k in self.movements.keys() if k.startswith(f"{base_name}_")]
                        variant = variants[self.alt_counters[base_name] % len(variants)]
                        return parse_movement_from_config(self.movements[variant])
                    return alt_fn
                triggers.append({"time": time_val, "fn": make_alt_fn(move_name)})
            else:
                print(f"[WARN] Unknown move: {move_name}")
                continue

        triggers.sort(key=lambda x: x['time'])
        self.triggers = triggers

    def reset(self):
        self.current_idx = 0
        self.active_sequence = None
        self.sequence_start_time = 0
        self.alt_counters = {}
        self.triggers_fired = 0

    def update(self, t: float) -> Tuple[Optional[Pose], float]:
        while (self.current_idx < len(self.triggers) and
               self.triggers[self.current_idx]['time'] <= t):
            trigger = self.triggers[self.current_idx]
            self.active_sequence = trigger['fn']()
            self.sequence_start_time = trigger['time']
            self.triggers_fired += 1
            self.current_idx += 1

        if self.active_sequence:
            elapsed = t - self.sequence_start_time
            pose, weight = self.active_sequence.get_pose_at(elapsed)
            if elapsed > self.active_sequence.get_total_duration() + 0.3:
                self.active_sequence = None
                return None, 0
            return pose.clamped(), weight
        return None, 0


def blend_choreo(beat_pose: Pose, choreo_pose: Optional[Pose], weight: float) -> Pose:
    """Blend beat-sync with choreography."""
    if choreo_pose is None or weight <= 0:
        return beat_pose
    return beat_pose.lerp(choreo_pose, weight).clamped()


# ============================================================================
# YOUTUBE/AUDIO FUNCTIONS
# ============================================================================

def validate_youtube_url(url):
    """Validate and extract video ID from various YouTube URL formats."""
    if not url:
        return False, None, "URL is empty"
    if not isinstance(url, str):
        return False, None, f"URL must be a string, got {type(url).__name__}"
    url = url.strip()
    if len(url) > 2048:
        return False, None, "URL is too long (max 2048 characters)"

    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            print(f"[URL] Valid YouTube URL detected")
            print(f"[URL] Video ID: {video_id}")
            return True, video_id, None

    if "youtube" in url.lower() or "youtu.be" in url.lower():
        return False, None, "URL contains 'youtube' but video ID could not be extracted."
    return False, None, "URL does not appear to be a valid YouTube link"


def check_yt_dlp_installed():
    """Check if yt-dlp is available."""
    print("[CHECK] Verifying yt-dlp installation...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"[CHECK] yt-dlp version: {result.stdout.strip()}")
            return True
        return False
    except:
        return False


def get_video_title(url):
    """Fetch video title for display."""
    print("[INFO] Fetching video title...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--get-title", "--no-warnings", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            title = result.stdout.strip()
            print(f"[INFO] Video title: {title}")
            return title
    except:
        pass
    return None


def download_audio(url, output_path, timeout=300):
    """Download audio from YouTube URL."""
    print("[DOWNLOAD] Starting audio download...")
    print(f"[DOWNLOAD] URL: {url}")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "-o", output_path,
        "--no-playlist", "--no-warnings", "--progress", url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("[DOWNLOAD] Download completed successfully")
            return True, None
        else:
            return False, result.stderr.strip() if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, f"Download timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)


def analyze_audio(filepath):
    """Analyze audio file for beat detection and spectral features."""
    print("[ANALYSIS] Starting audio analysis...")
    print(f"[ANALYSIS] File: {filepath}")

    y, sr = librosa.load(filepath, sr=None)
    print(f"[ANALYSIS] Duration: {len(y)/sr:.2f} seconds")

    hop_length = 256
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    onset_max = onset_env.max()
    if onset_max > 0:
        onset_env = onset_env / onset_max

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    bass = S[freqs < 150, :].mean(axis=0)
    mids = S[(freqs >= 500) & (freqs < 2000), :].mean(axis=0)
    highs = S[freqs >= 6000, :].mean(axis=0)

    def normalize(arr):
        m = arr.max()
        return arr / m if m > 0 else arr

    bass, mids, highs = normalize(bass), normalize(mids), normalize(highs)
    spec_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])

    print(f"[ANALYSIS] Tempo: {tempo:.1f} BPM")
    print(f"[ANALYSIS] Beats detected: {len(beat_times)}")

    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'bass': bass, 'mids': mids, 'highs': highs,
        'spec_times': spec_times,
        'duration': len(y) / sr,
        'sample_rate': sr,
    }


def get_value_at_time(times, values, t):
    """Safely get interpolated value at time t."""
    if len(times) == 0 or len(values) == 0:
        return 0.0
    idx = np.searchsorted(times, t)
    if idx >= len(values):
        idx = len(values) - 1
    return float(values[idx])


def dance_loop(mini, analysis, stop_event, error_event, config=None):
    """Smooth tempo-synced dance loop with optional config for time switches."""
    print("[DANCE] Dance loop started")

    try:
        bass, mids, highs = analysis['bass'], analysis['mids'], analysis['highs']
        spec_times = analysis['spec_times']

        # Detected tempo from librosa (the song's actual BPM)
        detected_tempo = analysis['tempo']
        print(f"[DANCE] Detected tempo: {detected_tempo:.1f} BPM")

        # Check if config has time switches
        if config and 'time_switches' in config:
            print(f"[DANCE] Time switches loaded: {len(config['time_switches'])} switch(es)")
            for sw in config['time_switches']:
                print(f"[DANCE]   At {sw['at']}s -> {sw['time']} time")
        else:
            time_mult = TIME_MULTIPLIERS.get(CURRENT_TIME, 1.0)
            print(f"[DANCE] Groove tempo: {detected_tempo * time_mult:.1f} BPM ({CURRENT_TIME} time)")

        start_time = time.time()
        frame_count = 0
        last_status = start_time

        while not stop_event.is_set():
            t = time.time() - start_time

            # Get time multiplier (may change during song if config has time_switches)
            time_mult = get_time_multiplier_at(t, config)
            beat_freq = (detected_tempo * time_mult) / 60.0

            bass_val = get_value_at_time(spec_times, bass, t)
            mid_val = get_value_at_time(spec_times, mids, t)
            high_val = get_value_at_time(spec_times, highs, t)

            head_pitch = 10 * (0.5 + bass_val * 0.5) * np.sin(2 * np.pi * beat_freq * t)
            head_yaw = 15 * (0.4 + mid_val * 0.6) * np.sin(2 * np.pi * beat_freq / 2 * t)
            head_roll = 8 * np.sin(2 * np.pi * beat_freq * t + np.pi / 4)
            antenna_r = 0.8 * (0.3 + high_val * 0.7) * np.sin(2 * np.pi * beat_freq * 2 * t)
            antenna_l = 0.8 * (0.3 + high_val * 0.7) * np.sin(2 * np.pi * beat_freq * 2 * t + np.pi)
            body_yaw = np.deg2rad(12 * np.sin(2 * np.pi * beat_freq / 4 * t))

            pose = create_head_pose(pitch=head_pitch, yaw=head_yaw, roll=head_roll, degrees=True)
            mini.set_target(head=pose, antennas=[antenna_r, antenna_l], body_yaw=body_yaw)

            frame_count += 1

            if time.time() - last_status >= 10.0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[DANCE] Status: {elapsed:.1f}s elapsed, {frame_count} frames, {fps:.1f} FPS")
                last_status = time.time()

            time.sleep(0.016)

        total_time = time.time() - start_time
        print(f"[DANCE] Dance loop ended: {frame_count} frames in {total_time:.1f}s")

    except Exception as e:
        print(f"[DANCE] Fatal error: {e}")
        error_event.set()


def prepare_track(url, volume_level='loud'):
    """Download and prepare a track for playback."""
    global CURRENT_VOLUME
    CURRENT_VOLUME = volume_level

    is_valid, video_id, error_msg = validate_youtube_url(url)
    if not is_valid:
        return None, f"Invalid URL: {error_msg}"

    get_video_title(url)

    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, "reachy_music.%(ext)s")
    output_file = os.path.join(temp_dir, "reachy_music.wav")

    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass

    success, error_msg = download_audio(url, output_template)
    if not success:
        return None, f"Download failed: {error_msg}"

    if not os.path.exists(output_file):
        return None, "Audio file not created"

    db_boost = VOLUME_LEVELS.get(volume_level, 6)
    print(f"[AUDIO] Applying volume: {volume_level} ({db_boost:+d} dB)")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(output_file)
        boosted = audio + db_boost
        boosted.export(output_file, format="wav")
    except Exception as e:
        print(f"[WARN] Could not adjust volume: {e}")

    try:
        analysis = analyze_audio(output_file)
    except Exception as e:
        return None, f"Analysis failed: {e}"

    return output_file, analysis


def play_track(mini, output_file, analysis, stop_event, error_event, config=None):
    """Play a track with dancing."""
    duration = analysis['duration']

    dance_thread = threading.Thread(
        target=dance_loop,
        args=(mini, analysis, stop_event, error_event, config),
        daemon=True
    )

    print()
    print("=" * 60)
    print("  DANCING TO THE BEAT!")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Tempo: {analysis['tempo']:.1f} BPM")
    print(f"  Volume: {CURRENT_VOLUME}")
    print("  Press Ctrl+C to stop/skip")
    print("=" * 60)
    print()

    # Start audio FIRST, then dance
    try:
        mini.media.play_sound(output_file)
    except Exception as e:
        print(f"[WARN] Audio playback error: {e}")

    dance_thread.start()

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            if error_event.is_set() or stop_event.is_set():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Skipping track...")

    stop_event.set()
    dance_thread.join(timeout=2.0)

    print("[ROBOT] Returning to rest...")
    try:
        mini.goto_target(head=create_head_pose(), antennas=[0, 0], body_yaw=0, duration=0.5)
    except:
        pass

    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass


def interactive_prompt():
    """Show interactive prompt."""
    print()
    print("-" * 60)
    print("Commands: [URL] play track | [v]olume | [q]uit")
    print(f"Current volume: {CURRENT_VOLUME}")
    print("-" * 60)
    try:
        return input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


def change_volume():
    """Prompt user to change volume."""
    global CURRENT_VOLUME
    print("\nVolume levels: quiet (-6dB) | normal (0dB) | loud (+6dB) | max (+12dB)")
    try:
        choice = input("New volume: ").strip().lower()
        if choice in VOLUME_LEVELS:
            CURRENT_VOLUME = choice
            print(f"[VOLUME] Set to {CURRENT_VOLUME}")
    except:
        pass


# ============================================================================
# CHOREOGRAPHED DANCE MODE
# ============================================================================

def run_choreographed_mode(config: Dict, volume_level='loud'):
    """Run choreographed dance with triggers from config."""
    global CURRENT_VOLUME
    CURRENT_VOLUME = volume_level

    song_title = config.get('title', 'Unknown')
    song_url = config.get('url')

    if not song_url:
        print("[ERROR] Config missing 'url' field")
        return 1

    if 'triggers' not in config:
        print("[ERROR] Config missing 'triggers' field")
        return 1

    print()
    print("=" * 60)
    print("  CHOREOGRAPHED DANCE MODE")
    print(f"  Song: {song_title}")
    print("=" * 60)
    print()

    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, "reachy_choreo.%(ext)s")
    output_file = os.path.join(temp_dir, "reachy_choreo.wav")

    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass

    print(f"[DOWNLOAD] Fetching: {song_title}")
    success, error_msg = download_audio(song_url, output_template)
    if not success:
        print(f"[ERROR] Download failed: {error_msg}")
        return 1

    if not os.path.exists(output_file):
        print("[ERROR] Audio file not created")
        return 1

    db_boost = VOLUME_LEVELS.get(volume_level, 6)
    print(f"[AUDIO] Applying volume: {volume_level} ({db_boost:+d} dB)")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(output_file)
        boosted = audio + db_boost
        boosted.export(output_file, format="wav")
    except Exception as e:
        print(f"[WARN] Could not adjust volume: {e}")

    print("[ANALYSIS] Detecting tempo...")
    y, sr = librosa.load(output_file, sr=None)
    duration = len(y) / sr
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    beat_freq = tempo / 60.0
    print(f"[ANALYSIS] Duration: {duration:.2f}s, Tempo: {tempo:.1f} BPM")

    engine = ChoreographyEngine(config)
    num_triggers = len(engine.triggers)

    print()
    print("[ROBOT] Connecting to Reachy Mini...")

    try:
        with ReachyMini() as mini:
            print("[ROBOT] Connected!")

            try:
                mini.wake_up()
                print("[ROBOT] Reachy is awake")
            except Exception as e:
                print(f"[WARN] Wake up: {e}")

            time.sleep(1)

            print("[ROBOT] Moving to start position...")
            try:
                mini.goto_target(
                    head=create_head_pose(pitch=0, yaw=0, roll=0, degrees=True),
                    antennas=[0, 0], body_yaw=0, duration=1.0
                )
                time.sleep(1)
            except:
                pass

            print()
            print("=" * 60)
            print("  STARTING CHOREOGRAPHED DANCE!")
            print(f"  Duration: {duration:.1f}s | Tempo: {tempo:.1f} BPM")
            print(f"  {num_triggers} lyric-triggered movement sequences")
            print("  Press Ctrl+C to stop")
            print("=" * 60)
            print()

            # Start audio FIRST
            try:
                mini.media.play_sound(output_file)
            except Exception as e:
                print(f"[WARN] Audio: {e}")

            # THEN start dance loop
            start_time = time.time()
            frame_count = 0
            last_status = start_time

            try:
                while True:
                    t = time.time() - start_time
                    if t >= duration:
                        break

                    # Beat sync base (can be disabled via config)
                    if config.get('disable_beat_dance', False):
                        beat_pose = Pose(0, 0, 0, 0, 0, 0)  # Use REST_POSE values
                    else:
                        bp = 8 * 0.5 * np.sin(2 * np.pi * beat_freq * t)
                        by = 12 * 0.4 * np.sin(2 * np.pi * beat_freq / 2 * t)
                        br = 6 * np.sin(2 * np.pi * beat_freq * t + np.pi / 4)
                        ba = 0.4 + 0.2 * np.sin(2 * np.pi * beat_freq * 2 * t)
                        bb = 10 * np.sin(2 * np.pi * beat_freq / 4 * t)
                        beat_pose = Pose(bp, by, br, ba, ba, bb)
                    choreo_pose, weight = engine.update(t)
                    final = blend_choreo(beat_pose, choreo_pose, weight)

                    try:
                        pose = create_head_pose(
                            pitch=final.head_pitch, yaw=final.head_yaw,
                            roll=final.head_roll, degrees=True
                        )
                        mini.set_target(
                            head=pose,
                            antennas=[final.antenna_left, final.antenna_right],
                            body_yaw=np.deg2rad(final.body_yaw)
                        )
                    except:
                        pass

                    frame_count += 1

                    if time.time() - last_status >= 10.0:
                        fps = frame_count / t if t > 0 else 0
                        print(f"[DANCE] {t:.1f}s, {frame_count} frames, {fps:.1f} FPS, "
                              f"triggers: {engine.triggers_fired}/{num_triggers}")
                        last_status = time.time()

                    time.sleep(0.016)

            except KeyboardInterrupt:
                print("\n[INFO] Dance interrupted")

            print()
            print("[ROBOT] Returning to rest...")
            try:
                mini.goto_target(
                    head=create_head_pose(pitch=0, yaw=0, roll=0, degrees=True),
                    antennas=[0, 0], body_yaw=0, duration=1.0
                )
                time.sleep(1)
            except:
                pass

            print("[ROBOT] Going to sleep...")
            try:
                mini.goto_sleep()
            except:
                pass

            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass

            total = time.time() - start_time
            print()
            print("=" * 60)
            print("  CHOREOGRAPHED DANCE COMPLETE!")
            print(f"  {frame_count} frames in {total:.1f}s ({frame_count/total:.1f} FPS)")
            print(f"  Triggers fired: {engine.triggers_fired}/{num_triggers}")
            print("=" * 60)

    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] {e}")
        if "7447" in error_str or "Zenoh" in error_str:
            print("\nStart the daemon first:")
            print("  python start_daemon.py")
        return 1

    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    global CURRENT_VOLUME, CURRENT_TIME, CURRENT_CONFIG

    parser = argparse.ArgumentParser(description="Reachy YouTube Dance")
    parser.add_argument('url', nargs='?', help='YouTube URL to play')
    parser.add_argument('--volume', '-v', choices=['quiet', 'normal', 'loud', 'max'],
                        default='loud', help='Volume level (default: loud)')
    parser.add_argument('--time', '-t', choices=['normal', 'half', 'double', 'quarter'],
                        default='normal', help='Groove timing relative to detected BPM (default: normal)')
    parser.add_argument('--config', '-c', type=str,
                        help='Path to JSON config file for song-specific settings')
    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            CURRENT_CONFIG = config
            print(f"[CONFIG] Loaded: {config.get('title', args.config)}")
            # Use URL from config if not provided on command line
            if not args.url and 'url' in config:
                args.url = config['url']
            # Use default_time from config if not overridden
            if 'default_time' in config and args.time == 'normal':
                args.time = config['default_time']
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return 1

    # If config has triggers, run choreographed mode
    if config and 'triggers' in config:
        return run_choreographed_mode(config, args.volume)

    CURRENT_VOLUME = args.volume
    CURRENT_TIME = args.time

    print("[START] Reachy YouTube Dance - Interactive Mode")
    print(f"[START] Volume: {CURRENT_VOLUME}")
    print(f"[START] Time: {CURRENT_TIME}")
    if config:
        print(f"[START] Config: {config.get('title', 'loaded')}")
    print()

    if not check_yt_dlp_installed():
        print("[ERROR] yt-dlp not installed")
        return 1

    print("[ROBOT] Connecting to Reachy Mini...")

    try:
        with ReachyMini() as mini:
            print("[ROBOT] Connected!")

            try:
                mini.wake_up()
                print("[ROBOT] Reachy is awake")
            except Exception as e:
                print(f"[WARN] Wake up failed: {e}")

            time.sleep(1)

            pending_url = args.url

            while True:
                if pending_url:
                    url = pending_url
                    pending_url = None
                else:
                    user_input = interactive_prompt()

                    if not user_input:
                        continue
                    elif user_input.lower() in ('q', 'quit', 'exit'):
                        print("[INFO] Goodbye!")
                        break
                    elif user_input.lower() in ('v', 'volume'):
                        change_volume()
                        continue
                    else:
                        url = user_input

                print()
                print(f"[PLAY] Loading: {url}")

                result = prepare_track(url, CURRENT_VOLUME)
                if result[0] is None:
                    print(f"[ERROR] {result[1]}")
                    continue

                output_file, analysis = result
                stop_event = threading.Event()
                error_event = threading.Event()

                play_track(mini, output_file, analysis, stop_event, error_event, config)

                print()
                print("=" * 60)
                print("  TRACK COMPLETE!")
                print("=" * 60)

            print("[ROBOT] Putting Reachy to sleep...")
            try:
                mini.goto_sleep()
            except:
                pass

    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] {e}")

        if "7447" in error_str or "Zenoh" in error_str:
            print()
            print("Start the daemon first:")
            print("  python start_daemon.py")

        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print()
        print("=" * 60)
        print("  UNEXPECTED ERROR")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        traceback.print_exc()
        print()
        input("Press Enter to exit...")
        sys.exit(1)
