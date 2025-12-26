"""
Reachy YouTube Dance Script
===========================
Makes Reachy Mini dance to YouTube videos with beat-synced movements.

Usage:
    python reachy_youtube_dance.py [URL] [--volume normal|loud|max]

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
import argparse
import tempfile
import threading
import time
import wave
import subprocess
import traceback

VOLUME_LEVELS = {
    'normal': 0,
    'loud': 6,
    'max': 12,
    'quiet': -6,
}

CURRENT_VOLUME = 'loud'

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


def validate_youtube_url(url):
    """
    Validate and extract video ID from various YouTube URL formats.
    Returns (is_valid, video_id, error_message)
    """
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
        return False, None, "URL contains 'youtube' but video ID could not be extracted. Check URL format."

    return False, None, "URL does not appear to be a valid YouTube link"


def check_yt_dlp_installed():
    """Check if yt-dlp is available."""
    print("[CHECK] Verifying yt-dlp installation...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[CHECK] yt-dlp version: {version}")
            return True
        else:
            print(f"[ERROR] yt-dlp check failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] yt-dlp version check timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to check yt-dlp: {e}")
        return False


def get_video_title(url):
    """Fetch video title for display."""
    print("[INFO] Fetching video title...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--get-title", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            title = result.stdout.strip()
            print(f"[INFO] Video title: {title}")
            return title
        else:
            print("[WARN] Could not fetch video title")
            return None
    except subprocess.TimeoutExpired:
        print("[WARN] Video title fetch timed out")
        return None
    except Exception as e:
        print(f"[WARN] Error fetching video title: {e}")
        return None


def download_audio(url, output_path, timeout=300):
    """
    Download audio from YouTube URL.
    Returns (success, error_message)
    """
    print("[DOWNLOAD] Starting audio download...")
    print(f"[DOWNLOAD] URL: {url}")
    print(f"[DOWNLOAD] Output template: {output_path}")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_path,
        "--no-playlist",
        "--no-warnings",
        "--progress",
        url
    ]

    print(f"[DOWNLOAD] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            print("[DOWNLOAD] Download completed successfully")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-3:]:
                    if line.strip():
                        print(f"[DOWNLOAD] {line.strip()}")
            return True, None
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"[ERROR] Download failed with code {result.returncode}")
            print(f"[ERROR] {error_msg}")

            if "Video unavailable" in error_msg:
                return False, "Video is unavailable (private, deleted, or region-locked)"
            elif "age-restricted" in error_msg.lower():
                return False, "Video is age-restricted and cannot be downloaded"
            elif "copyright" in error_msg.lower():
                return False, "Video blocked due to copyright"
            elif "Sign in" in error_msg:
                return False, "Video requires sign-in to access"
            else:
                return False, error_msg

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Download timed out after {timeout} seconds")
        return False, f"Download timed out after {timeout} seconds"
    except Exception as e:
        print(f"[ERROR] Download exception: {e}")
        return False, str(e)


def get_wav_duration(filepath):
    """Get duration of WAV file in seconds."""
    print(f"[AUDIO] Reading WAV file: {filepath}")
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration = frames / float(rate)

            print(f"[AUDIO] Sample rate: {rate} Hz")
            print(f"[AUDIO] Channels: {channels}")
            print(f"[AUDIO] Sample width: {sampwidth * 8} bits")
            print(f"[AUDIO] Total frames: {frames}")
            print(f"[AUDIO] Duration: {duration:.2f} seconds")

            return duration
    except wave.Error as e:
        print(f"[ERROR] Invalid WAV file: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Error reading WAV file: {e}")
        raise


def analyze_audio(filepath):
    """
    Analyze audio file for beat detection and spectral features.
    Uses high-resolution analysis for precise beat-reactive movement.
    Returns analysis dict or raises exception.
    """
    print()
    print("[ANALYSIS] Starting HIGH-RESOLUTION audio analysis...")
    print(f"[ANALYSIS] File: {filepath}")

    file_size = os.path.getsize(filepath)
    print(f"[ANALYSIS] File size: {file_size / (1024*1024):.2f} MB")

    print("[ANALYSIS] Loading audio into memory...")
    start_time = time.time()

    try:
        y, sr = librosa.load(filepath, sr=None)
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        raise

    load_time = time.time() - start_time
    print(f"[ANALYSIS] Audio loaded in {load_time:.2f} seconds")
    print(f"[ANALYSIS] Sample rate: {sr} Hz")
    print(f"[ANALYSIS] Samples: {len(y)}")
    print(f"[ANALYSIS] Duration: {len(y)/sr:.2f} seconds")

    hop_length = 256
    frame_rate = sr / hop_length
    print(f"[ANALYSIS] Using hop_length={hop_length} for {frame_rate:.1f} Hz temporal resolution")

    print("[ANALYSIS] Detecting tempo and beats...")
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    except Exception as e:
        print(f"[ERROR] Beat detection failed: {e}")
        raise

    print("[ANALYSIS] Computing onset strength envelope...")
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        onset_max = onset_env.max()
        if onset_max > 0:
            onset_env = onset_env / onset_max
        else:
            print("[WARN] Onset envelope is all zeros")
        print(f"[ANALYSIS] Onset envelope: {len(onset_env)} frames at {frame_rate:.1f} Hz")
    except Exception as e:
        print(f"[ERROR] Onset detection failed: {e}")
        raise

    print("[ANALYSIS] Detecting individual onsets (transients)...")
    try:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length,
            backtrack=True, units='frames'
        )
        onset_positions = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        print(f"[ANALYSIS] Found {len(onset_positions)} individual onsets/transients")
    except Exception as e:
        print(f"[WARN] Onset detection failed: {e}")
        onset_positions = np.array([])

    print("[ANALYSIS] Computing high-resolution spectral features...")
    try:
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr)

        bass_mask = freqs < 150
        low_mid_mask = (freqs >= 150) & (freqs < 500)
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mid_mask = (freqs >= 2000) & (freqs < 6000)
        high_mask = freqs >= 6000

        print(f"[ANALYSIS] Frequency bands:")
        print(f"[ANALYSIS]   Sub-bass (<150 Hz): {bass_mask.sum()} bins")
        print(f"[ANALYSIS]   Low-mid (150-500 Hz): {low_mid_mask.sum()} bins")
        print(f"[ANALYSIS]   Mid (500-2000 Hz): {mid_mask.sum()} bins")
        print(f"[ANALYSIS]   High-mid (2-6 kHz): {high_mid_mask.sum()} bins")
        print(f"[ANALYSIS]   High (>6 kHz): {high_mask.sum()} bins")

        bass = S[bass_mask, :].mean(axis=0)
        low_mids = S[low_mid_mask, :].mean(axis=0)
        mids = S[mid_mask, :].mean(axis=0)
        high_mids = S[high_mid_mask, :].mean(axis=0)
        highs = S[high_mask, :].mean(axis=0)

        def normalize(arr):
            m = arr.max()
            return arr / m if m > 0 else arr

        bass = normalize(bass)
        low_mids = normalize(low_mids)
        mids = normalize(mids)
        high_mids = normalize(high_mids)
        highs = normalize(highs)

        spec_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    except Exception as e:
        print(f"[ERROR] Spectral analysis failed: {e}")
        raise

    print("[ANALYSIS] Computing beat-synchronous features...")
    try:
        beat_onset_strength = np.zeros(len(beat_times))
        for i, bt in enumerate(beat_times):
            idx = np.searchsorted(onset_times, bt)
            if idx < len(onset_env):
                window_start = max(0, idx - 2)
                window_end = min(len(onset_env), idx + 3)
                beat_onset_strength[i] = onset_env[window_start:window_end].max()
        print(f"[ANALYSIS] Beat onset strengths computed for {len(beat_times)} beats")
    except Exception as e:
        print(f"[WARN] Beat-sync computation failed: {e}")
        beat_onset_strength = np.ones(len(beat_times))

    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    if tempo < 30 or tempo > 300:
        print(f"[WARN] Unusual tempo detected: {tempo:.1f} BPM")

    analysis_time = time.time() - start_time

    print()
    print("[ANALYSIS] === HIGH-RES RESULTS ===")
    print(f"[ANALYSIS] Tempo: {tempo:.1f} BPM")
    print(f"[ANALYSIS] Beats detected: {len(beat_times)}")
    print(f"[ANALYSIS] Onsets detected: {len(onset_positions)}")
    print(f"[ANALYSIS] Temporal resolution: {frame_rate:.1f} Hz ({1000/frame_rate:.1f} ms)")
    print(f"[ANALYSIS] Analysis time: {analysis_time:.2f} seconds")
    print()

    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'beat_strength': beat_onset_strength,
        'onset_positions': onset_positions,
        'onset_env': onset_env,
        'onset_times': onset_times,
        'bass': bass,
        'low_mids': low_mids,
        'mids': mids,
        'high_mids': high_mids,
        'highs': highs,
        'spec_times': spec_times,
        'duration': len(y) / sr,
        'sample_rate': sr,
        'hop_length': hop_length,
        'frame_rate': frame_rate
    }


def get_value_at_time(times, values, t):
    """Safely get interpolated value at time t."""
    if len(times) == 0 or len(values) == 0:
        return 0.0
    idx = np.searchsorted(times, t)
    if idx >= len(values):
        idx = len(values) - 1
    if idx < 0:
        idx = 0
    return float(values[idx])


def dance_loop(mini, analysis, stop_event, error_event):
    """
    Smooth tempo-synced dance loop.
    Uses detected BPM for groove with spectral modulation.
    """
    print("[DANCE] Dance loop started")

    try:
        bass = analysis['bass']
        mids = analysis['mids']
        highs = analysis['highs']
        spec_times = analysis['spec_times']
        tempo = analysis['tempo']

        beat_freq = tempo / 60.0
        half_beat_freq = beat_freq / 2.0

        print(f"[DANCE] Tempo: {tempo:.1f} BPM")
        print(f"[DANCE] Beat frequency: {beat_freq:.2f} Hz")

        start_time = time.time()
        frame_count = 0
        last_status_time = start_time

        while not stop_event.is_set():
            t = time.time() - start_time

            try:
                bass_val = get_value_at_time(spec_times, bass, t)
                mid_val = get_value_at_time(spec_times, mids, t)
                high_val = get_value_at_time(spec_times, highs, t)
            except Exception as e:
                print(f"[DANCE] Error getting spectral values: {e}")
                bass_val = mid_val = high_val = 0.5

            try:
                head_pitch = 10 * (0.5 + bass_val * 0.5) * np.sin(2 * np.pi * beat_freq * t)
                head_yaw = 15 * (0.4 + mid_val * 0.6) * np.sin(2 * np.pi * half_beat_freq * t)
                head_roll = 8 * np.sin(2 * np.pi * beat_freq * t + np.pi / 4)

                antenna_r = 0.8 * (0.3 + high_val * 0.7) * np.sin(2 * np.pi * beat_freq * 2 * t)
                antenna_l = 0.8 * (0.3 + high_val * 0.7) * np.sin(2 * np.pi * beat_freq * 2 * t + np.pi)

                body_yaw = np.deg2rad(12 * np.sin(2 * np.pi * half_beat_freq * t / 2))

                pose = create_head_pose(pitch=head_pitch, yaw=head_yaw, roll=head_roll, degrees=True)
                mini.set_target(head=pose, antennas=[antenna_r, antenna_l], body_yaw=body_yaw)

            except Exception as e:
                print(f"[DANCE] Error setting robot pose: {e}")
                error_event.set()
                break

            frame_count += 1

            if time.time() - last_status_time >= 10.0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[DANCE] Status: {elapsed:.1f}s elapsed, {frame_count} frames, {fps:.1f} FPS")
                last_status_time = time.time()

            time.sleep(0.016)

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"[DANCE] Dance loop ended: {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")

    except Exception as e:
        print(f"[DANCE] Fatal error in dance loop: {e}")
        traceback.print_exc()
        error_event.set()


def check_daemon_connection():
    """Check if Reachy daemon is accessible."""
    print("[CHECK] Testing connection to Reachy daemon...")
    try:
        with ReachyMini() as mini:
            print("[CHECK] Successfully connected to daemon")
            return True
    except Exception as e:
        error_str = str(e)
        if "7447" in error_str or "Zenoh" in error_str:
            print("[ERROR] Cannot connect to Reachy daemon")
            print("[ERROR] Make sure the daemon is running:")
            print("[ERROR]   D:\\reachy_mini_env\\Scripts\\python.exe D:\\run_reachy_daemon.py")
        else:
            print(f"[ERROR] Connection test failed: {e}")
        return False


def prepare_track(url, volume_level='loud'):
    """Download and prepare a track for playback. Returns (output_file, analysis) or (None, error_msg)."""
    global CURRENT_VOLUME
    CURRENT_VOLUME = volume_level

    is_valid, video_id, error_msg = validate_youtube_url(url)
    if not is_valid:
        return None, f"Invalid URL: {error_msg}"

    url = url.strip()
    get_video_title(url)

    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, "reachy_music.%(ext)s")
    output_file = os.path.join(temp_dir, "reachy_music.wav")

    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass

    print()
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
        print(f"[AUDIO] Volume adjusted: {db_boost:+d} dB")
    except Exception as e:
        print(f"[WARN] Could not adjust volume: {e}")

    print()
    try:
        analysis = analyze_audio(output_file)
    except Exception as e:
        return None, f"Analysis failed: {e}"

    return output_file, analysis


def play_track(mini, output_file, analysis, stop_event, error_event):
    """Play a track with dancing. Returns when complete or interrupted."""
    duration = analysis['duration']

    dance_thread = threading.Thread(
        target=dance_loop,
        args=(mini, analysis, stop_event, error_event),
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

    dance_thread.start()

    try:
        mini.media.play_sound(output_file)
    except Exception as e:
        print(f"[WARN] Audio playback error: {e}")

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            if error_event.is_set():
                print("[ERROR] Dance loop error")
                break
            if stop_event.is_set():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        print("[INFO] Skipping track...")

    stop_event.set()
    dance_thread.join(timeout=2.0)

    print("[ROBOT] Returning to rest...")
    try:
        mini.goto_target(
            head=create_head_pose(),
            antennas=[0, 0],
            body_yaw=0,
            duration=0.5
        )
    except:
        pass

    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass


def interactive_prompt():
    """Show interactive prompt and get user input."""
    print()
    print("-" * 60)
    print("Commands: [URL] play track | [v]olume | [q]uit")
    print(f"Current volume: {CURRENT_VOLUME}")
    print("-" * 60)
    try:
        return input("> ").strip()
    except EOFError:
        return "q"
    except KeyboardInterrupt:
        print()
        return ""


def change_volume():
    """Prompt user to change volume."""
    global CURRENT_VOLUME
    print()
    print("Volume levels: quiet (-6dB) | normal (0dB) | loud (+6dB) | max (+12dB)")
    print(f"Current: {CURRENT_VOLUME}")
    try:
        choice = input("New volume: ").strip().lower()
        if choice in VOLUME_LEVELS:
            CURRENT_VOLUME = choice
            print(f"[VOLUME] Set to {CURRENT_VOLUME}")
        else:
            print("[WARN] Invalid choice, keeping current volume")
    except:
        pass


def main():
    """Main entry point with interactive loop."""
    global CURRENT_VOLUME

    parser = argparse.ArgumentParser(description="Reachy YouTube Dance")
    parser.add_argument('url', nargs='?', help='YouTube URL to play')
    parser.add_argument('--volume', '-v', choices=['quiet', 'normal', 'loud', 'max'],
                        default='loud', help='Volume level (default: loud)')
    args = parser.parse_args()

    CURRENT_VOLUME = args.volume

    print("[START] Reachy YouTube Dance - Interactive Mode")
    print(f"[START] Volume: {CURRENT_VOLUME}")
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

                play_track(mini, output_file, analysis, stop_event, error_event)

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
            print("  D:\\reachy_mini_env\\Scripts\\python.exe D:\\run_reachy_daemon.py")

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
