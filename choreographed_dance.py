"""
Choreographed Dance for Reachy Mini
====================================

Lyric-triggered choreography layered on top of beat-synced dancing.
Includes safety checks, movement limits, and dry-run testing mode.

Usage:
    python choreographed_dance.py [--dry-run] [--audio-only] [audio_file]
"""

import sys
import os
import json
import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

# ============================================================================
# CONFIGURATION & SAFETY LIMITS
# ============================================================================

@dataclass
class ReachyLimits:
    """Hardware limits for Reachy Mini - MUST NOT EXCEED."""
    # Head limits (degrees)
    HEAD_PITCH_MIN: float = -30.0
    HEAD_PITCH_MAX: float = 20.0
    HEAD_YAW_MIN: float = -45.0
    HEAD_YAW_MAX: float = 45.0
    HEAD_ROLL_MIN: float = -20.0
    HEAD_ROLL_MAX: float = 20.0

    # Antenna limits (0-1 normalized)
    ANTENNA_MIN: float = 0.0
    ANTENNA_MAX: float = 1.0

    # Body yaw limits (degrees)
    BODY_YAW_MIN: float = -15.0
    BODY_YAW_MAX: float = 15.0

    # Movement speed limits
    MAX_MOVEMENT_SPEED: float = 60.0  # degrees per second
    MIN_TRANSITION_TIME: float = 0.05  # minimum seconds between poses

    def clamp_head_pitch(self, v: float) -> float:
        return max(self.HEAD_PITCH_MIN, min(self.HEAD_PITCH_MAX, v))

    def clamp_head_yaw(self, v: float) -> float:
        return max(self.HEAD_YAW_MIN, min(self.HEAD_YAW_MAX, v))

    def clamp_head_roll(self, v: float) -> float:
        return max(self.HEAD_ROLL_MIN, min(self.HEAD_ROLL_MAX, v))

    def clamp_antenna(self, v: float) -> float:
        return max(self.ANTENNA_MIN, min(self.ANTENNA_MAX, v))

    def clamp_body_yaw(self, v: float) -> float:
        return max(self.BODY_YAW_MIN, min(self.BODY_YAW_MAX, v))


LIMITS = ReachyLimits()


# ============================================================================
# MOVEMENT DEFINITIONS
# ============================================================================

@dataclass
class Movement:
    """Defines a Reachy movement pose with safety validation."""
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    body_yaw: float = 0.0
    duration: float = 0.3
    blend_weight: float = 1.0  # How much this overrides beat-sync (0-1)

    def validate_and_clamp(self) -> 'Movement':
        """Validate and clamp all values to safe limits."""
        return Movement(
            head_pitch=LIMITS.clamp_head_pitch(self.head_pitch),
            head_yaw=LIMITS.clamp_head_yaw(self.head_yaw),
            head_roll=LIMITS.clamp_head_roll(self.head_roll),
            antenna_left=LIMITS.clamp_antenna(self.antenna_left),
            antenna_right=LIMITS.clamp_antenna(self.antenna_right),
            body_yaw=LIMITS.clamp_body_yaw(self.body_yaw),
            duration=max(LIMITS.MIN_TRANSITION_TIME, self.duration),
            blend_weight=max(0.0, min(1.0, self.blend_weight))
        )


# Rest/neutral pose
REST_POSE = Movement(
    head_pitch=0, head_yaw=0, head_roll=0,
    antenna_left=0, antenna_right=0, body_yaw=0,
    duration=0.5, blend_weight=0
)

# All choreography movements
MOVEMENTS: Dict[str, Movement] = {
    # Verse 1 - Swagger movements
    'HEAD_RAISE_PROUD': Movement(
        head_pitch=15, head_yaw=0, head_roll=0,
        antenna_left=0.9, antenna_right=0.9, body_yaw=0,
        duration=0.4, blend_weight=0.9
    ),
    'BODY_SWAY': Movement(
        head_pitch=5, head_yaw=20, head_roll=10,
        antenna_left=0.6, antenna_right=0.4, body_yaw=12,
        duration=0.5, blend_weight=0.8
    ),
    'LOOK_AROUND': Movement(
        head_pitch=0, head_yaw=35, head_roll=5,
        antenna_left=0.7, antenna_right=0.7, body_yaw=8,
        duration=0.4, blend_weight=0.85
    ),
    'SWAGGER_POSE': Movement(
        head_pitch=10, head_yaw=15, head_roll=-5,
        antenna_left=1.0, antenna_right=1.0, body_yaw=10,
        duration=0.3, blend_weight=0.9
    ),
    'HEAD_UP': Movement(
        head_pitch=18, head_yaw=0, head_roll=0,
        antenna_left=0.8, antenna_right=0.8, body_yaw=0,
        duration=0.25, blend_weight=0.7
    ),
    'FRUSTRATED_SHAKE': Movement(
        head_pitch=0, head_yaw=-15, head_roll=-8,
        antenna_left=0.3, antenna_right=0.5, body_yaw=-5,
        duration=0.2, blend_weight=0.75
    ),
    'SMOOTH_SWAY': Movement(
        head_pitch=5, head_yaw=10, head_roll=5,
        antenna_left=0.5, antenna_right=0.5, body_yaw=8,
        duration=0.4, blend_weight=0.6
    ),

    # Chorus - Pointing and mimicking
    'POINT_FORWARD': Movement(
        head_pitch=-5, head_yaw=0, head_roll=0,
        antenna_left=0.5, antenna_right=0.5, body_yaw=0,
        duration=0.2, blend_weight=0.7
    ),
    'MIMIC_GESTURE': Movement(
        head_pitch=0, head_yaw=10, head_roll=8,
        antenna_left=0.7, antenna_right=0.3, body_yaw=5,
        duration=0.25, blend_weight=0.8
    ),
    'POINT_UP': Movement(
        head_pitch=12, head_yaw=0, head_roll=0,
        antenna_left=1.0, antenna_right=1.0, body_yaw=0,
        duration=0.2, blend_weight=0.85
    ),
    'BOB_HEAD': Movement(
        head_pitch=-8, head_yaw=0, head_roll=0,
        antenna_left=0.4, antenna_right=0.4, body_yaw=0,
        duration=0.15, blend_weight=0.6
    ),
    'HEAD_TILT_TALK': Movement(
        head_pitch=5, head_yaw=20, head_roll=12,
        antenna_left=0.6, antenna_right=0.8, body_yaw=8,
        duration=0.2, blend_weight=0.75
    ),
    'NOD_AFFIRM': Movement(
        head_pitch=10, head_yaw=0, head_roll=0,
        antenna_left=0.6, antenna_right=0.6, body_yaw=0,
        duration=0.25, blend_weight=0.65
    ),
    'HUMBLE_TILT': Movement(
        head_pitch=-5, head_yaw=5, head_roll=10,
        antenna_left=0.3, antenna_right=0.4, body_yaw=3,
        duration=0.3, blend_weight=0.7
    ),
    'ASPIRING_REACH': Movement(
        head_pitch=15, head_yaw=0, head_roll=0,
        antenna_left=1.0, antenna_right=1.0, body_yaw=0,
        duration=0.4, blend_weight=0.9
    ),

    # Verse 2 - Scheming movements
    'CONSPIRATORIAL_LEAN': Movement(
        head_pitch=-8, head_yaw=25, head_roll=5,
        antenna_left=0.4, antenna_right=0.6, body_yaw=10,
        duration=0.3, blend_weight=0.8
    ),
    'ANTENNA_FLARE': Movement(
        head_pitch=10, head_yaw=0, head_roll=0,
        antenna_left=1.0, antenna_right=1.0, body_yaw=0,
        duration=0.4, blend_weight=0.95
    ),
    'HEAD_SHAKE_NO': Movement(
        head_pitch=0, head_yaw=-20, head_roll=0,
        antenna_left=0.5, antenna_right=0.5, body_yaw=-8,
        duration=0.2, blend_weight=0.7
    ),
    'YEARNING_REACH': Movement(
        head_pitch=8, head_yaw=5, head_roll=3,
        antenna_left=0.8, antenna_right=0.9, body_yaw=5,
        duration=0.35, blend_weight=0.85
    ),
    'DREAMY_SWAY': Movement(
        head_pitch=8, head_yaw=25, head_roll=15,
        antenna_left=0.5, antenna_right=0.7, body_yaw=10,
        duration=0.5, blend_weight=0.8
    ),
    'POWER_POSE': Movement(
        head_pitch=15, head_yaw=0, head_roll=0,
        antenna_left=1.0, antenna_right=1.0, body_yaw=0,
        duration=0.35, blend_weight=0.95
    ),

    # Scat - Rhythmic bouncing
    'SCAT_BOUNCE': Movement(
        head_pitch=-5, head_yaw=5, head_roll=3,
        antenna_left=0.6, antenna_right=0.4, body_yaw=5,
        duration=0.15, blend_weight=0.5
    ),
    'RHYTHMIC_BOUNCE': Movement(
        head_pitch=-10, head_yaw=0, head_roll=5,
        antenna_left=0.3, antenna_right=0.3, body_yaw=0,
        duration=0.12, blend_weight=0.6
    ),
}

# Validate all movements at load time
for name, move in MOVEMENTS.items():
    MOVEMENTS[name] = move.validate_and_clamp()


# ============================================================================
# CHOREOGRAPHY TRIGGERS (from lyrics_correction.py)
# ============================================================================

CHOREOGRAPHY_TRIGGERS = [
    # Verse 1 - Swagger and boast
    {"time": 8.50, "word": "king", "movement": "HEAD_RAISE_PROUD"},
    {"time": 9.20, "word": "swingers", "movement": "BODY_SWAY"},
    {"time": 10.58, "word": "jungle", "movement": "LOOK_AROUND"},
    {"time": 10.94, "word": "VIP", "movement": "SWAGGER_POSE"},
    {"time": 13.56, "word": "top", "movement": "HEAD_UP"},
    {"time": 14.70, "word": "stop", "movement": "HEAD_UP"},
    {"time": 16.44, "word": "botherin'", "movement": "FRUSTRATED_SHAKE"},
    {"time": 17.90, "word": "wanna", "movement": "POINT_FORWARD"},
    {"time": 18.54, "word": "man", "movement": "POINT_FORWARD"},
    {"time": 20.20, "word": "stroll", "movement": "SMOOTH_SWAY"},
    {"time": 25.18, "word": "tired", "movement": "FRUSTRATED_SHAKE"},
    {"time": 26.26, "word": "around", "movement": "BODY_SWAY"},

    # Chorus 1 - Pointing and mimicking
    {"time": 27.12, "word": "oobee", "movement": "SCAT_BOUNCE"},
    {"time": 29.80, "word": "like", "movement": "MIMIC_GESTURE"},
    {"time": 30.14, "word": "you", "movement": "POINT_UP"},
    {"time": 32.64, "word": "walk", "movement": "BOB_HEAD"},
    {"time": 32.94, "word": "like", "movement": "MIMIC_GESTURE"},
    {"time": 33.26, "word": "you", "movement": "POINT_UP"},
    {"time": 33.74, "word": "talk", "movement": "HEAD_TILT_TALK"},
    {"time": 34.02, "word": "like", "movement": "MIMIC_GESTURE"},
    {"time": 34.42, "word": "you", "movement": "POINT_UP"},
    {"time": 36.00, "word": "true", "movement": "NOD_AFFIRM"},
    {"time": 38.50, "word": "ape", "movement": "HUMBLE_TILT"},
    {"time": 40.50, "word": "human", "movement": "ASPIRING_REACH"},

    # Verse 2 - Scheming for fire
    {"time": 71.76, "word": "deal", "movement": "CONSPIRATORIAL_LEAN"},
    {"time": 75.40, "word": "secret", "movement": "CONSPIRATORIAL_LEAN"},
    {"time": 77.50, "word": "fire", "movement": "ANTENNA_FLARE"},
    {"time": 79.92, "word": "fire", "movement": "ANTENNA_FLARE"},
    {"time": 82.28, "word": "kid", "movement": "HEAD_SHAKE_NO"},
    {"time": 86.54, "word": "desire", "movement": "YEARNING_REACH"},
    {"time": 88.12, "word": "fire", "movement": "ANTENNA_FLARE"},
    {"time": 89.30, "word": "dream", "movement": "DREAMY_SWAY"},
    {"time": 91.56, "word": "secret", "movement": "CONSPIRATORIAL_LEAN"},
    {"time": 96.46, "word": "power", "movement": "POWER_POSE"},
    {"time": 97.82, "word": "flower", "movement": "ANTENNA_FLARE"},
    {"time": 99.70, "word": "you", "movement": "POINT_UP"},

    # Scat section - rhythmic bouncing
    {"time": 125.52, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 126.14, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 126.56, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 126.90, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},

    # Final chorus
    {"time": 159.22, "word": "like", "movement": "MIMIC_GESTURE"},
    {"time": 163.24, "word": "like", "movement": "MIMIC_GESTURE"},
    {"time": 167.36, "word": "learn", "movement": "ASPIRING_REACH"},
    {"time": 169.60, "word": "you", "movement": "POINT_UP"},
    {"time": 170.42, "word": "One more time", "movement": "POWER_POSE"},
    {"time": 173.02, "word": "like", "movement": "MIMIC_GESTURE"},

    # Outro scat
    {"time": 177.10, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 177.56, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 177.88, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
    {"time": 178.00, "word": "bam", "movement": "RHYTHMIC_BOUNCE"},
]


# ============================================================================
# CHOREOGRAPHY ENGINE
# ============================================================================

class ChoreographyEngine:
    """Manages choreography triggers and blending with beat-sync."""

    def __init__(self, triggers: List[dict], dry_run: bool = False):
        self.triggers = sorted(triggers, key=lambda x: x['time'])
        self.dry_run = dry_run
        self.current_trigger_idx = 0
        self.active_movement: Optional[Movement] = None
        self.movement_start_time: float = 0
        self.movement_end_time: float = 0
        self.last_logged_trigger = -1

    def reset(self):
        """Reset for new playback."""
        self.current_trigger_idx = 0
        self.active_movement = None
        self.movement_start_time = 0
        self.movement_end_time = 0
        self.last_logged_trigger = -1

    def update(self, t: float) -> Tuple[Optional[Movement], float]:
        """
        Update choreography state at time t.
        Returns (active_movement, blend_weight) or (None, 0) if no active movement.
        """
        # Check for new triggers
        while (self.current_trigger_idx < len(self.triggers) and
               self.triggers[self.current_trigger_idx]['time'] <= t):

            trigger = self.triggers[self.current_trigger_idx]
            movement_name = trigger['movement']

            if movement_name in MOVEMENTS:
                self.active_movement = MOVEMENTS[movement_name]
                self.movement_start_time = trigger['time']
                self.movement_end_time = trigger['time'] + self.active_movement.duration

                # Log trigger (only once per trigger)
                if self.current_trigger_idx != self.last_logged_trigger:
                    if self.dry_run:
                        print(f"  [{t:6.2f}s] TRIGGER: '{trigger['word']}' -> {movement_name}")
                    self.last_logged_trigger = self.current_trigger_idx

            self.current_trigger_idx += 1

        # Check if active movement has expired
        if self.active_movement and t > self.movement_end_time:
            self.active_movement = None

        # Return current state
        if self.active_movement:
            # Calculate blend weight with fade-out
            elapsed = t - self.movement_start_time
            duration = self.active_movement.duration
            if elapsed < duration * 0.7:
                # Full strength for 70% of duration
                weight = self.active_movement.blend_weight
            else:
                # Fade out over last 30%
                fade_progress = (elapsed - duration * 0.7) / (duration * 0.3)
                weight = self.active_movement.blend_weight * (1 - fade_progress)
            return self.active_movement, max(0, min(1, weight))

        return None, 0

    def get_upcoming_triggers(self, t: float, lookahead: float = 5.0) -> List[dict]:
        """Get triggers coming up in the next lookahead seconds."""
        upcoming = []
        for trigger in self.triggers:
            if t <= trigger['time'] <= t + lookahead:
                upcoming.append(trigger)
        return upcoming


# ============================================================================
# POSE BLENDING
# ============================================================================

def blend_poses(beat_sync_pose: Movement, choreo_pose: Optional[Movement],
                choreo_weight: float) -> Movement:
    """
    Blend beat-synced pose with choreography pose.
    choreo_weight of 0 = pure beat-sync, 1 = pure choreography.
    """
    if choreo_pose is None or choreo_weight <= 0:
        return beat_sync_pose

    w = choreo_weight
    inv_w = 1 - w

    blended = Movement(
        head_pitch=beat_sync_pose.head_pitch * inv_w + choreo_pose.head_pitch * w,
        head_yaw=beat_sync_pose.head_yaw * inv_w + choreo_pose.head_yaw * w,
        head_roll=beat_sync_pose.head_roll * inv_w + choreo_pose.head_roll * w,
        antenna_left=beat_sync_pose.antenna_left * inv_w + choreo_pose.antenna_left * w,
        antenna_right=beat_sync_pose.antenna_right * inv_w + choreo_pose.antenna_right * w,
        body_yaw=beat_sync_pose.body_yaw * inv_w + choreo_pose.body_yaw * w,
    )

    return blended.validate_and_clamp()


# ============================================================================
# DRY RUN TESTING
# ============================================================================

def dry_run_test(duration: float = 184.76):
    """
    Run choreography without robot - for testing timing and triggers.
    """
    print("=" * 60)
    print("DRY RUN TEST - Choreography Timing Validation")
    print("=" * 60)
    print()
    print(f"Song duration: {duration:.2f}s")
    print(f"Total triggers: {len(CHOREOGRAPHY_TRIGGERS)}")
    print()

    # Validate all movements
    print("MOVEMENT VALIDATION:")
    print("-" * 40)
    all_valid = True
    for name, move in MOVEMENTS.items():
        issues = []
        if move.head_pitch < LIMITS.HEAD_PITCH_MIN or move.head_pitch > LIMITS.HEAD_PITCH_MAX:
            issues.append(f"pitch {move.head_pitch}")
        if move.head_yaw < LIMITS.HEAD_YAW_MIN or move.head_yaw > LIMITS.HEAD_YAW_MAX:
            issues.append(f"yaw {move.head_yaw}")
        if move.head_roll < LIMITS.HEAD_ROLL_MIN or move.head_roll > LIMITS.HEAD_ROLL_MAX:
            issues.append(f"roll {move.head_roll}")
        if move.body_yaw < LIMITS.BODY_YAW_MIN or move.body_yaw > LIMITS.BODY_YAW_MAX:
            issues.append(f"body_yaw {move.body_yaw}")

        if issues:
            print(f"  [WARN] {name}: {', '.join(issues)}")
            all_valid = False
        else:
            print(f"  [OK]   {name}")

    if all_valid:
        print("\n  All movements within safe limits!")
    print()

    # Check trigger timing
    print("TRIGGER TIMING ANALYSIS:")
    print("-" * 40)

    engine = ChoreographyEngine(CHOREOGRAPHY_TRIGGERS, dry_run=True)

    # Check for overlapping triggers
    for i in range(len(CHOREOGRAPHY_TRIGGERS) - 1):
        curr = CHOREOGRAPHY_TRIGGERS[i]
        next_t = CHOREOGRAPHY_TRIGGERS[i + 1]
        move = MOVEMENTS.get(curr['movement'])
        if move:
            end_time = curr['time'] + move.duration
            if end_time > next_t['time']:
                gap = next_t['time'] - curr['time']
                print(f"  [OVERLAP] {curr['word']} ({curr['time']:.2f}s) -> "
                      f"{next_t['word']} ({next_t['time']:.2f}s), gap={gap:.2f}s")

    print()
    print("SIMULATED PLAYBACK:")
    print("-" * 40)

    # Simulate playback at 10 FPS for testing
    t = 0
    dt = 0.1
    trigger_count = 0

    while t < duration:
        movement, weight = engine.update(t)
        if movement and weight > 0:
            trigger_count += 1
        t += dt

    print(f"\n  Triggers activated: {trigger_count}")
    print(f"  Expected triggers: {len(CHOREOGRAPHY_TRIGGERS)}")
    print()

    # Summary
    print("=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)

    return all_valid


# ============================================================================
# MAIN DANCE LOOP (with robot)
# ============================================================================

def run_choreographed_dance(audio_file: str, dry_run: bool = False, audio_only: bool = False):
    """
    Run the full choreographed dance with audio and robot.
    """
    if dry_run:
        return dry_run_test()

    # Import robot libraries only when needed
    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
        import librosa
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False

    print("=" * 60)
    print("CHOREOGRAPHED DANCE - I Wanna Be Like You")
    print("=" * 60)
    print()

    # Load and analyze audio
    print("[AUDIO] Loading audio file...")
    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        return False

    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    print(f"[AUDIO] Duration: {duration:.2f}s")
    print(f"[AUDIO] Sample rate: {sr} Hz")

    # Detect tempo for beat-sync base
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    beat_freq = tempo / 60.0
    print(f"[AUDIO] Tempo: {tempo:.1f} BPM")
    print()

    # Initialize choreography engine
    engine = ChoreographyEngine(CHOREOGRAPHY_TRIGGERS)

    # Connect to robot
    print("[ROBOT] Connecting to Reachy Mini...")
    try:
        with ReachyMini() as mini:
            print("[ROBOT] Connected!")

            # Wake up sequence
            print("[ROBOT] Waking up...")
            try:
                mini.wake_up()
                time.sleep(1.0)
                print("[ROBOT] Awake and ready!")
            except Exception as e:
                print(f"[WARN] Wake up issue: {e}")

            # Move to starting pose
            print("[ROBOT] Moving to start position...")
            try:
                mini.goto_target(
                    head=create_head_pose(pitch=0, yaw=0, roll=0, degrees=True),
                    antennas=[0, 0],
                    body_yaw=0,
                    duration=1.0
                )
                time.sleep(1.0)
            except Exception as e:
                print(f"[WARN] Start position issue: {e}")

            # Start audio playback
            print()
            print("=" * 60)
            print("  STARTING CHOREOGRAPHED DANCE!")
            print(f"  Duration: {duration:.1f}s")
            print("  Press Ctrl+C to stop")
            print("=" * 60)
            print()

            # Play audio
            if not audio_only:
                try:
                    mini.media.play_sound(audio_file)
                except Exception as e:
                    print(f"[WARN] Audio playback error: {e}")

            # Dance loop
            start_time = time.time()
            frame_count = 0
            last_status = start_time

            stop_requested = False

            try:
                while not stop_requested:
                    t = time.time() - start_time

                    if t >= duration:
                        break

                    # Calculate beat-sync base pose
                    head_pitch = 10 * 0.5 * np.sin(2 * np.pi * beat_freq * t)
                    head_yaw = 15 * 0.4 * np.sin(2 * np.pi * beat_freq / 2 * t)
                    head_roll = 8 * np.sin(2 * np.pi * beat_freq * t + np.pi / 4)
                    antenna_val = 0.5 + 0.3 * np.sin(2 * np.pi * beat_freq * 2 * t)
                    body_yaw_rad = np.deg2rad(12 * np.sin(2 * np.pi * beat_freq / 4 * t))

                    beat_sync_pose = Movement(
                        head_pitch=head_pitch,
                        head_yaw=head_yaw,
                        head_roll=head_roll,
                        antenna_left=antenna_val,
                        antenna_right=antenna_val,
                        body_yaw=np.rad2deg(body_yaw_rad)
                    )

                    # Get choreography override
                    choreo_pose, choreo_weight = engine.update(t)

                    # Blend poses
                    final_pose = blend_poses(beat_sync_pose, choreo_pose, choreo_weight)

                    # Apply to robot
                    try:
                        pose = create_head_pose(
                            pitch=final_pose.head_pitch,
                            yaw=final_pose.head_yaw,
                            roll=final_pose.head_roll,
                            degrees=True
                        )
                        mini.set_target(
                            head=pose,
                            antennas=[final_pose.antenna_left, final_pose.antenna_right],
                            body_yaw=np.deg2rad(final_pose.body_yaw)
                        )
                    except Exception as e:
                        print(f"[DANCE] Pose error: {e}")

                    frame_count += 1

                    # Status update every 10 seconds
                    if time.time() - last_status >= 10.0:
                        fps = frame_count / t if t > 0 else 0
                        print(f"[DANCE] {t:.1f}s elapsed, {frame_count} frames, {fps:.1f} FPS")
                        last_status = time.time()

                    time.sleep(0.016)  # ~60 FPS

            except KeyboardInterrupt:
                print("\n[INFO] Dance interrupted by user")
                stop_requested = True

            # Return to rest
            print()
            print("[ROBOT] Returning to rest position...")
            try:
                mini.goto_target(
                    head=create_head_pose(pitch=0, yaw=0, roll=0, degrees=True),
                    antennas=[0, 0],
                    body_yaw=0,
                    duration=1.0
                )
                time.sleep(1.0)
            except Exception as e:
                print(f"[WARN] Rest position issue: {e}")

            # Sleep sequence
            print("[ROBOT] Going to sleep...")
            try:
                mini.goto_sleep()
            except Exception as e:
                print(f"[WARN] Sleep issue: {e}")

            print()
            print("=" * 60)
            print("  DANCE COMPLETE!")
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"  {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
            print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return False

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Choreographed Dance for Reachy Mini")
    parser.add_argument('audio_file', nargs='?', default='choreography_audio.wav',
                        help='Audio file to play')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test choreography timing without robot')
    parser.add_argument('--audio-only', action='store_true',
                        help='Run without audio playback (robot moves only)')
    args = parser.parse_args()

    if args.dry_run:
        success = dry_run_test()
    else:
        success = run_choreographed_dance(
            args.audio_file,
            dry_run=False,
            audio_only=args.audio_only
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
