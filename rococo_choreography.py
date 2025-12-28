"""
Rococo Choreography Module for Reachy Mini
===========================================

Lyric-triggered choreography for "I Wanna Be Like You" (The Jungle Book)
with multi-phase sequences, anticipation, follow-through, and asymmetric personality.

This module contains all hardcoded choreography data including:
- YouTube URL for the song
- Whisper transcription timestamps
- 50 choreography triggers with multi-phase movement sequences
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# ============================================================================
# HARDCODED SONG DATA
# ============================================================================

# "I Wanna Be Like You" - The Jungle Book (King Louie)
ROCOCO_SONG_URL = "https://www.youtube.com/watch?v=GokDMNue2Xc"
ROCOCO_SONG_TITLE = "I Wanna Be Like You - The Jungle Book"
ROCOCO_SONG_DURATION = 184.76  # seconds

# Whisper transcription timestamps (word-level, corrected)
ROCOCO_TRANSCRIPTION = {
    "language": "en",
    "duration": 184.76,
    "key_timestamps": {
        "king": 8.50,
        "swingers": 9.20,
        "jungle": 10.58,
        "VIP": 10.94,
        "top": 13.56,
        "stop": 14.70,
        "botherin": 16.44,
        "wanna_1": 17.90,
        "man": 18.54,
        "stroll": 20.20,
        "tired": 25.18,
        "around": 26.26,
        "oobee": 27.12,
        "like_1": 29.80,
        "you_1": 30.14,
        "walk": 32.64,
        "like_2": 32.94,
        "you_2": 33.26,
        "talk": 33.74,
        "like_3": 34.02,
        "you_3": 34.42,
        "true": 36.00,
        "ape": 38.50,
        "human": 40.50,
        "deal": 71.76,
        "secret_1": 75.40,
        "fire_1": 77.50,
        "fire_2": 79.92,
        "kid": 82.28,
        "desire": 86.54,
        "fire_3": 88.12,
        "dream": 89.30,
        "secret_2": 91.56,
        "power": 96.46,
        "flower": 97.82,
        "you_4": 99.70,
        "bam_1": 125.52,
        "bam_2": 126.14,
        "bam_3": 126.56,
        "bam_4": 126.90,
        "like_4": 159.22,
        "like_5": 163.24,
        "learn": 167.36,
        "you_5": 169.60,
        "one_more_time": 170.42,
        "like_6": 173.02,
        "bam_5": 177.10,
        "bam_6": 177.56,
        "bam_7": 177.88,
        "bam_8": 178.00,
    }
}


# ============================================================================
# SAFETY LIMITS
# ============================================================================

@dataclass
class ReachyLimits:
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


LIMITS = ReachyLimits()


# ============================================================================
# POSE & SEQUENCE CLASSES
# ============================================================================

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
        """Linear interpolation between poses."""
        t = max(0, min(1, t))
        return Pose(
            head_pitch=self.head_pitch + (other.head_pitch - self.head_pitch) * t,
            head_yaw=self.head_yaw + (other.head_yaw - self.head_yaw) * t,
            head_roll=self.head_roll + (other.head_roll - self.head_roll) * t,
            antenna_left=self.antenna_left + (other.antenna_left - self.antenna_left) * t,
            antenna_right=self.antenna_right + (other.antenna_right - self.antenna_right) * t,
            body_yaw=self.body_yaw + (other.body_yaw - self.body_yaw) * t,
        )

    def clamped(self) -> 'Pose':
        """Return pose with values clamped to safe limits."""
        p, y, r, al, ar, by = LIMITS.clamp(
            self.head_pitch, self.head_yaw, self.head_roll,
            self.antenna_left, self.antenna_right, self.body_yaw
        )
        return Pose(p, y, r, al, ar, by)


REST = Pose(0, 0, 0, 0, 0, 0)


@dataclass
class SequencePhase:
    """A phase within a movement sequence."""
    pose: Pose
    duration: float
    ease: str = "smooth"  # "linear", "smooth", "snap", "ease_in", "ease_out"


@dataclass
class MovementSequence:
    """A multi-phase movement with anticipation and follow-through."""
    phases: List[SequencePhase]
    blend_weight: float = 0.9

    def get_total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def get_pose_at(self, elapsed: float) -> Tuple[Pose, float]:
        """Get interpolated pose at elapsed time."""
        if not self.phases:
            return REST, 0

        t = 0
        for i, phase in enumerate(self.phases):
            if elapsed < t + phase.duration:
                phase_progress = (elapsed - t) / phase.duration if phase.duration > 0 else 1

                if phase.ease == "smooth":
                    phase_progress = phase_progress * phase_progress * (3 - 2 * phase_progress)
                elif phase.ease == "snap":
                    phase_progress = 1 if phase_progress > 0.3 else phase_progress / 0.3
                elif phase.ease == "ease_in":
                    phase_progress = phase_progress * phase_progress
                elif phase.ease == "ease_out":
                    phase_progress = 1 - (1 - phase_progress) ** 2

                prev_pose = self.phases[i-1].pose if i > 0 else REST
                return prev_pose.lerp(phase.pose, phase_progress), self.blend_weight
            t += phase.duration

        final_pose = self.phases[-1].pose
        overshoot = elapsed - t
        decay = max(0, 1 - overshoot / 0.3)
        return final_pose.lerp(REST, 1 - decay), self.blend_weight * decay


# ============================================================================
# MOVEMENT SEQUENCE FACTORIES
# ============================================================================

def create_king_sequence() -> MovementSequence:
    """King of the swingers - regal head raise + slow turn + crown antennas."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-5, 0, 0, 0.2, 0.2, 0), 0.1, "smooth"),
            SequencePhase(Pose(15, 0, 0, 0.5, 0.5, 0), 0.2, "ease_out"),
            SequencePhase(Pose(18, 20, 3, 0.9, 0.95, 5), 0.3, "smooth"),
            SequencePhase(Pose(15, 15, 0, 1.0, 1.0, 3), 0.3, "smooth"),
            SequencePhase(Pose(8, 5, 0, 0.7, 0.7, 0), 0.2, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_fire_sequence() -> MovementSequence:
    """Man's red fire - flicker -> FLARE -> hold."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-3, 0, 0, 0.2, 0.3, 0), 0.08, "snap"),
            SequencePhase(Pose(5, 0, 0, 0.6, 0.4, 0), 0.06, "snap"),
            SequencePhase(Pose(3, 0, 0, 0.4, 0.7, 0), 0.06, "snap"),
            SequencePhase(Pose(6, 0, 0, 0.8, 0.5, 0), 0.06, "snap"),
            SequencePhase(Pose(15, 0, 0, 1.0, 1.0, 0), 0.12, "snap"),
            SequencePhase(Pose(12, 0, 2, 1.0, 1.0, 0), 0.25, "smooth"),
            SequencePhase(Pose(8, 0, 0, 0.8, 0.8, 0), 0.15, "ease_in"),
        ],
        blend_weight=0.98
    )


def create_like_you_left() -> MovementSequence:
    """Like you - mimicking gesture, left variation."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(0, -5, -3, 0.4, 0.5, -2), 0.08, "smooth"),
            SequencePhase(Pose(5, -15, 10, 0.8, 0.4, -5), 0.15, "ease_out"),
            SequencePhase(Pose(10, -10, 8, 0.9, 0.5, -3), 0.12, "smooth"),
            SequencePhase(Pose(3, -5, 3, 0.6, 0.5, -1), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_like_you_right() -> MovementSequence:
    """Like you - mimicking gesture, right variation."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(0, 5, 3, 0.5, 0.4, 2), 0.08, "smooth"),
            SequencePhase(Pose(5, 15, -10, 0.4, 0.8, 5), 0.15, "ease_out"),
            SequencePhase(Pose(10, 10, -8, 0.5, 0.9, 3), 0.12, "smooth"),
            SequencePhase(Pose(3, 5, -3, 0.5, 0.6, 1), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_walk_strut() -> MovementSequence:
    """Walk like you - exaggerated strut bobbing."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-10, 8, 5, 0.3, 0.5, 4), 0.1, "snap"),
            SequencePhase(Pose(5, -5, -3, 0.6, 0.4, -2), 0.1, "ease_out"),
            SequencePhase(Pose(3, 10, 6, 0.5, 0.6, 3), 0.1, "smooth"),
        ],
        blend_weight=0.75
    )


def create_talk_sassy() -> MovementSequence:
    """Talk like you - sassy head tilts."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(3, 20, 12, 0.5, 0.7, 6), 0.1, "ease_out"),
            SequencePhase(Pose(5, 25, 15, 0.6, 0.8, 8), 0.12, "smooth"),
            SequencePhase(Pose(0, 15, 8, 0.5, 0.6, 4), 0.1, "ease_in"),
        ],
        blend_weight=0.8
    )


def create_swingers_swing() -> MovementSequence:
    """Swingers - full body swing with antenna countermotion."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(3, -25, -8, 0.8, 0.3, -10), 0.15, "smooth"),
            SequencePhase(Pose(8, 0, 0, 0.5, 0.5, 0), 0.1, "ease_out"),
            SequencePhase(Pose(5, 30, 10, 0.3, 0.9, 12), 0.2, "smooth"),
            SequencePhase(Pose(5, 10, 3, 0.5, 0.6, 4), 0.15, "ease_in"),
        ],
        blend_weight=0.9
    )


def create_bam_bounce_left() -> MovementSequence:
    """Bam - bounce left."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-8, -10, -5, 0.2, 0.5, -5), 0.06, "snap"),
            SequencePhase(Pose(0, -3, 0, 0.4, 0.4, -1), 0.06, "ease_out"),
        ],
        blend_weight=0.7
    )


def create_bam_bounce_right() -> MovementSequence:
    """Bam - bounce right."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-8, 10, 5, 0.5, 0.2, 5), 0.06, "snap"),
            SequencePhase(Pose(0, 3, 0, 0.4, 0.4, 1), 0.06, "ease_out"),
        ],
        blend_weight=0.7
    )


def create_bam_bounce_center() -> MovementSequence:
    """Bam - bounce center (building energy)."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-12, 0, 0, 0.6, 0.6, 0), 0.05, "snap"),
            SequencePhase(Pose(5, 0, 0, 0.8, 0.8, 0), 0.05, "snap"),
        ],
        blend_weight=0.75
    )


def create_dream_float() -> MovementSequence:
    """Dream - slow dreamy roll with asymmetric antenna float."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(8, 10, 5, 0.4, 0.7, 3), 0.2, "ease_out"),
            SequencePhase(Pose(10, 20, 12, 0.5, 0.9, 6), 0.25, "smooth"),
            SequencePhase(Pose(12, 25, 15, 0.6, 1.0, 8), 0.2, "smooth"),
            SequencePhase(Pose(8, 15, 8, 0.5, 0.8, 5), 0.2, "ease_in"),
            SequencePhase(Pose(5, 8, 3, 0.4, 0.6, 2), 0.15, "smooth"),
        ],
        blend_weight=0.85
    )


def create_vip_snap() -> MovementSequence:
    """VIP - dramatic pause then snap into pose."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-2, 0, 0, 0.3, 0.3, 0), 0.15, "linear"),
            SequencePhase(Pose(12, 18, -5, 1.0, 1.0, 8), 0.08, "snap"),
            SequencePhase(Pose(10, 15, -3, 0.95, 0.95, 6), 0.2, "smooth"),
            SequencePhase(Pose(5, 8, 0, 0.7, 0.7, 3), 0.12, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_point_up_emphatic() -> MovementSequence:
    """Point up on 'you' - emphatic with asymmetry."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-3, 0, 0, 0.3, 0.4, 0), 0.06, "smooth"),
            SequencePhase(Pose(15, 0, 0, 1.0, 0.95, 0), 0.1, "snap"),
            SequencePhase(Pose(12, 0, 2, 0.95, 1.0, 0), 0.12, "smooth"),
            SequencePhase(Pose(5, 0, 0, 0.6, 0.6, 0), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_power_pose() -> MovementSequence:
    """Power - strong pose with buildup."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-5, 0, 0, 0.3, 0.3, 0), 0.1, "smooth"),
            SequencePhase(Pose(18, 0, 0, 1.0, 1.0, 0), 0.12, "snap"),
            SequencePhase(Pose(15, 0, 2, 1.0, 1.0, 0), 0.2, "smooth"),
            SequencePhase(Pose(10, 0, 0, 0.8, 0.8, 0), 0.1, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_frustrated_shake() -> MovementSequence:
    """Frustrated - shake with personality."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(0, -18, -6, 0.4, 0.6, -4), 0.08, "snap"),
            SequencePhase(Pose(0, 18, 6, 0.6, 0.4, 4), 0.08, "snap"),
            SequencePhase(Pose(0, -10, -3, 0.5, 0.5, -2), 0.06, "snap"),
            SequencePhase(Pose(-3, 0, 0, 0.4, 0.4, 0), 0.08, "ease_in"),
        ],
        blend_weight=0.8
    )


def create_conspiratorial_lean() -> MovementSequence:
    """Conspiratorial - sneaky lean in."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(0, -20, 0, 0.5, 0.5, -5), 0.12, "smooth"),
            SequencePhase(Pose(0, 15, 0, 0.5, 0.5, 3), 0.1, "smooth"),
            SequencePhase(Pose(-8, 25, 8, 0.4, 0.6, 8), 0.15, "ease_out"),
            SequencePhase(Pose(-5, 20, 5, 0.5, 0.6, 6), 0.15, "smooth"),
        ],
        blend_weight=0.85
    )


def create_aspiring_reach() -> MovementSequence:
    """Aspiring/human - reaching upward hopefully."""
    return MovementSequence(
        phases=[
            SequencePhase(Pose(-5, 0, 0, 0.2, 0.2, 0), 0.1, "smooth"),
            SequencePhase(Pose(15, 5, 3, 0.8, 0.9, 2), 0.2, "ease_out"),
            SequencePhase(Pose(18, 0, 0, 1.0, 1.0, 0), 0.15, "smooth"),
            SequencePhase(Pose(10, 0, 0, 0.7, 0.8, 0), 0.15, "ease_in"),
        ],
        blend_weight=0.9
    )


# ============================================================================
# CHOREOGRAPHY ENGINE
# ============================================================================

class RococoChoreographyEngine:
    """Enhanced choreography engine with sequences and state tracking."""

    def __init__(self):
        self.active_sequence: Optional[MovementSequence] = None
        self.sequence_start_time: float = 0
        self.like_you_counter: int = 0
        self.bam_counter: int = 0
        self.triggers_fired: int = 0
        self.triggers = self._build_triggers()
        self.current_trigger_idx = 0

    def _build_triggers(self) -> List[dict]:
        """Build the trigger list with sequence factories."""
        ts = ROCOCO_TRANSCRIPTION["key_timestamps"]
        return [
            {"time": ts["king"], "fn": create_king_sequence},
            {"time": ts["swingers"], "fn": create_swingers_swing},
            {"time": ts["jungle"], "fn": create_conspiratorial_lean},
            {"time": ts["VIP"], "fn": create_vip_snap},
            {"time": ts["top"], "fn": create_point_up_emphatic},
            {"time": ts["stop"], "fn": create_point_up_emphatic},
            {"time": ts["botherin"], "fn": create_frustrated_shake},
            {"time": ts["wanna_1"], "fn": lambda: self._get_like_you()},
            {"time": ts["man"], "fn": create_point_up_emphatic},
            {"time": ts["stroll"], "fn": create_walk_strut},
            {"time": ts["tired"], "fn": create_frustrated_shake},
            {"time": ts["around"], "fn": create_swingers_swing},
            {"time": ts["oobee"], "fn": lambda: self._get_bam()},
            {"time": ts["like_1"], "fn": lambda: self._get_like_you()},
            {"time": ts["you_1"], "fn": create_point_up_emphatic},
            {"time": ts["walk"], "fn": create_walk_strut},
            {"time": ts["like_2"], "fn": lambda: self._get_like_you()},
            {"time": ts["you_2"], "fn": create_point_up_emphatic},
            {"time": ts["talk"], "fn": create_talk_sassy},
            {"time": ts["like_3"], "fn": lambda: self._get_like_you()},
            {"time": ts["you_3"], "fn": create_point_up_emphatic},
            {"time": ts["true"], "fn": create_point_up_emphatic},
            {"time": ts["ape"], "fn": create_frustrated_shake},
            {"time": ts["human"], "fn": create_aspiring_reach},
            {"time": ts["deal"], "fn": create_conspiratorial_lean},
            {"time": ts["secret_1"], "fn": create_conspiratorial_lean},
            {"time": ts["fire_1"], "fn": create_fire_sequence},
            {"time": ts["fire_2"], "fn": create_fire_sequence},
            {"time": ts["kid"], "fn": create_frustrated_shake},
            {"time": ts["desire"], "fn": create_dream_float},
            {"time": ts["fire_3"], "fn": create_fire_sequence},
            {"time": ts["dream"], "fn": create_dream_float},
            {"time": ts["secret_2"], "fn": create_conspiratorial_lean},
            {"time": ts["power"], "fn": create_power_pose},
            {"time": ts["flower"], "fn": create_fire_sequence},
            {"time": ts["you_4"], "fn": create_point_up_emphatic},
            {"time": ts["bam_1"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_2"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_3"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_4"], "fn": lambda: self._get_bam()},
            {"time": ts["like_4"], "fn": lambda: self._get_like_you()},
            {"time": ts["like_5"], "fn": lambda: self._get_like_you()},
            {"time": ts["learn"], "fn": create_aspiring_reach},
            {"time": ts["you_5"], "fn": create_point_up_emphatic},
            {"time": ts["one_more_time"], "fn": create_power_pose},
            {"time": ts["like_6"], "fn": lambda: self._get_like_you()},
            {"time": ts["bam_5"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_6"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_7"], "fn": lambda: self._get_bam()},
            {"time": ts["bam_8"], "fn": lambda: self._get_bam()},
        ]

    def _get_like_you(self) -> MovementSequence:
        self.like_you_counter += 1
        return create_like_you_left() if self.like_you_counter % 2 == 0 else create_like_you_right()

    def _get_bam(self) -> MovementSequence:
        self.bam_counter += 1
        p = self.bam_counter % 4
        if p == 0:
            return create_bam_bounce_left()
        elif p == 1:
            return create_bam_bounce_right()
        elif p == 2:
            return create_bam_bounce_center()
        return create_bam_bounce_right()

    def reset(self):
        self.current_trigger_idx = 0
        self.active_sequence = None
        self.sequence_start_time = 0
        self.like_you_counter = 0
        self.bam_counter = 0
        self.triggers_fired = 0

    def update(self, t: float) -> Tuple[Optional[Pose], float]:
        """Update at time t. Returns (pose, blend_weight) or (None, 0)."""
        while (self.current_trigger_idx < len(self.triggers) and
               self.triggers[self.current_trigger_idx]['time'] <= t):
            trigger = self.triggers[self.current_trigger_idx]
            self.active_sequence = trigger['fn']()
            self.sequence_start_time = trigger['time']
            self.triggers_fired += 1
            self.current_trigger_idx += 1

        if self.active_sequence:
            elapsed = t - self.sequence_start_time
            pose, weight = self.active_sequence.get_pose_at(elapsed)
            if elapsed > self.active_sequence.get_total_duration() + 0.3:
                self.active_sequence = None
                return None, 0
            return pose.clamped(), weight

        return None, 0


def blend_with_beat_sync(beat_pose: Pose, choreo_pose: Optional[Pose], weight: float) -> Pose:
    """Blend beat-sync with choreography."""
    if choreo_pose is None or weight <= 0:
        return beat_pose
    return beat_pose.lerp(choreo_pose, weight).clamped()
