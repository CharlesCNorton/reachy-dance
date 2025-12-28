"""
Rococo Choreographed Dance for Reachy Mini
============================================

Enhanced lyric-triggered choreography with:
- Multi-phase movement sequences
- Anticipation before big moments
- Follow-through after poses
- Asymmetric antenna personality
- Alternating variations for repeated triggers

Usage:
    python choreographed_dance_rococo.py [--dry-run] [audio_file]
"""

import sys
import os
import time
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import numpy as np

# ============================================================================
# CONFIGURATION & SAFETY LIMITS
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


# Neutral rest pose
REST = Pose(0, 0, 0, 0, 0, 0)


@dataclass
class SequencePhase:
    """A phase within a movement sequence."""
    pose: Pose
    duration: float  # How long this phase lasts
    ease: str = "smooth"  # "linear", "smooth", "snap", "ease_in", "ease_out"


@dataclass
class MovementSequence:
    """A multi-phase movement with anticipation and follow-through."""
    phases: List[SequencePhase]
    blend_weight: float = 0.9

    def get_total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def get_pose_at(self, elapsed: float) -> Tuple[Pose, float]:
        """Get interpolated pose at elapsed time. Returns (pose, remaining_weight)."""
        if not self.phases:
            return REST, 0

        t = 0
        for i, phase in enumerate(self.phases):
            if elapsed < t + phase.duration:
                # We're in this phase
                phase_progress = (elapsed - t) / phase.duration if phase.duration > 0 else 1

                # Apply easing
                if phase.ease == "smooth":
                    # Smooth step
                    phase_progress = phase_progress * phase_progress * (3 - 2 * phase_progress)
                elif phase.ease == "snap":
                    # Quick snap
                    phase_progress = 1 if phase_progress > 0.3 else phase_progress / 0.3
                elif phase.ease == "ease_in":
                    phase_progress = phase_progress * phase_progress
                elif phase.ease == "ease_out":
                    phase_progress = 1 - (1 - phase_progress) ** 2

                # Interpolate from previous phase (or rest) to this phase
                prev_pose = self.phases[i-1].pose if i > 0 else REST
                return prev_pose.lerp(phase.pose, phase_progress), self.blend_weight
            t += phase.duration

        # Past the end - return final pose with decaying weight
        final_pose = self.phases[-1].pose
        overshoot = elapsed - t
        decay = max(0, 1 - overshoot / 0.3)  # Fade out over 0.3s
        return final_pose.lerp(REST, 1 - decay), self.blend_weight * decay


# ============================================================================
# ROCOCO MOVEMENT SEQUENCES
# ============================================================================

def create_king_sequence() -> MovementSequence:
    """King of the swingers - regal head raise + slow turn + crown antennas."""
    return MovementSequence(
        phases=[
            # Anticipation - slight dip
            SequencePhase(Pose(-5, 0, 0, 0.2, 0.2, 0), 0.1, "smooth"),
            # Rise up proudly
            SequencePhase(Pose(15, 0, 0, 0.5, 0.5, 0), 0.2, "ease_out"),
            # Slow regal turn with crown rising
            SequencePhase(Pose(18, 20, 3, 0.9, 0.95, 5), 0.3, "smooth"),
            # Hold the crown
            SequencePhase(Pose(15, 15, 0, 1.0, 1.0, 3), 0.3, "smooth"),
            # Settle back
            SequencePhase(Pose(8, 5, 0, 0.7, 0.7, 0), 0.2, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_fire_sequence() -> MovementSequence:
    """Man's red fire - flicker → FLARE → hold."""
    return MovementSequence(
        phases=[
            # Anticipation - pull back slightly
            SequencePhase(Pose(-3, 0, 0, 0.2, 0.3, 0), 0.08, "snap"),
            # Flicker 1
            SequencePhase(Pose(5, 0, 0, 0.6, 0.4, 0), 0.06, "snap"),
            # Flicker 2
            SequencePhase(Pose(3, 0, 0, 0.4, 0.7, 0), 0.06, "snap"),
            # Flicker 3
            SequencePhase(Pose(6, 0, 0, 0.8, 0.5, 0), 0.06, "snap"),
            # BIG FLARE!
            SequencePhase(Pose(15, 0, 0, 1.0, 1.0, 0), 0.12, "snap"),
            # Hold the fire high
            SequencePhase(Pose(12, 0, 2, 1.0, 1.0, 0), 0.25, "smooth"),
            # Gentle settle
            SequencePhase(Pose(8, 0, 0, 0.8, 0.8, 0), 0.15, "ease_in"),
        ],
        blend_weight=0.98
    )


def create_like_you_left() -> MovementSequence:
    """Like you - mimicking gesture, left variation."""
    return MovementSequence(
        phases=[
            # Quick anticipation
            SequencePhase(Pose(0, -5, -3, 0.4, 0.5, -2), 0.08, "smooth"),
            # Playful tilt left
            SequencePhase(Pose(5, -15, 10, 0.8, 0.4, -5), 0.15, "ease_out"),
            # Point moment
            SequencePhase(Pose(10, -10, 8, 0.9, 0.5, -3), 0.12, "smooth"),
            # Settle with personality
            SequencePhase(Pose(3, -5, 3, 0.6, 0.5, -1), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_like_you_right() -> MovementSequence:
    """Like you - mimicking gesture, right variation."""
    return MovementSequence(
        phases=[
            # Quick anticipation
            SequencePhase(Pose(0, 5, 3, 0.5, 0.4, 2), 0.08, "smooth"),
            # Playful tilt right
            SequencePhase(Pose(5, 15, -10, 0.4, 0.8, 5), 0.15, "ease_out"),
            # Point moment
            SequencePhase(Pose(10, 10, -8, 0.5, 0.9, 3), 0.12, "smooth"),
            # Settle with personality
            SequencePhase(Pose(3, 5, -3, 0.5, 0.6, 1), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_walk_strut() -> MovementSequence:
    """Walk like you - exaggerated strut bobbing."""
    return MovementSequence(
        phases=[
            # Bob down with attitude
            SequencePhase(Pose(-10, 8, 5, 0.3, 0.5, 4), 0.1, "snap"),
            # Strut up
            SequencePhase(Pose(5, -5, -3, 0.6, 0.4, -2), 0.1, "ease_out"),
            # Sassy hold
            SequencePhase(Pose(3, 10, 6, 0.5, 0.6, 3), 0.1, "smooth"),
        ],
        blend_weight=0.75
    )


def create_talk_sassy() -> MovementSequence:
    """Talk like you - sassy head tilts."""
    return MovementSequence(
        phases=[
            # Tilt into it
            SequencePhase(Pose(3, 20, 12, 0.5, 0.7, 6), 0.1, "ease_out"),
            # Emphasize
            SequencePhase(Pose(5, 25, 15, 0.6, 0.8, 8), 0.12, "smooth"),
            # Punctuate
            SequencePhase(Pose(0, 15, 8, 0.5, 0.6, 4), 0.1, "ease_in"),
        ],
        blend_weight=0.8
    )


def create_swingers_swing() -> MovementSequence:
    """Swingers - full body swing with antenna countermotion."""
    return MovementSequence(
        phases=[
            # Wind up left
            SequencePhase(Pose(3, -25, -8, 0.8, 0.3, -10), 0.15, "smooth"),
            # Swing through center
            SequencePhase(Pose(8, 0, 0, 0.5, 0.5, 0), 0.1, "ease_out"),
            # Swing right with antenna counter
            SequencePhase(Pose(5, 30, 10, 0.3, 0.9, 12), 0.2, "smooth"),
            # Swing back center
            SequencePhase(Pose(5, 10, 3, 0.5, 0.6, 4), 0.15, "ease_in"),
        ],
        blend_weight=0.9
    )


def create_bam_bounce_left() -> MovementSequence:
    """Bam - bounce left."""
    return MovementSequence(
        phases=[
            # Quick hit left
            SequencePhase(Pose(-8, -10, -5, 0.2, 0.5, -5), 0.06, "snap"),
            # Rebound
            SequencePhase(Pose(0, -3, 0, 0.4, 0.4, -1), 0.06, "ease_out"),
        ],
        blend_weight=0.7
    )


def create_bam_bounce_right() -> MovementSequence:
    """Bam - bounce right."""
    return MovementSequence(
        phases=[
            # Quick hit right
            SequencePhase(Pose(-8, 10, 5, 0.5, 0.2, 5), 0.06, "snap"),
            # Rebound
            SequencePhase(Pose(0, 3, 0, 0.4, 0.4, 1), 0.06, "ease_out"),
        ],
        blend_weight=0.7
    )


def create_bam_bounce_center() -> MovementSequence:
    """Bam - bounce center (building energy)."""
    return MovementSequence(
        phases=[
            # Strong center hit
            SequencePhase(Pose(-12, 0, 0, 0.6, 0.6, 0), 0.05, "snap"),
            # Pop up
            SequencePhase(Pose(5, 0, 0, 0.8, 0.8, 0), 0.05, "snap"),
        ],
        blend_weight=0.75
    )


def create_dream_float() -> MovementSequence:
    """Dream - slow dreamy roll with asymmetric antenna float."""
    return MovementSequence(
        phases=[
            # Float up dreamily
            SequencePhase(Pose(8, 10, 5, 0.4, 0.7, 3), 0.2, "ease_out"),
            # Gentle roll
            SequencePhase(Pose(10, 20, 12, 0.5, 0.9, 6), 0.25, "smooth"),
            # Dreamy peak
            SequencePhase(Pose(12, 25, 15, 0.6, 1.0, 8), 0.2, "smooth"),
            # Slow float down
            SequencePhase(Pose(8, 15, 8, 0.5, 0.8, 5), 0.2, "ease_in"),
            # Settle
            SequencePhase(Pose(5, 8, 3, 0.4, 0.6, 2), 0.15, "smooth"),
        ],
        blend_weight=0.85
    )


def create_vip_snap() -> MovementSequence:
    """VIP - dramatic pause then snap into pose."""
    return MovementSequence(
        phases=[
            # Dramatic pause - stillness with slight tension
            SequencePhase(Pose(-2, 0, 0, 0.3, 0.3, 0), 0.15, "linear"),
            # THE SNAP!
            SequencePhase(Pose(12, 18, -5, 1.0, 1.0, 8), 0.08, "snap"),
            # Hold with swagger
            SequencePhase(Pose(10, 15, -3, 0.95, 0.95, 6), 0.2, "smooth"),
            # Cool settle
            SequencePhase(Pose(5, 8, 0, 0.7, 0.7, 3), 0.12, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_point_up_emphatic() -> MovementSequence:
    """Point up on 'you' - emphatic with asymmetry."""
    return MovementSequence(
        phases=[
            # Wind up
            SequencePhase(Pose(-3, 0, 0, 0.3, 0.4, 0), 0.06, "smooth"),
            # Point UP!
            SequencePhase(Pose(15, 0, 0, 1.0, 0.95, 0), 0.1, "snap"),
            # Hold
            SequencePhase(Pose(12, 0, 2, 0.95, 1.0, 0), 0.12, "smooth"),
            # Settle
            SequencePhase(Pose(5, 0, 0, 0.6, 0.6, 0), 0.1, "ease_in"),
        ],
        blend_weight=0.85
    )


def create_power_pose() -> MovementSequence:
    """Power - strong pose with buildup."""
    return MovementSequence(
        phases=[
            # Gather strength
            SequencePhase(Pose(-5, 0, 0, 0.3, 0.3, 0), 0.1, "smooth"),
            # POWER!
            SequencePhase(Pose(18, 0, 0, 1.0, 1.0, 0), 0.12, "snap"),
            # Hold strong
            SequencePhase(Pose(15, 0, 2, 1.0, 1.0, 0), 0.2, "smooth"),
            # Powerful settle
            SequencePhase(Pose(10, 0, 0, 0.8, 0.8, 0), 0.1, "ease_in"),
        ],
        blend_weight=0.95
    )


def create_frustrated_shake() -> MovementSequence:
    """Frustrated - shake with personality."""
    return MovementSequence(
        phases=[
            # Shake left
            SequencePhase(Pose(0, -18, -6, 0.4, 0.6, -4), 0.08, "snap"),
            # Shake right
            SequencePhase(Pose(0, 18, 6, 0.6, 0.4, 4), 0.08, "snap"),
            # Shake left smaller
            SequencePhase(Pose(0, -10, -3, 0.5, 0.5, -2), 0.06, "snap"),
            # Settle annoyed
            SequencePhase(Pose(-3, 0, 0, 0.4, 0.4, 0), 0.08, "ease_in"),
        ],
        blend_weight=0.8
    )


def create_conspiratorial_lean() -> MovementSequence:
    """Conspiratorial - sneaky lean in."""
    return MovementSequence(
        phases=[
            # Look around first
            SequencePhase(Pose(0, -20, 0, 0.5, 0.5, -5), 0.12, "smooth"),
            # Check other side
            SequencePhase(Pose(0, 15, 0, 0.5, 0.5, 3), 0.1, "smooth"),
            # Lean in close
            SequencePhase(Pose(-8, 25, 8, 0.4, 0.6, 8), 0.15, "ease_out"),
            # Whisper pose
            SequencePhase(Pose(-5, 20, 5, 0.5, 0.6, 6), 0.15, "smooth"),
        ],
        blend_weight=0.85
    )


def create_aspiring_reach() -> MovementSequence:
    """Aspiring/human - reaching upward hopefully."""
    return MovementSequence(
        phases=[
            # Start humble
            SequencePhase(Pose(-5, 0, 0, 0.2, 0.2, 0), 0.1, "smooth"),
            # Reach up with hope
            SequencePhase(Pose(15, 5, 3, 0.8, 0.9, 2), 0.2, "ease_out"),
            # Full aspiration
            SequencePhase(Pose(18, 0, 0, 1.0, 1.0, 0), 0.15, "smooth"),
            # Gentle settle with hope
            SequencePhase(Pose(10, 0, 0, 0.7, 0.8, 0), 0.15, "ease_in"),
        ],
        blend_weight=0.9
    )


# ============================================================================
# ROCOCO CHOREOGRAPHY ENGINE
# ============================================================================

class RococoChoreographyEngine:
    """Enhanced choreography engine with sequences and state tracking."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.active_sequence: Optional[MovementSequence] = None
        self.sequence_start_time: float = 0
        self.like_you_counter: int = 0  # Alternates left/right
        self.bam_counter: int = 0  # Cycles through bounce patterns
        self.triggers_fired: List[str] = []

        # Build trigger list with timestamps
        self.triggers = self._build_triggers()
        self.current_trigger_idx = 0

    def _build_triggers(self) -> List[dict]:
        """Build the trigger list with sequence factories."""
        return [
            # Verse 1 - Swagger
            {"time": 8.50, "word": "king", "sequence_fn": create_king_sequence},
            {"time": 9.20, "word": "swingers", "sequence_fn": create_swingers_swing},
            {"time": 10.58, "word": "jungle", "sequence_fn": create_conspiratorial_lean},
            {"time": 10.94, "word": "VIP", "sequence_fn": create_vip_snap},
            {"time": 13.56, "word": "top", "sequence_fn": create_point_up_emphatic},
            {"time": 14.70, "word": "stop", "sequence_fn": create_point_up_emphatic},
            {"time": 16.44, "word": "botherin'", "sequence_fn": create_frustrated_shake},
            {"time": 17.90, "word": "wanna", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 18.54, "word": "man", "sequence_fn": create_point_up_emphatic},
            {"time": 20.20, "word": "stroll", "sequence_fn": create_walk_strut},
            {"time": 25.18, "word": "tired", "sequence_fn": create_frustrated_shake},
            {"time": 26.26, "word": "around", "sequence_fn": create_swingers_swing},

            # Chorus 1
            {"time": 27.12, "word": "oobee", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 29.80, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 30.14, "word": "you", "sequence_fn": create_point_up_emphatic},
            {"time": 32.64, "word": "walk", "sequence_fn": create_walk_strut},
            {"time": 32.94, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 33.26, "word": "you", "sequence_fn": create_point_up_emphatic},
            {"time": 33.74, "word": "talk", "sequence_fn": create_talk_sassy},
            {"time": 34.02, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 34.42, "word": "you", "sequence_fn": create_point_up_emphatic},
            {"time": 36.00, "word": "true", "sequence_fn": create_point_up_emphatic},
            {"time": 38.50, "word": "ape", "sequence_fn": create_frustrated_shake},
            {"time": 40.50, "word": "human", "sequence_fn": create_aspiring_reach},

            # Verse 2 - Scheming
            {"time": 71.76, "word": "deal", "sequence_fn": create_conspiratorial_lean},
            {"time": 75.40, "word": "secret", "sequence_fn": create_conspiratorial_lean},
            {"time": 77.50, "word": "fire", "sequence_fn": create_fire_sequence},
            {"time": 79.92, "word": "fire", "sequence_fn": create_fire_sequence},
            {"time": 82.28, "word": "kid", "sequence_fn": create_frustrated_shake},
            {"time": 86.54, "word": "desire", "sequence_fn": create_dream_float},
            {"time": 88.12, "word": "fire", "sequence_fn": create_fire_sequence},
            {"time": 89.30, "word": "dream", "sequence_fn": create_dream_float},
            {"time": 91.56, "word": "secret", "sequence_fn": create_conspiratorial_lean},
            {"time": 96.46, "word": "power", "sequence_fn": create_power_pose},
            {"time": 97.82, "word": "flower", "sequence_fn": create_fire_sequence},
            {"time": 99.70, "word": "you", "sequence_fn": create_point_up_emphatic},

            # Scat section
            {"time": 125.52, "word": "bam1", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 126.14, "word": "bam2", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 126.56, "word": "bam3", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 126.90, "word": "bam4", "sequence_fn": lambda: self._get_bam_sequence()},

            # Final chorus
            {"time": 159.22, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 163.24, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},
            {"time": 167.36, "word": "learn", "sequence_fn": create_aspiring_reach},
            {"time": 169.60, "word": "you", "sequence_fn": create_point_up_emphatic},
            {"time": 170.42, "word": "One more time", "sequence_fn": create_power_pose},
            {"time": 173.02, "word": "like", "sequence_fn": lambda: self._get_like_you_sequence()},

            # Outro scat
            {"time": 177.10, "word": "bam5", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 177.56, "word": "bam6", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 177.88, "word": "bam7", "sequence_fn": lambda: self._get_bam_sequence()},
            {"time": 178.00, "word": "bam8", "sequence_fn": lambda: self._get_bam_sequence()},
        ]

    def _get_like_you_sequence(self) -> MovementSequence:
        """Get alternating like you sequence."""
        self.like_you_counter += 1
        if self.like_you_counter % 2 == 0:
            return create_like_you_left()
        else:
            return create_like_you_right()

    def _get_bam_sequence(self) -> MovementSequence:
        """Get cycling bam bounce pattern."""
        self.bam_counter += 1
        pattern = self.bam_counter % 4
        if pattern == 0:
            return create_bam_bounce_left()
        elif pattern == 1:
            return create_bam_bounce_right()
        elif pattern == 2:
            return create_bam_bounce_center()
        else:
            return create_bam_bounce_right()

    def reset(self):
        """Reset for new playback."""
        self.current_trigger_idx = 0
        self.active_sequence = None
        self.sequence_start_time = 0
        self.like_you_counter = 0
        self.bam_counter = 0
        self.triggers_fired = []

    def update(self, t: float) -> Tuple[Optional[Pose], float]:
        """
        Update at time t.
        Returns (pose, blend_weight) or (None, 0).
        """
        # Check for new triggers
        while (self.current_trigger_idx < len(self.triggers) and
               self.triggers[self.current_trigger_idx]['time'] <= t):

            trigger = self.triggers[self.current_trigger_idx]
            sequence = trigger['sequence_fn']()

            self.active_sequence = sequence
            self.sequence_start_time = trigger['time']
            self.triggers_fired.append(trigger['word'])

            if self.dry_run:
                print(f"  [{t:6.2f}s] TRIGGER: '{trigger['word']}' "
                      f"({len(sequence.phases)} phases, {sequence.get_total_duration():.2f}s)")

            self.current_trigger_idx += 1

        # Get pose from active sequence
        if self.active_sequence:
            elapsed = t - self.sequence_start_time
            pose, weight = self.active_sequence.get_pose_at(elapsed)

            # Check if sequence is done
            if elapsed > self.active_sequence.get_total_duration() + 0.3:
                self.active_sequence = None
                return None, 0

            return pose.clamped(), weight

        return None, 0


# ============================================================================
# MAIN DANCE LOOP
# ============================================================================

def blend_with_beat_sync(beat_sync_pose: Pose, choreo_pose: Optional[Pose],
                          choreo_weight: float) -> Pose:
    """Blend beat-sync with choreography."""
    if choreo_pose is None or choreo_weight <= 0:
        return beat_sync_pose

    return beat_sync_pose.lerp(choreo_pose, choreo_weight).clamped()


def dry_run_test(duration: float = 184.76):
    """Test choreography without robot."""
    print("=" * 60)
    print("ROCOCO DRY RUN TEST")
    print("=" * 60)
    print()

    engine = RococoChoreographyEngine(dry_run=True)

    print(f"Total triggers: {len(engine.triggers)}")
    print()
    print("SIMULATING PLAYBACK:")
    print("-" * 40)

    t = 0
    dt = 0.016  # 60 FPS
    while t < duration:
        pose, weight = engine.update(t)
        t += dt

    print()
    print(f"Triggers fired: {len(engine.triggers_fired)}")
    print()
    print("=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)

    return True


def run_rococo_dance(audio_file: str, dry_run: bool = False):
    """Run the rococo choreographed dance."""
    if dry_run:
        return dry_run_test()

    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
        import librosa
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False

    print("=" * 60)
    print("ROCOCO CHOREOGRAPHED DANCE")
    print("I Wanna Be Like You - Enhanced Edition")
    print("=" * 60)
    print()

    # Load audio
    print("[AUDIO] Loading audio file...")
    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        return False

    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    beat_freq = tempo / 60.0

    print(f"[AUDIO] Duration: {duration:.2f}s")
    print(f"[AUDIO] Tempo: {tempo:.1f} BPM")
    print()

    engine = RococoChoreographyEngine(dry_run=False)

    print("[ROBOT] Connecting to Reachy Mini...")
    try:
        with ReachyMini() as mini:
            print("[ROBOT] Connected!")

            # Wake up
            print("[ROBOT] Waking up...")
            try:
                mini.wake_up()
                time.sleep(1.0)
            except Exception as e:
                print(f"[WARN] Wake up: {e}")

            # Start position
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
                print(f"[WARN] Start position: {e}")

            print()
            print("=" * 60)
            print("  STARTING ROCOCO DANCE!")
            print(f"  Duration: {duration:.1f}s")
            print("  Press Ctrl+C to stop")
            print("=" * 60)
            print()

            # Start audio
            try:
                mini.media.play_sound(audio_file)
            except Exception as e:
                print(f"[WARN] Audio: {e}")

            # Dance loop
            start_time = time.time()
            frame_count = 0
            last_status = start_time

            try:
                while True:
                    t = time.time() - start_time
                    if t >= duration:
                        break

                    # Beat sync base
                    bp = 8 * 0.5 * np.sin(2 * np.pi * beat_freq * t)
                    by = 12 * 0.4 * np.sin(2 * np.pi * beat_freq / 2 * t)
                    br = 6 * np.sin(2 * np.pi * beat_freq * t + np.pi / 4)
                    ba = 0.4 + 0.2 * np.sin(2 * np.pi * beat_freq * 2 * t)
                    bb = 10 * np.sin(2 * np.pi * beat_freq / 4 * t)

                    beat_pose = Pose(bp, by, br, ba, ba, bb)

                    # Get choreography
                    choreo_pose, weight = engine.update(t)

                    # Blend
                    final_pose = blend_with_beat_sync(beat_pose, choreo_pose, weight)

                    # Apply
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
                        pass

                    frame_count += 1

                    if time.time() - last_status >= 10.0:
                        fps = frame_count / t if t > 0 else 0
                        print(f"[DANCE] {t:.1f}s, {frame_count} frames, {fps:.1f} FPS")
                        last_status = time.time()

                    time.sleep(0.016)

            except KeyboardInterrupt:
                print("\n[INFO] Interrupted")

            # Rest
            print()
            print("[ROBOT] Returning to rest...")
            try:
                mini.goto_target(
                    head=create_head_pose(pitch=0, yaw=0, roll=0, degrees=True),
                    antennas=[0, 0],
                    body_yaw=0,
                    duration=1.0
                )
                time.sleep(1.0)
            except:
                pass

            # Sleep
            print("[ROBOT] Going to sleep...")
            try:
                mini.goto_sleep()
            except:
                pass

            print()
            print("=" * 60)
            total = time.time() - start_time
            print(f"  ROCOCO DANCE COMPLETE!")
            print(f"  {frame_count} frames in {total:.1f}s ({frame_count/total:.1f} FPS)")
            print(f"  Triggers fired: {len(engine.triggers_fired)}")
            print("=" * 60)

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Rococo Choreographed Dance")
    parser.add_argument('audio_file', nargs='?', default='choreography_audio.wav')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        return 0 if dry_run_test() else 1
    else:
        return 0 if run_rococo_dance(args.audio_file) else 1


if __name__ == "__main__":
    sys.exit(main())
