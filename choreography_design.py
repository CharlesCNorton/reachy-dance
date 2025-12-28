"""
Choreography Design for "I Wanna Be Like You" - The Jungle Book
================================================================

This module defines lyric-triggered movements for Reachy Mini,
layered on top of the beat-synced dancing.

Song: I Wanna Be Like You (The Monkey Song)
Duration: 184.76 seconds
Total Triggers: 50 word-based choreography moments
"""

import json
from dataclasses import dataclass
from typing import List, Tuple

# Reachy Mini movement ranges (degrees for head, 0-1 for antennas)
# Head: pitch (-30 to +20), yaw (-45 to +45), roll (-20 to +20)
# Antennas: 0.0 (down) to 1.0 (up)
# Body yaw: -15 to +15 degrees


@dataclass
class Movement:
    """Defines a Reachy movement pose."""
    head_pitch: float = 0.0   # negative = look down, positive = look up
    head_yaw: float = 0.0     # negative = look left, positive = look right
    head_roll: float = 0.0    # negative = tilt left, positive = tilt right
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    body_yaw: float = 0.0     # degrees
    duration: float = 0.3     # how long to hold/transition


# Define choreography movements
MOVEMENTS = {
    # "king of the swingers" - proud, regal pose
    'HEAD_RAISE_PROUD': Movement(
        head_pitch=15,      # look up proudly
        head_yaw=0,
        head_roll=0,
        antenna_left=0.9,   # antennas up like a crown
        antenna_right=0.9,
        body_yaw=0,
        duration=0.4
    ),

    # "swingers" - swaying side to side
    'BODY_SWAY': Movement(
        head_pitch=5,
        head_yaw=20,        # look to the side
        head_roll=10,       # slight tilt
        antenna_left=0.6,
        antenna_right=0.4,
        body_yaw=12,        # body sway
        duration=0.5
    ),

    # "jungle VIP" - looking around like surveying territory
    'LOOK_AROUND': Movement(
        head_pitch=0,
        head_yaw=35,        # look far to side
        head_roll=5,
        antenna_left=0.7,
        antenna_right=0.7,
        body_yaw=8,
        duration=0.4
    ),

    # "VIP" - swagger/confident pose
    'SWAGGER_POSE': Movement(
        head_pitch=10,      # chin up
        head_yaw=15,        # slight turn
        head_roll=-5,
        antenna_left=1.0,   # antennas full up
        antenna_right=1.0,
        body_yaw=10,
        duration=0.3
    ),

    # "top" - head up high
    'HEAD_UP': Movement(
        head_pitch=18,      # look up
        head_yaw=0,
        head_roll=0,
        antenna_left=0.8,
        antenna_right=0.8,
        body_yaw=0,
        duration=0.25
    ),

    # "I want" - pointing forward gesture (head forward)
    'POINT_FORWARD': Movement(
        head_pitch=-5,      # slight look down/forward
        head_yaw=0,
        head_roll=0,
        antenna_left=0.5,
        antenna_right=0.5,
        body_yaw=0,
        duration=0.2
    ),

    # "like you" - mimicking gesture
    'MIMIC_GESTURE': Movement(
        head_pitch=0,
        head_yaw=10,
        head_roll=8,        # playful tilt
        antenna_left=0.7,
        antenna_right=0.3,  # asymmetric = playful
        body_yaw=5,
        duration=0.25
    ),

    # "you" - pointing up
    'POINT_UP': Movement(
        head_pitch=12,
        head_yaw=0,
        head_roll=0,
        antenna_left=1.0,   # point up with antennas
        antenna_right=1.0,
        body_yaw=0,
        duration=0.2
    ),

    # "walk like you" - bobbing motion
    'BOB_HEAD': Movement(
        head_pitch=-8,      # nod down
        head_yaw=0,
        head_roll=0,
        antenna_left=0.4,
        antenna_right=0.4,
        body_yaw=0,
        duration=0.15
    ),

    # "talk like you" - tilting like talking
    'HEAD_TILT_TALK': Movement(
        head_pitch=5,
        head_yaw=20,
        head_roll=12,       # expressive tilt
        antenna_left=0.6,
        antenna_right=0.8,
        body_yaw=8,
        duration=0.2
    ),

    # "man's red fire" - dramatic antenna flare
    'ANTENNA_FLARE': Movement(
        head_pitch=10,
        head_yaw=0,
        head_roll=0,
        antenna_left=1.0,   # full antenna extension
        antenna_right=1.0,
        body_yaw=0,
        duration=0.4
    ),

    # "dream" - dreamy swaying
    'DREAMY_SWAY': Movement(
        head_pitch=8,
        head_yaw=25,
        head_roll=15,
        antenna_left=0.5,
        antenna_right=0.7,
        body_yaw=10,
        duration=0.5
    ),

    # "power" - power pose
    'POWER_POSE': Movement(
        head_pitch=15,
        head_yaw=0,
        head_roll=0,
        antenna_left=1.0,
        antenna_right=1.0,
        body_yaw=0,
        duration=0.35
    ),

    # "bomb bomb bomb" - rhythmic bouncing
    'RHYTHMIC_BOUNCE': Movement(
        head_pitch=-10,     # quick nod
        head_yaw=0,
        head_roll=5,
        antenna_left=0.3,
        antenna_right=0.3,
        body_yaw=0,
        duration=0.12       # fast for rhythm
    ),
}


def load_choreography() -> List[dict]:
    """Load the choreography triggers from JSON."""
    with open('choreography.json', 'r') as f:
        return json.load(f)


def get_movement_at_time(t: float, choreography: List[dict]) -> Tuple[Movement, bool]:
    """
    Check if there's a choreography trigger at time t.
    Returns (movement, is_triggered).

    The choreography overlays on top of beat-synced dancing.
    """
    for trigger in choreography:
        # Check if we're within the trigger window
        if trigger['time'] <= t <= trigger['time'] + 0.5:  # 500ms window
            movement_name = trigger['movement']
            if movement_name in MOVEMENTS:
                return MOVEMENTS[movement_name], True

    return None, False


# Song structure analysis
SONG_STRUCTURE = {
    'intro': (0, 7.36),
    'verse1': (7.36, 29.48),
    'chorus1': (29.48, 59.46),
    'instrumental1': (59.46, 69.44),
    'verse2_dialogue': (69.44, 100.16),
    'instrumental2': (100.16, 116.06),
    'verse3': (116.06, 127.18),
    'instrumental3': (127.18, 157.18),
    'chorus2': (157.18, 181.84),
    'outro': (181.84, 184.76),
}


if __name__ == '__main__':
    # Print choreography summary
    print("=" * 60)
    print("CHOREOGRAPHY DESIGN: I Wanna Be Like You")
    print("=" * 60)
    print()

    print("MOVEMENT DEFINITIONS:")
    print("-" * 40)
    for name, move in MOVEMENTS.items():
        print(f"\n{name}:")
        print(f"  Head: pitch={move.head_pitch}, yaw={move.head_yaw}, roll={move.head_roll}")
        print(f"  Antennas: L={move.antenna_left}, R={move.antenna_right}")
        print(f"  Body yaw: {move.body_yaw}")
        print(f"  Duration: {move.duration}s")

    print()
    print("SONG STRUCTURE:")
    print("-" * 40)
    for section, (start, end) in SONG_STRUCTURE.items():
        print(f"  {section:20s}: {start:6.2f}s - {end:6.2f}s ({end-start:.1f}s)")

    print()
    print("CHOREOGRAPHY TRIGGERS:")
    print("-" * 40)
    choreo = load_choreography()
    for trigger in choreo:
        print(f"  [{trigger['time']:6.2f}s] {trigger['word']:12s} -> {trigger['movement']}")

    print()
    print(f"Total triggers: {len(choreo)}")
