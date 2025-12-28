"""
Lyrics Correction Mapping for "I Wanna Be Like You"
====================================================

This module corrects Whisper transcription errors and provides
accurate lyric-to-timestamp mapping for choreography.

Analysis based on comparing Whisper output (with confidence scores)
against actual song lyrics.
"""

# Corrections: (start_time, whisper_word, actual_word, is_scat)
CORRECTIONS = [
    # Segment 2: "man cup" -> "mancub"
    (19.04, "man", "man-", False),
    (19.22, "cup,", "cub,", False),

    # Segment 3: "I'm all gonna round" -> "of monkeyin' around"
    (25.62, "I'm", "of", False),
    (25.92, "all", "monkey-", False),
    (26.04, "gonna", "in'", False),
    (26.26, "round,", "around!", False),

    # Segment 3: "oh, I want" -> "Oh, oobee doo, I wanna"
    (27.12, "oh,", "Oh,", False),
    (27.76, "I", "oobee", True),  # Start of scat
    (27.92, "want", "doo", True),

    # Segment 4: This is the chorus
    (29.48, "to", "(I wanna)", False),
    (33.62, "she", "", False),  # Remove - doesn't exist
    (33.74, "talks", "talk", False),
    (34.90, "she's", "", False),  # Remove
    (36.00, "true,", "too", False),
    (36.26, "we", "You'll", False),
    (36.38, "be", "see", False),
    (36.56, "deep", "it's", False),

    # Segment 5: More chorus/scat confusion
    (36.78, "and", "", False),  # Remove
    (37.24, "say", "see", False),
    (38.48, "she'll", "An ape", False),
    (38.92, "be", "like", False),
    (38.98, "like", "me", False),
    (39.62, "and", "Can", False),
    (39.72, "they", "learn", False),
    (39.84, "like", "to", False),
    (40.12, "me,", "be", False),
    (40.98, "she'll", "human", False),
    (41.56, "like", "too", False),
    # 41.80 onwards is scat/repeat

    # Segment 6: Mowgli's line then King Louie
    (69.44, "You're", "Gee,", False),  # Actually this is correct-ish
    (70.00, "doing", "cousin", False),
    (70.24, "real", "Louie", False),  # Whisper got confused
    (73.66, "cuz", "cuz", False),  # Correct
    (74.00, "they've", "Lay", False),

    # Segment 7: "besieged on me" -> "the secret on me"
    (75.40, "besieged", "the secret", False),

    # Segment 8: "man cup" -> "mancub" again, "I'll make" -> "I made"
    (82.86, "man", "man-", False),
    (83.08, "cup,", "cub,", False),
    (83.76, "I'll", "I", False),
    (84.16, "make", "made", False),

    # Segments 12-14: SCAT SECTION - Mark all as scat
    (116.06, "Hey,", "(scat)", True),
    (117.12, "there's", "oobee", True),
    (117.48, "ban", "doo", True),
    (118.88, "I", "I", True),
    (119.36, "glad", "wanna", True),
    (119.80, "doubt", "be", True),
    (120.12, "that", "like", True),
    (120.30, "long.", "you", True),

    (121.46, "And", "(scat)", True),
    (122.18, "be", "scooby", True),
    (122.24, "ready", "dooby", True),
    (122.84, "serve", "doo", True),
    (123.38, "duke", "bee", True),
    (124.16, "pop,", "doo", True),
    (124.56, "book", "bop", True),

    # Segment 14: bomb bomb - actually correct scat
    (125.42, "so", "sha", True),
    (125.52, "bomb,", "bam", True),

    # Segments 15-19: Final chorus - many errors
    (157.18, "It's", "Oh", False),
    (157.78, "who", "oobee", True),
    (158.34, "should", "doo", True),
    (158.62, "do", "I", False),
    (158.84, "someone", "wanna", False),

    (161.70, "Can't", "Can", False),
    (165.56, "Behind", "Take", False),
    (165.90, "that,", "me", False),
    (166.48, "can't", "home,", False),
    # ... continues with corrections

    (170.42, "One", "One", False),  # Correct!
    (176.24, "She", "Sha", True),
    (176.92, "a", "da", True),
]

# Actual lyrics with corrected timestamps
CORRECTED_LYRICS = [
    # Verse 1
    {"time": 7.36, "text": "Now, I'm the king of the swingers", "section": "verse1"},
    {"time": 9.86, "text": "Oh, the jungle VIP", "section": "verse1"},
    {"time": 12.26, "text": "I've reached the top and had to stop", "section": "verse1"},
    {"time": 15.28, "text": "And that's what's botherin' me", "section": "verse1"},
    {"time": 17.02, "text": "I wanna be a man, mancub", "section": "verse1"},
    {"time": 20.14, "text": "And stroll right into town", "section": "verse1"},
    {"time": 22.02, "text": "And be just like the other men", "section": "verse1"},
    {"time": 25.02, "text": "I'm tired of monkeyin' around!", "section": "verse1"},

    # Chorus 1
    {"time": 27.12, "text": "Oh, oobee doo", "section": "chorus", "is_scat": True},
    {"time": 28.50, "text": "I wanna be like you", "section": "chorus"},
    {"time": 30.70, "text": "I wanna walk like you", "section": "chorus"},
    {"time": 33.62, "text": "Talk like you, too", "section": "chorus"},
    {"time": 35.50, "text": "You'll see it's true", "section": "chorus"},
    {"time": 37.50, "text": "An ape like me", "section": "chorus"},
    {"time": 39.50, "text": "Can learn to be human too", "section": "chorus"},

    # Scat/Repeat section
    {"time": 41.00, "text": "(oobee doo repeat)", "section": "scat", "is_scat": True},

    # Instrumental
    {"time": 59.46, "text": "(instrumental)", "section": "instrumental"},

    # Verse 2 - Dialogue
    {"time": 69.44, "text": "Gee, cousin Louie, you're doin' real good!", "section": "verse2", "speaker": "Mowgli"},
    {"time": 71.76, "text": "Now here's your part of the deal, cuz", "section": "verse2"},
    {"time": 74.00, "text": "Lay the secret on me of man's red fire", "section": "verse2"},
    {"time": 78.74, "text": "But I don't know how to make fire", "section": "verse2", "speaker": "Mowgli"},
    {"time": 81.06, "text": "Now don't try to kid me, mancub", "section": "verse2"},
    {"time": 83.76, "text": "I made a deal with you", "section": "verse2"},
    {"time": 85.52, "text": "What I desire is man's red fire", "section": "verse2"},
    {"time": 88.44, "text": "To make my dream come true", "section": "verse2"},
    {"time": 90.54, "text": "Now give me the secret, mancub", "section": "verse2"},
    {"time": 93.08, "text": "C'mon, clue me what to do", "section": "verse2"},
    {"time": 95.90, "text": "Give me the power of man's red flower", "section": "verse2"},
    {"time": 98.50, "text": "So I can be like you!", "section": "verse2"},

    # Instrumental 2
    {"time": 100.16, "text": "(instrumental - Baloo scene)", "section": "instrumental"},

    # Scat section
    {"time": 116.06, "text": "(scat singing)", "section": "scat", "is_scat": True},
    {"time": 125.42, "text": "Sha-bam, sha-bam, bam, bam", "section": "scat", "is_scat": True},

    # Instrumental 3
    {"time": 127.18, "text": "(instrumental - chase)", "section": "instrumental"},

    # Final Chorus
    {"time": 157.18, "text": "Oh, oobee doo", "section": "chorus_final", "is_scat": True},
    {"time": 159.00, "text": "I wanna be like you", "section": "chorus_final"},
    {"time": 161.70, "text": "Can learn to be like someone like me", "section": "chorus_final"},
    {"time": 165.56, "text": "Take me home, daddy", "section": "chorus_final"},
    {"time": 167.36, "text": "Can learn to be like someone like you", "section": "chorus_final"},
    {"time": 170.42, "text": "One more time!", "section": "chorus_final"},
    {"time": 171.90, "text": "Can learn to be like someone like me", "section": "chorus_final"},

    # Outro
    {"time": 176.24, "text": "Sha-da-da, bam bam bam bam", "section": "outro", "is_scat": True},
    {"time": 178.12, "text": "Da da da da da da", "section": "outro", "is_scat": True},
]

# Key choreography trigger words with corrected times
CHOREOGRAPHY_TRIGGERS = [
    # Verse 1 - Swagger and boast
    {"time": 8.50, "word": "king", "movement": "HEAD_RAISE_PROUD", "duration": 0.4},
    {"time": 9.20, "word": "swingers", "movement": "BODY_SWAY", "duration": 0.5},
    {"time": 10.58, "word": "jungle", "movement": "LOOK_AROUND", "duration": 0.4},
    {"time": 10.94, "word": "VIP", "movement": "SWAGGER_POSE", "duration": 0.3},
    {"time": 13.56, "word": "top", "movement": "HEAD_UP", "duration": 0.25},
    {"time": 14.70, "word": "stop", "movement": "HEAD_UP", "duration": 0.25},
    {"time": 16.44, "word": "botherin'", "movement": "FRUSTRATED_SHAKE", "duration": 0.3},
    {"time": 17.90, "word": "wanna", "movement": "POINT_FORWARD", "duration": 0.2},
    {"time": 18.54, "word": "man", "movement": "POINT_FORWARD", "duration": 0.2},
    {"time": 20.20, "word": "stroll", "movement": "SMOOTH_SWAY", "duration": 0.4},
    {"time": 25.18, "word": "tired", "movement": "FRUSTRATED_SHAKE", "duration": 0.3},
    {"time": 26.26, "word": "around", "movement": "BODY_SWAY", "duration": 0.3},

    # Chorus 1 - Pointing and mimicking
    {"time": 27.12, "word": "oobee", "movement": "SCAT_BOUNCE", "duration": 0.2, "is_scat": True},
    {"time": 29.80, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},
    {"time": 30.14, "word": "you", "movement": "POINT_UP", "duration": 0.2},
    {"time": 32.64, "word": "walk", "movement": "BOB_HEAD", "duration": 0.15},
    {"time": 32.94, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},
    {"time": 33.26, "word": "you", "movement": "POINT_UP", "duration": 0.2},
    {"time": 33.74, "word": "talk", "movement": "HEAD_TILT_TALK", "duration": 0.2},
    {"time": 34.02, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},
    {"time": 34.42, "word": "you", "movement": "POINT_UP", "duration": 0.2},
    {"time": 36.00, "word": "true", "movement": "NOD_AFFIRM", "duration": 0.25},
    {"time": 38.50, "word": "ape", "movement": "HUMBLE_TILT", "duration": 0.3},
    {"time": 40.50, "word": "human", "movement": "ASPIRING_REACH", "duration": 0.4},

    # Verse 2 - Scheming for fire
    {"time": 71.76, "word": "deal", "movement": "CONSPIRATORIAL_LEAN", "duration": 0.3},
    {"time": 75.40, "word": "secret", "movement": "CONSPIRATORIAL_LEAN", "duration": 0.3},
    {"time": 77.50, "word": "fire", "movement": "ANTENNA_FLARE", "duration": 0.4},
    {"time": 79.92, "word": "fire", "movement": "ANTENNA_FLARE", "duration": 0.4},
    {"time": 82.28, "word": "kid", "movement": "HEAD_SHAKE_NO", "duration": 0.25},
    {"time": 86.54, "word": "desire", "movement": "YEARNING_REACH", "duration": 0.35},
    {"time": 88.12, "word": "fire", "movement": "ANTENNA_FLARE", "duration": 0.4},
    {"time": 89.30, "word": "dream", "movement": "DREAMY_SWAY", "duration": 0.5},
    {"time": 91.56, "word": "secret", "movement": "CONSPIRATORIAL_LEAN", "duration": 0.3},
    {"time": 96.46, "word": "power", "movement": "POWER_POSE", "duration": 0.35},
    {"time": 97.82, "word": "flower", "movement": "ANTENNA_FLARE", "duration": 0.4},
    {"time": 99.70, "word": "you", "movement": "POINT_UP", "duration": 0.3},

    # Scat section - rhythmic bouncing
    {"time": 125.52, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 126.14, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 126.56, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 126.90, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},

    # Final chorus
    {"time": 159.22, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},
    {"time": 163.24, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},
    {"time": 167.36, "word": "learn", "movement": "ASPIRING_REACH", "duration": 0.3},
    {"time": 169.60, "word": "you", "movement": "POINT_UP", "duration": 0.2},
    {"time": 170.42, "word": "One more time", "movement": "POWER_POSE", "duration": 0.35},
    {"time": 173.02, "word": "like", "movement": "MIMIC_GESTURE", "duration": 0.25},

    # Outro scat
    {"time": 177.10, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 177.56, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 177.88, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
    {"time": 178.00, "word": "bam", "movement": "RHYTHMIC_BOUNCE", "duration": 0.12},
]


def get_section_at_time(t: float) -> str:
    """Get the song section at a given time."""
    sections = [
        (0, 7.36, "intro"),
        (7.36, 27.12, "verse1"),
        (27.12, 41.00, "chorus1"),
        (41.00, 59.46, "scat1"),
        (59.46, 69.44, "instrumental1"),
        (69.44, 100.16, "verse2"),
        (100.16, 116.06, "instrumental2"),
        (116.06, 127.18, "scat2"),
        (127.18, 157.18, "instrumental3"),
        (157.18, 176.24, "chorus_final"),
        (176.24, 184.76, "outro"),
    ]
    for start, end, name in sections:
        if start <= t < end:
            return name
    return "unknown"


def get_energy_level(section: str) -> str:
    """Get the energy level for choreography blending."""
    energy_map = {
        "intro": "low",
        "verse1": "medium_swagger",
        "chorus1": "high_pleading",
        "scat1": "high_rhythmic",
        "instrumental1": "medium",
        "verse2": "medium_scheming",
        "instrumental2": "medium",
        "scat2": "high_rhythmic",
        "instrumental3": "medium",
        "chorus_final": "high_pleading",
        "outro": "high_rhythmic",
    }
    return energy_map.get(section, "medium")


if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTED CHOREOGRAPHY TRIGGERS")
    print("=" * 60)
    print()

    current_section = None
    for trigger in CHOREOGRAPHY_TRIGGERS:
        section = get_section_at_time(trigger["time"])
        if section != current_section:
            print(f"\n--- {section.upper()} ({get_energy_level(section)}) ---")
            current_section = section

        scat_marker = " [SCAT]" if trigger.get("is_scat") else ""
        print(f"  [{trigger['time']:6.2f}s] {trigger['word']:15s} -> {trigger['movement']}{scat_marker}")

    print()
    print(f"Total triggers: {len(CHOREOGRAPHY_TRIGGERS)}")
