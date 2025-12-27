# reachy-dance

Beat-synced dancing for Reachy Mini robots. Drop a YouTube link, watch your robot groove.

## Features

- Downloads audio from any YouTube video
- Analyzes tempo and spectral features with librosa
- Smooth sine-wave movements synced to detected BPM
- Interactive mode: queue tracks, adjust volume on the fly
- Maps frequency bands to body parts:
  - Bass → head pitch (nodding)
  - Mids → head yaw (looking around)
  - Highs → antennas

## Requirements

- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot with daemon running
- Python 3.10+
- FFmpeg

## Installation

```bash
pip install reachy-mini yt-dlp librosa pydub numpy
```

## Usage

### Quick Start

```bash
# Play a track
python reachy_youtube_dance.py https://www.youtube.com/watch?v=VIDEO_ID

# With volume control
python reachy_youtube_dance.py https://www.youtube.com/watch?v=VIDEO_ID --volume max

# Interactive mode
python reachy_youtube_dance.py
```

### Volume Levels

| Level | dB | Description |
|-------|-----|-------------|
| `quiet` | -6 | Reduced |
| `normal` | 0 | Original |
| `loud` | +6 | Default |
| `max` | +12 | Maximum |

### Interactive Commands

| Command | Action |
|---------|--------|
| `<URL>` | Play new track |
| `v` | Change volume |
| `q` | Quit |
| `Ctrl+C` | Skip track |

## How It Works

1. **Download**: yt-dlp extracts audio as WAV
2. **Boost**: pydub applies volume adjustment
3. **Analyze**: librosa detects tempo and computes spectral features
4. **Dance**: Smooth oscillations at beat frequency, amplitude-modulated by spectral energy

```python
beat_freq = tempo / 60.0  # Convert BPM to Hz

head_pitch = 10 * (0.5 + bass * 0.5) * sin(2π * beat_freq * t)
head_yaw   = 15 * (0.4 + mids * 0.6) * sin(2π * beat_freq/2 * t)
head_roll  = 8 * sin(2π * beat_freq * t + π/4)
antennas   = 0.8 * (0.3 + highs * 0.7) * sin(2π * beat_freq * 2 * t)
body_yaw   = 12° * sin(2π * beat_freq/4 * t)
```

## Tested Tracks

| Track | Artist | BPM | Notes |
|-------|--------|-----|-------|
| Get Down Saturday Night | Oliver Cheatham | 117 | Classic funk, excellent detection |
| Ecstatic Vibrations | Sea Power | ~80 | Ambient, tests slow tempo |
| Blue Christmas | Elvis Presley | 94.5 | Holiday classic, smooth detection |
| All Shook Up | Elvis Presley | 76 | Rock and roll, solid beat tracking |

## License

MIT
