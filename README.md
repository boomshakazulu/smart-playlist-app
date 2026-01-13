# Smart Playlist App

Enhances Spotify playlists with smooth, AI-generated transitions between tracks. The app fetches your playlists, plays 30-second previews, and generates a transition clip that blends from track A to track B using tempo/energy analysis.

## Features

- Spotify OAuth login and playlist browsing.
- 30-second Spotify preview playback.
- AI transition generation using preview analysis (tempo/energy/timbre).
- Hidden per-track tempo and speed overrides stored in localStorage.
- Playlist and track pagination (50 tracks per page).

## How It Works

1. The client uses Spotify OAuth to fetch playlists and track previews.
2. When you play a preview, the next track preview is analyzed server-side using `librosa`.
3. The transition model generates a short clip that ramps from track A to track B.
4. The UI crossfades the preview into the generated transition, then into the next preview.

## Tech Stack

- Frontend: React (Vite)
- Backend: Node.js + Express
- Audio Analysis: Python + librosa
- Audio Generation: Stable Audio (`stabilityai/stable-audio-open-small`)

## Project Structure

- `client/` - React frontend
- `server/` - Express API
- `server/routes/transitions.js` - Transition generation route
- `server/py/analyze_preview.py` - Feature extraction
- `server/py/generate_transition.py` - Prompt building + generation (CLI)
- `server/py/transition_service.py` - Long-lived generation service

## Requirements

- Node.js 18+
- Python 3.10+ (tested with conda)
- CUDA-capable GPU for generation (required by Stable Audio)
- Spotify Developer App credentials

## Setup

### 1) Install dependencies

```bash
npm install
```

```bash
cd client
npm install
```

### 2) Configure Spotify OAuth

Create a Spotify app and set the redirect URI to match the client URL.

Create a `.env` file in `server/`:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:{your-port}/callback
```

### 3) Python environment

Create a conda environment and install the Python dependencies:

```bash
conda create -n stableaudio python=3.10
conda activate stableaudio
pip install torch soundfile librosa numpy einops stable-audio-tools
```

Update these paths if needed:

- `server/routes/transitions.js` -> `CONDA_EXE`
- `server/routes/transitions.js` -> `PYTHON_EXE`

## Run Locally

### Start the server

```bash
cd server
npm install
npm run dev
```

### Start the client

```bash
cd client
npm run dev
```

Open: `http://localhost:5173`

## Hidden Per-Track Overrides

You can reveal per-track overrides (tempo/speed) using the "Show settings" button in the tracks panel. Settings are stored in localStorage under:

- `transition_track_settings_v1`

These overrides are averaged between track A and track B and passed to the generator as `user.bpmOffset` and `user.energyOffset`.

## API Endpoints

### POST `/api/transitions/generate-from-previews`

Request body:

```json
{
  "aUrl": "https://p.scdn.co/.../preview.mp3",
  "bUrl": "https://p.scdn.co/.../preview.mp3",
  "seconds": 16,
  "seed": -1,
  "user": {
    "bpmOffset": 0,
    "energyOffset": 0
  }
}
```

Response:

```json
{
  "id": "hash",
  "url": "/transitions/hash.wav",
  "cached": false
}
```

## Tuning Generation

Defaults live in `server/py/transition_service.py`:

- `seconds`: 16
- `steps`: 22
- `cfg`: 1.3

Prompt logic lives in `server/py/generate_transition.py`. Reducing prompt complexity can make outputs more natural but less controlled.

## Known Limitations

- The `stable-audio-open-small` model can produce organ-like or synthetic textures.
- Generation requires CUDA; CPU-only machines will fail.
- Background tabs in Chrome can throttle audio timers (playback may pause until focus returns).

## Troubleshooting

- **Preview missing:** Some Spotify tracks do not have 30-second previews.
- **Generation fails:** Verify CUDA availability and Python env paths.
- **No transitions:** Ensure the server can write to `server/generated`.

## License

MIT
