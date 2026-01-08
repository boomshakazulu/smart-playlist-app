import argparse
import os
import json
import math
import random
import torch
import soundfile as sf
from einops import rearrange

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def energy_to_words(e):
    if e < 0.35:
        return "low energy, smooth, airy"
    if e < 0.7:
        return "medium energy, steady, smooth"
    return "high energy, punchy, driving"

def valence_to_words(v):
    if v < 0.35:
        return "dark, moody"
    if v < 0.7:
        return "neutral"
    return "bright, uplifting"

def build_prompt(a, b, seconds, user=None):
    # Pull required fields with safe defaults
    a_tempo = float(a.get("tempo", 120))
    b_tempo = float(b.get("tempo", 120))
    a_energy = float(a.get("energy", 0.5))
    b_energy = float(b.get("energy", 0.5))
    a_valence = float(a.get("valence", 0.5))
    b_valence = float(b.get("valence", 0.5))

    target_bpm = int(round((a_tempo + b_tempo) / 2.0))
    target_bpm = clamp(target_bpm, 60, 180)

    target_energy = (a_energy + b_energy) / 2.0
    target_valence = (a_valence + b_valence) / 2.0

    energy_words = energy_to_words(target_energy)
    mood_words = valence_to_words(target_valence)

    # Optional user overrides
    if user:
        bpm_offset = float(user.get("bpmOffset", 0))
        energy_offset = float(user.get("energyOffset", 0))
        target_bpm = int(clamp(target_bpm + bpm_offset, 60, 180))
        target_energy = clamp(target_energy + energy_offset, 0.0, 1.0)
        energy_words = energy_to_words(target_energy)

    # Keep prompts short + specific. This model responds better to concise cues.
    return (
        f"{seconds}-second transition between two songs, "
        f"{energy_words}, {mood_words}, "
        f"smooth riser and gentle drum fill, "
        f"around {target_bpm} BPM, no vocals."
    ), target_bpm, target_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="JSON audio features for track A")
    parser.add_argument("--b", required=True, help="JSON audio features for track B")
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--user", default=None, help="Optional JSON user settings")
    args = parser.parse_args()

    a = json.loads(args.a)
    b = json.loads(args.b)
    user = json.loads(args.user) if args.user else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)
    if args.half and device == "cuda":
        model = model.half()

    seed_value = int(args.seed)
    if seed_value < 0:
        seed_value = random.randint(0, 2_147_483_647)

    prompt, target_bpm, target_energy = build_prompt(a, b, int(args.seconds), user=user)

    conditioning = [{
        "prompt": prompt,
        "seconds_total": int(args.seconds),
    }]

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with torch.no_grad():
        audio = generate_diffusion_cond(
            model,
            steps=int(args.steps),
            cfg_scale=float(args.cfg),
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type="pingpong",
            device=device,
            seed=seed_value,
        )

    audio = rearrange(audio, "b d n -> d (b n)")

    desired_samples = int(args.seconds * sample_rate)
    audio = audio[:, :desired_samples]

    # optional fade-out
    fade_ms = 60
    fade_samples = int(sample_rate * (fade_ms / 1000))
    fade_samples = min(fade_samples, audio.shape[1])
    if fade_samples > 1:
        fade = torch.linspace(1.0, 0.0, fade_samples, device=audio.device)
        audio[:, -fade_samples:] *= fade

    audio = audio.float()
    audio = audio / (audio.abs().max() + 1e-9)
    audio = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu()

    audio_np = audio.numpy().T
    sf.write(out_path, audio_np, sample_rate, subtype="PCM_16")


    # Print useful metadata for Node to parse if you want
    print(out_path)
    print(f"PROMPT={prompt}")
    print(f"TARGET_BPM={target_bpm}")
    print(f"TARGET_ENERGY={target_energy:.3f}")
    print(f"SEED={seed_value}")


if __name__ == "__main__":
    main()
