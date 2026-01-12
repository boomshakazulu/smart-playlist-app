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

def loudness_to_words(l):
    if l < -18:
        return "soft, restrained"
    if l < -10:
        return "moderate loudness"
    return "loud, impactful"

def brightness_to_words(b):
    if b < 0.35:
        return "warm, dark tones"
    if b < 0.7:
        return "balanced brightness"
    return "bright, crisp tones"

def flatness_to_words(f):
    if f < 0.15:
        return "tonal, harmonic"
    if f < 0.35:
        return "mostly tonal"
    return "noisy, textured"

def percussive_to_words(p):
    if p < 0.35:
        return "melodic, sustained"
    if p < 0.65:
        return "balanced groove"
    return "percussive, drum-forward"

def key_to_words(key, mode, strength):
    if not key:
        return "harmonically neutral"
    if strength < 0.08:
        return "tonally ambiguous"
    return f"{key} {mode}"

def groove_to_words(strength):
    if strength < 0.8:
        return "loose groove"
    if strength < 1.6:
        return "steady groove"
    return "driving groove"

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
    a_loudness = float(a.get("loudness", -12))
    b_loudness = float(b.get("loudness", -12))
    a_brightness = float(a.get("spectral_centroid", {}).get("mean", 2000))
    b_brightness = float(b.get("spectral_centroid", {}).get("mean", 2000))
    a_flatness = float(a.get("spectral_flatness", {}).get("mean", 0.2))
    b_flatness = float(b.get("spectral_flatness", {}).get("mean", 0.2))
    a_perc = float(a.get("percussive_ratio", 0.5))
    b_perc = float(b.get("percussive_ratio", 0.5))
    a_key = a.get("key", "")
    b_key = b.get("key", "")
    a_mode = a.get("mode", "major")
    b_mode = b.get("mode", "major")
    a_key_strength = float(a.get("key_strength", 0.0))
    b_key_strength = float(b.get("key_strength", 0.0))
    a_groove = float(a.get("tempo_strength", 1.0))
    b_groove = float(b.get("tempo_strength", 1.0))

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

    a_desc = ", ".join([
        energy_to_words(a_energy),
        valence_to_words(a_valence),
        loudness_to_words(a_loudness),
        brightness_to_words(clamp(a_brightness / (a.get("sample_rate", 22050) / 2.0), 0.0, 1.0)),
        flatness_to_words(a_flatness),
        percussive_to_words(a_perc),
        groove_to_words(a_groove),
        key_to_words(a_key, a_mode, a_key_strength),
    ])

    b_desc = ", ".join([
        energy_to_words(b_energy),
        valence_to_words(b_valence),
        loudness_to_words(b_loudness),
        brightness_to_words(clamp(b_brightness / (b.get("sample_rate", 22050) / 2.0), 0.0, 1.0)),
        flatness_to_words(b_flatness),
        percussive_to_words(b_perc),
        groove_to_words(b_groove),
        key_to_words(b_key, b_mode, b_key_strength),
    ])

    energy_delta = b_energy - a_energy
    valence_delta = b_valence - a_valence
    tempo_delta = b_tempo - a_tempo
    brightness_delta = b_brightness - a_brightness
    loudness_delta = b_loudness - a_loudness
    perc_delta = b_perc - a_perc

    def delta_words(value, neg, pos):
        if value < -0.15:
            return neg
        if value > 0.15:
            return pos
        return "subtle shift"

    energy_shift = delta_words(energy_delta, "ease down in intensity", "build intensity")
    mood_shift = delta_words(valence_delta, "darker mood", "brighter mood")
    tempo_shift = delta_words(tempo_delta / 60.0, "slow down", "speed up")
    brightness_shift = delta_words(brightness_delta / (a.get("sample_rate", 22050) / 2.0), "warm/darken the tone", "brighten the tone")
    loudness_shift = delta_words(loudness_delta / 10.0, "soften the loudness", "increase loudness")
    perc_shift = delta_words(perc_delta, "less percussive", "more percussive")

    # Keep prompts short + specific. This model responds better to concise cues.
    prompt = (
        f"{seconds}-second transition that clearly moves from track A to track B. "
        f"Start in A only: {a_desc}. "
        f"Then morph to B only: {b_desc}. "
        f"Make the change obvious: {energy_shift}, {mood_shift}, {brightness_shift}, {loudness_shift}, {perc_shift}. "
        f"Tempo must start at {int(round(a_tempo))} BPM and end at {int(round(b_tempo))} BPM, "
        f"with a continuous tempo ramp across the middle. "
        f"First 35% matches A, middle 30% is the shift, last 35% matches B. "
        f"Keep a tight rhythmic pulse locked to the BPM at each stage, emphasize percussion and groove, "
        f"avoid organ, slow keyboard pads, or ambient washes, no vocals."
    )

    return prompt, target_bpm, target_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="JSON audio features for track A")
    parser.add_argument("--b", required=True, help="JSON audio features for track B")
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--user", default=None, help="Optional JSON user settings")
    args = parser.parse_args()

    a = json.loads(args.a)
    b = json.loads(args.b)
    user = json.loads(args.user) if args.user else None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transition generation.")
    device = "cuda"

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
