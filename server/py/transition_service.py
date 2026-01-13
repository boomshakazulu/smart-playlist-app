import json
import os
import random
import sys
import time
import warnings

import torch
import soundfile as sf
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from generate_transition import build_prompt
from analyze_preview import analyze_audio


os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transition generation.")
    return "cuda"

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def normalize_brightness(centroid_mean: float, sample_rate: int) -> float:
    return clamp(centroid_mean / (sample_rate / 2.0), 0.0, 1.0)

def build_targets(a, b, target_bpm, target_energy, target_valence):
    a_sr = int(a.get("sample_rate", 22050))
    b_sr = int(b.get("sample_rate", 22050))
    a_brightness = normalize_brightness(
        float(a.get("spectral_centroid", {}).get("mean", 0.0)), a_sr
    )
    b_brightness = normalize_brightness(
        float(b.get("spectral_centroid", {}).get("mean", 0.0)), b_sr
    )
    return {
        "tempo": float(target_bpm),
        "energy": float(target_energy),
        "valence": float(target_valence),
        "brightness": (a_brightness + b_brightness) / 2.0,
        "percussive": (float(a.get("percussive_ratio", 0.5)) + float(b.get("percussive_ratio", 0.5))) / 2.0,
    }

def score_candidate(features, targets):
    tempo = float(features.get("tempo", targets["tempo"]))
    energy = float(features.get("energy", targets["energy"]))
    valence = float(features.get("valence", targets["valence"]))
    sample_rate = int(features.get("sample_rate", 22050))
    brightness = normalize_brightness(
        float(features.get("spectral_centroid", {}).get("mean", 0.0)), sample_rate
    )
    percussive = float(features.get("percussive_ratio", targets["percussive"]))

    tempo_penalty = abs(tempo - targets["tempo"]) / max(targets["tempo"], 1.0)
    return (
        tempo_penalty * 0.35
        + abs(energy - targets["energy"]) * 0.25
        + abs(valence - targets["valence"]) * 0.2
        + abs(brightness - targets["brightness"]) * 0.1
        + abs(percussive - targets["percussive"]) * 0.1
    )


def generate_transition(model, model_config, req):
    a = req.get("a", {})
    b = req.get("b", {})
    seconds = int(req.get("seconds", 16))
    steps = int(req.get("steps", 22))
    cfg = float(req.get("cfg", 1.3))
    seed = int(req.get("seed", -1))
    out_path = req.get("outPath")
    half = bool(req.get("half", True))
    candidates = int(req.get("candidates", 2))

    if not out_path:
        raise RuntimeError("Missing outPath")

    device = ensure_cuda()
    model = model.to(device)
    if half:
        model = model.half()
    else:
        model = model.float()

    if seed < 0:
        seed = random.randint(0, 2_147_483_647)

    prompt, target_bpm, target_energy, target_valence = build_prompt(
        a, b, seconds, user=req.get("user")
    )
    targets = build_targets(a, b, target_bpm, target_energy, target_valence)
    candidates = max(1, min(candidates, 4))

    conditioning = [{
        "prompt": prompt,
        "seconds_total": seconds,
    }]

    sample_rate = int(model_config["sample_rate"])
    base_seed = seed
    best_score = None
    best_seed = seed
    best_path = None
    output_paths = []

    def render_audio(seed_value, output_path):
        with torch.no_grad():
            audio = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg,
                conditioning=conditioning,
                sample_size=model_config["sample_size"],
                sampler_type="pingpong",
                device=device,
                seed=seed_value,
            )

        audio = rearrange(audio, "b d n -> d (b n)")
        desired_samples = int(seconds * sample_rate)
        audio = audio[:, :desired_samples]

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
        sf.write(output_path, audio_np, sample_rate, subtype="PCM_16")

    for idx in range(candidates):
        seed_value = base_seed + idx
        output_path = f"{out_path}.cand{idx}.wav"
        output_paths.append(output_path)
        render_audio(seed_value, output_path)
        features = analyze_audio(output_path, sr=sample_rate)
        score = score_candidate(features, targets)
        if best_score is None or score < best_score:
            best_score = score
            best_seed = seed_value
            best_path = output_path

    if not best_path:
        raise RuntimeError("No candidate generated")

    os.replace(best_path, out_path)
    for candidate_path in output_paths:
        if candidate_path != best_path and os.path.exists(candidate_path):
            os.remove(candidate_path)

    return {
        "prompt": prompt,
        "target_bpm": target_bpm,
        "target_energy": target_energy,
        "target_valence": target_valence,
        "seed": best_seed,
        "outPath": out_path,
    }


def main():
    device = ensure_cuda()
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
    model = model.to(device)
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    log("transition_service_ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id", "")
            log(f"transition_service_job_start {req_id}")
            start = time.time()
            result = generate_transition(model, model_config, req)
            elapsed = time.time() - start
            log(f"transition_service_job_end {req_id} {elapsed:.2f}s")
            sys.stdout.write(json.dumps({"id": req_id, "ok": True, **result}) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            log(f"transition_service_job_error {req.get('id', '')} {exc}")
            sys.stdout.write(
                json.dumps(
                    {"id": req.get("id", ""), "ok": False, "error": str(exc)}
                )
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
