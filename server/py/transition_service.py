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


os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transition generation.")
    return "cuda"


def generate_transition(model, model_config, req):
    a = req.get("a", {})
    b = req.get("b", {})
    seconds = int(req.get("seconds", 6))
    steps = int(req.get("steps", 12))
    cfg = float(req.get("cfg", 2.0))
    seed = int(req.get("seed", -1))
    out_path = req.get("outPath")
    half = bool(req.get("half", True))

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

    prompt, target_bpm, target_energy = build_prompt(a, b, seconds, user=req.get("user"))

    conditioning = [{
        "prompt": prompt,
        "seconds_total": seconds,
    }]

    with torch.no_grad():
        audio = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg,
            conditioning=conditioning,
            sample_size=model_config["sample_size"],
            sampler_type="pingpong",
            device=device,
            seed=seed,
        )

    audio = rearrange(audio, "b d n -> d (b n)")
    sample_rate = int(model_config["sample_rate"])

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
    sf.write(out_path, audio_np, sample_rate, subtype="PCM_16")

    return {
        "prompt": prompt,
        "target_bpm": target_bpm,
        "target_energy": target_energy,
        "seed": seed,
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
