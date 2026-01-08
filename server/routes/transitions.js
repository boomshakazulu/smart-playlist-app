import { Router } from "express";
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const router = Router();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths relative to this file: server/routes/transitions.js
const OUT_DIR = path.resolve(__dirname, "..", "generated");
const PY_SCRIPT = path.resolve(__dirname, "..", "py", "generate_transition.py");

// full conda.exe path on Windows to avoid shell quoting issues
const CONDA_EXE = "C:\\Users\\Dallon\\miniconda3\\Scripts\\conda.exe";

fs.mkdirSync(OUT_DIR, { recursive: true });

function runPy({ a, b, seconds, outPath, seed = 12345, half = true }) {
  const args = [
    "run",
    "-n",
    "stableaudio",
    "python",
    PY_SCRIPT,
    "--a",
    JSON.stringify(a),
    "--b",
    JSON.stringify(b),
    "--seconds",
    String(seconds),
    "--out",
    outPath,
    "--seed",
    String(seed),
  ];

  if (half) args.push("--half");

  return spawn(CONDA_EXE, args, { stdio: ["ignore", "pipe", "pipe"] });
}

router.post("/generate", (req, res) => {
  const { a, b, seconds = 6, seed = 12345 } = req.body ?? {};
  if (!a || !b) return res.status(400).json({ error: "a_and_b_required" });

  const id = crypto
    .createHash("sha1")
    .update(JSON.stringify({ a, b, seconds }))
    .digest("hex");

  const outPath = path.join(OUT_DIR, `${id}.wav`);

  if (fs.existsSync(outPath)) {
    return res.json({ id, url: `/transitions/${id}.wav`, cached: true });
  }

  const p = runPy({ a, b, seconds, outPath, seed, half: true });

  let stderr = "";
  p.stderr.on("data", (d) => (stderr += d.toString()));

  p.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: "gen_failed",
        details: stderr.slice(-4000),
      });
    }
    if (!fs.existsSync(outPath)) {
      return res
        .status(500)
        .json({ error: "no_output_file", expected: outPath });
    }
    res.json({ id, url: `/transitions/${id}.wav`, cached: false });
  });
});

export default router;
