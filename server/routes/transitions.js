import { Router } from "express";
import { spawn } from "node:child_process";
import readline from "node:readline";
import crypto from "node:crypto";
import path from "node:path";
import fs from "node:fs";
import axios from "axios";
import { fileURLToPath } from "node:url";

const router = Router();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths relative to this file: server/routes/transitions.js
const OUT_DIR = path.resolve(__dirname, "..", "generated");
const PY_SCRIPT = path.resolve(__dirname, "..", "py", "generate_transition.py");
const ANALYZER_SCRIPT = path.resolve(
  __dirname,
  "..",
  "py",
  "analyze_preview.py"
);
const TRANSITION_SERVICE = path.resolve(
  __dirname,
  "..",
  "py",
  "transition_service.py"
);

// full conda.exe path on Windows to avoid shell quoting issues
const CONDA_EXE = "C:\\Users\\Dallon\\miniconda3\\Scripts\\conda.exe";
const PYTHON_EXE = "C:\\Users\\Dallon\\miniconda3\\envs\\stableaudio\\python.exe";

const PREVIEW_CACHE_DIR = path.resolve(OUT_DIR, "preview_cache");

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(PREVIEW_CACHE_DIR, { recursive: true });

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

let serviceProc = null;
let serviceRl = null;
let servicePending = new Map();
let serviceQueue = Promise.resolve();
let exitHookInstalled = false;

const startService = () => {
  if (serviceProc) {
    return;
  }
  serviceProc = spawn(PYTHON_EXE, [TRANSITION_SERVICE], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONWARNINGS: "ignore" },
  });
  serviceRl = readline.createInterface({ input: serviceProc.stdout });
  serviceRl.on("line", (line) => {
    try {
      const trimmed = line.trim();
      if (!trimmed.startsWith("{")) {
        if (trimmed) {
        }
        return;
      }
      const msg = JSON.parse(trimmed);
      if (msg.ready) {
        return;
      }
      const pending = servicePending.get(msg.id);
      if (!pending) {
        return;
      }
      servicePending.delete(msg.id);
      if (msg.ok) {
        pending.resolve(msg);
      } else {
        pending.reject(new Error(msg.error || "generation_failed"));
      }
    } catch (error) {
    }
  });
  serviceProc.stderr.on("data", () => {});
  serviceProc.on("exit", (code) => {
    servicePending.forEach((pending) =>
      pending.reject(new Error("transition_service_exit"))
    );
    servicePending.clear();
    serviceProc = null;
    serviceRl = null;
  });

  if (!exitHookInstalled) {
    exitHookInstalled = true;
    process.on("exit", () => {
      if (serviceProc) {
        serviceProc.kill();
      }
    });
  }
};

const enqueueService = (payload) => {
  const current = serviceQueue.then(
    () =>
      new Promise((resolve, reject) => {
        startService();
        const timeoutId = setTimeout(() => {
          servicePending.delete(payload.id);
          reject(new Error("transition_service_timeout"));
        }, 120000);
        servicePending.set(payload.id, {
          resolve: (msg) => {
            clearTimeout(timeoutId);
            resolve(msg);
          },
          reject: (err) => {
            clearTimeout(timeoutId);
            reject(err);
          },
        });
        serviceProc.stdin.write(`${JSON.stringify(payload)}\n`);
      })
  );
  serviceQueue = current.catch(() => {});
  return current;
};

startService();

const isAllowedPreviewUrl = (urlString) => {
  try {
    const url = new URL(urlString);
    if (url.protocol !== "https:") {
      return false;
    }
    const host = url.hostname.toLowerCase();
    return host.endsWith(".scdn.co") || host.endsWith(".spotifycdn.com");
  } catch (error) {
    return false;
  }
};

const createTempPreviewPath = () =>
  path.join(PREVIEW_CACHE_DIR, `${crypto.randomBytes(12).toString("hex")}.mp3`);

const downloadPreview = async (url, outPath) => {
  if (fs.existsSync(outPath)) {
    return;
  }
  const response = await axios.get(url, { responseType: "arraybuffer" });
  fs.writeFileSync(outPath, response.data);
};

const analyzePreview = (filePath) =>
  new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_EXE, [ANALYZER_SCRIPT, "--path", filePath], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(stderr || "Analyzer failed"));
      }
      const lines = stdout.trim().split(/\r?\n/);
      const lastLine = lines[lines.length - 1];
      try {
        resolve(JSON.parse(lastLine));
      } catch (error) {
        reject(error);
      }
    });
  });

const getPreviewFeatures = async (url, outPath) => {
  await downloadPreview(url, outPath);
  return await analyzePreview(outPath);
};

router.post("/generate", (req, res) => {
  const { a, b, seconds = 12, seed = -1 } = req.body ?? {};
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

router.post("/generate-from-previews", async (req, res) => {
  const {
    aUrl,
    bUrl,
    seconds = 12,
    seed = -1,
    user,
  } = req.body ?? {};
  if (!aUrl || !bUrl) {
    return res.status(400).json({ error: "preview_urls_required" });
  }
  if (!isAllowedPreviewUrl(aUrl) || !isAllowedPreviewUrl(bUrl)) {
    return res.status(400).json({ error: "invalid_preview_url" });
  }

  const aPath = createTempPreviewPath();
  const bPath = createTempPreviewPath();

  try {
    const [a, b] = await Promise.all([
      getPreviewFeatures(aUrl, aPath),
      getPreviewFeatures(bUrl, bPath),
    ]);

    const id = crypto
      .createHash("sha1")
      .update(JSON.stringify({ a, b, seconds }))
      .digest("hex");

    const outPath = path.join(OUT_DIR, `${id}.wav`);

    if (fs.existsSync(outPath)) {
      fs.unlinkSync(outPath);
    }

    try {
      await enqueueService({
        id,
        a,
        b,
        seconds,
        seed,
        outPath,
        half: true,
        user,
      });
    } catch (error) {
      fs.unlink(aPath, () => {});
      fs.unlink(bPath, () => {});
      return res.status(500).json({ error: "gen_failed" });
    }

    fs.unlink(aPath, () => {});
    fs.unlink(bPath, () => {});
    if (!fs.existsSync(outPath)) {
      return res
        .status(500)
        .json({ error: "no_output_file", expected: outPath });
    }
    res.json({ id, url: `/transitions/${id}.wav`, cached: false });
  } catch (error) {
    fs.unlink(aPath, () => {});
    fs.unlink(bPath, () => {});
    res.status(500).json({ error: "preview_transition_failed" });
  }
});

export default router;
