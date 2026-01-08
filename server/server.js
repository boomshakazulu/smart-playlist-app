import express from "express";
import cors from "cors";
import "dotenv/config";
import transitionsRouter from "./routes/transitions.js";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// serve wav files
app.use("/transitions", express.static(path.join(__dirname, "generated")));

// generation endpoint
app.use("/api/transitions", transitionsRouter);

app.get("/api/health", (_req, res) => res.json({ ok: true }));

const port = process.env.PORT || 5174;
app.listen(port, () => console.log(`API on http://localhost:${port}`));
