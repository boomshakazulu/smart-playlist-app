import express from "express";
import cors from "cors";
import "dotenv/config";
import axios from "axios";
import querystring from "node:querystring";
import transitionsRouter from "./routes/transitions.js";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const CLIENT_ID = process.env.SPOTIFY_CLIENT_ID;
const CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET;
const REDIRECT_URI = process.env.REDIRECT_URI;
const CLIENT_URL = process.env.CLIENT_URL || "http://localhost:5173";

const tokenCache = {
  accessToken: "",
  expiresAt: 0,
};

const getAppAccessToken = async () => {
  if (tokenCache.accessToken && Date.now() < tokenCache.expiresAt - 30000) {
    return tokenCache.accessToken;
  }

  if (!CLIENT_ID || !CLIENT_SECRET) {
    throw new Error("Missing Spotify client credentials");
  }

  const authHeader = Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString(
    "base64"
  );
  const response = await axios.post(
    "https://accounts.spotify.com/api/token",
    querystring.stringify({ grant_type: "client_credentials" }),
    {
      headers: {
        Authorization: `Basic ${authHeader}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
    }
  );

  const { access_token, expires_in } = response.data;
  tokenCache.accessToken = access_token;
  tokenCache.expiresAt = Date.now() + (expires_in || 3600) * 1000;
  return access_token;
};

// Redirect User to Spotify Auth
app.get("/login", (req, res) => {
  const scope = [
    "user-read-private",
    "user-read-email",
    "playlist-read-private",
    "playlist-read-collaborative",
  ].join(" ");

  const authURL = `https://accounts.spotify.com/authorize?${querystring.stringify(
    {
      response_type: "code",
      client_id: CLIENT_ID,
      scope,
      redirect_uri: REDIRECT_URI,
    }
  )}`;
  res.json({ authURL });
});

// Handle Spotify Callback and Exchange Code for Token
app.get("/callback", async (req, res) => {
  const code = req.query.code || null;
  if (!code) {
    return res.status(400).json({ error: "No code provided" });
  }

  try {
    const response = await axios.post(
      "https://accounts.spotify.com/api/token",
      querystring.stringify({
        grant_type: "authorization_code",
        code: code,
        redirect_uri: REDIRECT_URI,
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
      }),
      {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      }
    );

    const { access_token, refresh_token, expires_in } = response.data;
    const redirectParams = querystring.stringify({
      access_token,
      refresh_token,
      expires_in,
    });

    // Redirect user back to frontend with tokens as query params
    res.redirect(`${CLIENT_URL}/auth?${redirectParams}`);
  } catch (error) {
    res.status(400).json({ error: "Failed to authenticate" });
  }
});

app.get("/refresh", async (req, res) => {
  const refresh_token = req.query.refresh_token || null;
  if (!refresh_token) {
    return res.status(400).json({ error: "No refresh token provided" });
  }

  try {
    const response = await axios.post(
      "https://accounts.spotify.com/api/token",
      querystring.stringify({
        grant_type: "refresh_token",
        refresh_token,
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
      }),
      {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      }
    );

    const { access_token, expires_in } = response.data;
    res.json({ access_token, expires_in });
  } catch (error) {
    res.status(400).json({ error: "Failed to refresh access token" });
  }
});

// serve wav files
app.use("/transitions", express.static(path.join(__dirname, "generated")));

// generation endpoint
app.use("/api/transitions", transitionsRouter);

app.get("/api/audio-features", async (req, res) => {
  const ids = String(req.query.ids || "").trim();
  if (!ids) {
    return res.status(400).json({ error: "ids_required" });
  }

  try {
    let authHeader = req.headers.authorization || "";
    if (!authHeader) {
      const token = await getAppAccessToken();
      authHeader = `Bearer ${token}`;
    }
    const response = await axios.get(
      `https://api.spotify.com/v1/audio-features?ids=${encodeURIComponent(ids)}`,
      {
        headers: { Authorization: authHeader },
      }
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({
      error: "audio_features_failed",
      details: error.response?.data || null,
    });
  }
});

app.get("/api/audio-features/:id", async (req, res) => {
  const { id } = req.params;
  if (!id) {
    return res.status(400).json({ error: "id_required" });
  }

  try {
    let authHeader = req.headers.authorization || "";
    if (!authHeader) {
      const token = await getAppAccessToken();
      authHeader = `Bearer ${token}`;
    }
    const response = await axios.get(
      `https://api.spotify.com/v1/audio-features/${id}`,
      {
        headers: { Authorization: authHeader },
      }
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({
      error: "audio_feature_failed",
      details: error.response?.data || null,
    });
  }
});

app.get("/api/previews/playlist/:playlistId", async (req, res) => {
  const { playlistId } = req.params;
  if (!playlistId) {
    return res.status(400).json({ error: "playlist_id_required" });
  }

  try {
    const response = await axios.get(
      `https://open.spotify.com/embed/playlist/${playlistId}`
    );
    const htmlContent = response.data || "";
    const match = htmlContent.match(
      /<script id="__NEXT_DATA__" type="application\/json">([\s\S]+?)<\/script>/
    );

    if (!match) {
      return res.json({ previews: {} });
    }

    const jsonData = JSON.parse(match[1]);
    const trackList =
      jsonData?.props?.pageProps?.state?.data?.entity?.trackList || [];

    const previews = {};
    trackList.forEach((item) => {
      const audioPreview = item?.audioPreview;
      const previewUrl =
        audioPreview?.url || audioPreview?.previewUrl || audioPreview || "";
      const uri = item?.uri || item?.track?.uri || item?.id || item?.track?.id;
      const id =
        typeof uri === "string" && uri.startsWith("spotify:track:")
          ? uri.split(":").pop()
          : uri;

      if (id && previewUrl) {
        previews[id] = previewUrl;
      }
    });

    res.json({ previews });
  } catch (error) {
    res.status(500).json({ error: "preview_fetch_failed" });
  }
});

app.get("/api/health", (_req, res) => res.json({ ok: true }));

const port = process.env.PORT || 5174;
app.listen(port);
