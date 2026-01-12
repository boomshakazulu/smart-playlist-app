import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_BASE, loadTokens } from "../utils/spotifyAuth.js";

function LoginPage() {
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const stored = loadTokens();
    if (stored.accessToken) {
      navigate("/app", { replace: true });
    }
  }, [navigate]);

  const handleLogin = async () => {
    setError("");
    try {
      const response = await fetch(`${API_BASE}/login`);
      if (!response.ok) {
        throw new Error("Login request failed");
      }
      const data = await response.json();
      if (!data.authURL) {
        throw new Error("Missing Spotify auth URL");
      }
      window.location.assign(data.authURL);
    } catch (err) {
      setError("Unable to reach the login service.");
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-text">
          <p className="eyebrow">Smart Playlist</p>
          <h1>Curate your Spotify playlists, then preview every vibe.</h1>
          <p className="subhead">
            Sign in to surface your playlists, scan their tracks, and audition 30
            second previews without leaving the dashboard.
          </p>
        </div>
        <div className="auth-panel">
          <div>
            <p className="label">Spotify access</p>
            <p className="value">Connect to load playlists.</p>
          </div>
          <button className="button primary" onClick={handleLogin}>
            Connect Spotify
          </button>
        </div>
      </header>

      {error ? <div className="alert">{error}</div> : null}

      <section className="cta">
        <div className="cta-card">
          <h2>One login, then it is all yours.</h2>
          <p>
            We only ask for read access to your playlists so we can list them and
            surface track previews.
          </p>
          <button className="button primary" onClick={handleLogin}>
            Start with Spotify
          </button>
        </div>
      </section>
    </div>
  );
}

export default LoginPage;
