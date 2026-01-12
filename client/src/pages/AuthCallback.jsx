import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getTokensFromUrl, saveTokens } from "../utils/spotifyAuth.js";

function AuthCallback() {
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const tokens = getTokensFromUrl();
    if (!tokens) {
      setError("Missing Spotify auth response.");
      return;
    }

    saveTokens(tokens);
    navigate("/app", { replace: true });
  }, [navigate]);

  return (
    <div className="app">
      <section className="cta">
        <div className="cta-card">
          <h2>{error ? "Login failed" : "Finishing sign-in..."}</h2>
          <p>
            {error
              ? "We could not read the Spotify tokens. Please try again."
              : "Hang tight while we connect your Spotify account."}
          </p>
          {error ? (
            <button className="button primary" onClick={() => navigate("/")}>
              Back to login
            </button>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export default AuthCallback;

