import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  API_BASE,
  clearTokens,
  loadTokens,
  saveTokens,
} from "../utils/spotifyAuth.js";
import { fetchPlaylistPreviewMap } from "../utils/spotifyPreview.js";

function DashboardPage() {
  const [accessToken, setAccessToken] = useState("");
  const [refreshToken, setRefreshToken] = useState("");
  const [expiresAt, setExpiresAt] = useState(0);
  const [profile, setProfile] = useState(null);
  const [playlists, setPlaylists] = useState([]);
  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [previewMap, setPreviewMap] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [nowPlayingUrl, setNowPlayingUrl] = useState("");
  const audioRef = useRef(null);
  const transitionAudioRef = useRef(null);
  const transitionMetaRef = useRef({
    nextIndex: -1,
    url: "",
    pendingPlayback: false,
    failed: false,
    crossfading: false,
    handoff: false,
  });
  const transitionRequestIdRef = useRef(0);
  const tracksRef = useRef([]);
  const previewMapRef = useRef({});
  const crossfadeTimerRef = useRef(null);
  const nextCrossfadeTimerRef = useRef(null);
  const preloadAudioRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  useEffect(() => {
    previewMapRef.current = previewMap;
  }, [previewMap]);

  useEffect(() => {
    audioRef.current = new Audio();
    transitionAudioRef.current = new Audio();
    preloadAudioRef.current = new Audio();

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }
      if (transitionAudioRef.current) {
        transitionAudioRef.current.pause();
        transitionAudioRef.current.src = "";
      }
      if (preloadAudioRef.current) {
        preloadAudioRef.current.pause();
        preloadAudioRef.current.src = "";
      }
    };
  }, []);

  useEffect(() => {
    const stored = loadTokens();
    if (!stored.accessToken) {
      navigate("/", { replace: true });
      return;
    }

    setAccessToken(stored.accessToken);
    setRefreshToken(stored.refreshToken);
    setExpiresAt(stored.expiresAt);
  }, [navigate]);

  useEffect(() => {
    if (!selectedPlaylist?.id) {
      setPreviewMap({});
      return;
    }

    setPreviewMap({});
    fetchPlaylistPreviewMap(selectedPlaylist.id, API_BASE)
      .then((map) => setPreviewMap(map))
      .catch((err) => console.error(err));
  }, [selectedPlaylist?.id]);

  const updateTransitionMeta = (updates) => {
    transitionMetaRef.current = { ...transitionMetaRef.current, ...updates };
  };

  const stopAllAudio = () => {
    transitionRequestIdRef.current += 1;
    if (crossfadeTimerRef.current) {
      clearTimeout(crossfadeTimerRef.current);
      crossfadeTimerRef.current = null;
    }
    if (nextCrossfadeTimerRef.current) {
      clearTimeout(nextCrossfadeTimerRef.current);
      nextCrossfadeTimerRef.current = null;
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
      audioRef.current.volume = 1;
    }
    if (transitionAudioRef.current) {
      transitionAudioRef.current.pause();
      transitionAudioRef.current.currentTime = 0;
      transitionAudioRef.current.onended = null;
      transitionAudioRef.current.volume = 1;
    }

    updateTransitionMeta({
      nextIndex: -1,
      url: "",
      pendingPlayback: false,
      failed: false,
      crossfading: false,
      handoff: false,
    });
    setNowPlayingUrl("");
  };

  const clearSession = () => {
    stopAllAudio();
    clearTokens();
    setAccessToken("");
    setRefreshToken("");
    setExpiresAt(0);
    setPlaylists([]);
    setSelectedPlaylist(null);
    setTracks([]);
    setProfile(null);
    navigate("/", { replace: true });
  };

  const ensureAccessToken = async () => {
    if (!accessToken) {
      return "";
    }

    if (!expiresAt || Date.now() < expiresAt - 10000) {
      return accessToken;
    }

    if (!refreshToken) {
      return accessToken;
    }

    try {
      const response = await fetch(
        `${API_BASE}/refresh?refresh_token=${encodeURIComponent(refreshToken)}`
      );
      if (!response.ok) {
        throw new Error("Failed to refresh access token");
      }
      const data = await response.json();
      if (!data.access_token) {
        throw new Error("Missing access token");
      }

      const stored = saveTokens({
        access_token: data.access_token,
        refresh_token: refreshToken,
        expires_in: data.expires_in || 3600,
      });
      setAccessToken(stored.accessToken);
      setExpiresAt(stored.expiresAt);
      return stored.accessToken;
    } catch (err) {
      setError("Session expired. Please sign in again.");
      clearSession();
      return "";
    }
  };

  const fetchProfileAndPlaylists = async () => {
    setLoading(true);
    setError("");
    try {
      const token = await ensureAccessToken();
      if (!token) {
        setLoading(false);
        return;
      }

      const [profileRes, playlistRes] = await Promise.all([
        fetch("https://api.spotify.com/v1/me", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("https://api.spotify.com/v1/me/playlists?limit=50", {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      if (!profileRes.ok || !playlistRes.ok) {
        throw new Error("Failed to load Spotify data");
      }

      const profileData = await profileRes.json();
      const playlistData = await playlistRes.json();
      setProfile(profileData);
      setPlaylists(playlistData.items || []);
    } catch (err) {
      setError("We could not load your Spotify library.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (accessToken) {
      fetchProfileAndPlaylists();
    }
  }, [accessToken]);

  const handleSelectPlaylist = async (playlist) => {
    stopAllAudio();
    setSelectedPlaylist(playlist);
    setTracks([]);
    setPreviewMap({});
    setLoading(true);
    setError("");

    try {
      const token = await ensureAccessToken();
      if (!token) {
        setLoading(false);
        return;
      }

      const response = await fetch(
        `https://api.spotify.com/v1/playlists/${playlist.id}/tracks?limit=50`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!response.ok) {
        throw new Error("Failed to load playlist tracks");
      }
      const data = await response.json();
      setTracks(data.items || []);
    } catch (err) {
      setError("We could not load that playlist's tracks.");
    } finally {
      setLoading(false);
    }
  };

  const getPreviewUrl = (track) => {
    if (!track?.id) {
      return "";
    }
    return track.preview_url || previewMapRef.current[track.id] || "";
  };

  const findNextPlayableIndex = (startIndex) => {
    const list = tracksRef.current;
    for (let i = startIndex + 1; i < list.length; i += 1) {
      const nextTrack = list[i]?.track;
      if (nextTrack?.id && getPreviewUrl(nextTrack)) {
        return i;
      }
    }
    return -1;
  };

  const playNextPreviewFromMeta = () => {
    const meta = transitionMetaRef.current;
    const nextIdx = meta.nextIndex;
    const next = tracksRef.current[nextIdx]?.track;
    if (next) {
      const previewUrl = getPreviewUrl(next);
      if (preloadAudioRef.current?.src === previewUrl && previewUrl) {
        audioRef.current = preloadAudioRef.current;
        preloadAudioRef.current = new Audio();
        startPreview(next, nextIdx, { skipStop: true });
      } else {
        startPreview(next, nextIdx);
      }
    }
  };

  const adoptPreviewPlayback = (previewAudio, previewUrl, index) => {
    audioRef.current = previewAudio;
    setNowPlayingUrl(previewUrl);

    const nextIndex = findNextPlayableIndex(index);
    const nextTrack =
      nextIndex >= 0 ? tracksRef.current[nextIndex]?.track : null;
    const nextPreviewUrl = nextTrack ? getPreviewUrl(nextTrack) : "";
    if (nextPreviewUrl && preloadAudioRef.current) {
      preloadAudioRef.current.src = nextPreviewUrl;
      preloadAudioRef.current.preload = "auto";
      preloadAudioRef.current.load();
    }

    updateTransitionMeta({
      nextIndex,
      url: "",
      pendingPlayback: false,
      failed: false,
      crossfading: false,
      handoff: true,
    });

    previewAudio.onended = () => {
      setNowPlayingUrl("");

      const meta = transitionMetaRef.current;
      if (meta.crossfading) {
        return;
      }
      if (!meta.url || !transitionAudioRef.current) {
        if (meta.failed) {
          playNextPreviewFromMeta();
          return;
        }

        updateTransitionMeta({ pendingPlayback: true });
        return;
      }

      playTransition();
    };

    if (nextTrack) {
      requestTransition(tracksRef.current[index]?.track, nextTrack);
    }
  };

  const scheduleCrossfade = (transitionUrl) => {
    const previewAudio = audioRef.current;
    const transitionAudio = transitionAudioRef.current;
    if (!previewAudio || !transitionAudio || previewAudio.paused) {
      console.log("crossfade_skip", "no_preview_or_paused");
      return false;
    }

    const duration = previewAudio.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      console.log("crossfade_skip", "unknown_duration");
      return false;
    }

    const crossfadeMs = 2200;
    const crossfadeLeadMs = 4000;
    const remainingMs = Math.max(
      0,
      Math.floor((duration - previewAudio.currentTime) * 1000)
    );
    const delayMs = Math.max(0, remainingMs - crossfadeLeadMs);

    if (crossfadeTimerRef.current) {
      clearTimeout(crossfadeTimerRef.current);
    }

    crossfadeTimerRef.current = setTimeout(() => {
      console.log("crossfade_start", transitionUrl);
      transitionAudio.pause();
      transitionAudio.currentTime = 0;
      transitionAudio.src = transitionUrl;
      transitionAudio.volume = 0;
      transitionAudio.onloadedmetadata = () => {
        console.log(
          "transition_duration_sec",
          transitionAudio.duration,
          transitionUrl
        );
      };
      transitionAudio.play().catch((err) => {
        console.warn("transition_play_failed", err);
      });
      transitionAudio.onended = () => {
        console.log("transition_end", transitionUrl);
        playNextPreviewFromMeta();
      };

      startNextOverlap(transitionAudio);

      updateTransitionMeta({ crossfading: true });

      const start = performance.now();
      const step = () => {
        const t = Math.min(1, (performance.now() - start) / crossfadeMs);
        previewAudio.volume = 1 - t;
        transitionAudio.volume = t;

        if (t < 1) {
          requestAnimationFrame(step);
        } else {
          previewAudio.pause();
          previewAudio.currentTime = 0;
          previewAudio.volume = 1;
          updateTransitionMeta({ crossfading: false });
        }
      };
      requestAnimationFrame(step);
    }, delayMs);

    return true;
  };

  const startNextOverlap = (transitionAudio) => {
    const meta = transitionMetaRef.current;
    const nextIdx = meta.nextIndex;
    const nextTrack = tracksRef.current[nextIdx]?.track;
    const nextPreviewUrl = nextTrack ? getPreviewUrl(nextTrack) : "";
    if (!nextPreviewUrl || !preloadAudioRef.current) {
      return;
    }

    const nextAudio = preloadAudioRef.current;
    const fadeMs = 2000;
    const overlapDelayMs = 7000;

    if (nextCrossfadeTimerRef.current) {
      clearTimeout(nextCrossfadeTimerRef.current);
    }

    nextCrossfadeTimerRef.current = setTimeout(() => {
      console.log("next_preview_overlap_start", nextPreviewUrl);
      nextAudio.pause();
      nextAudio.currentTime = 0;
      nextAudio.src = nextPreviewUrl;
      nextAudio.volume = 0;
      nextAudio.play().catch((err) => {
        console.warn("next_preview_play_failed", err);
      });

      const start = performance.now();
      const step = () => {
        const t = Math.min(1, (performance.now() - start) / fadeMs);
        transitionAudio.volume = 1 - t;
        nextAudio.volume = t;

        if (t < 1) {
          requestAnimationFrame(step);
        } else {
          transitionAudio.pause();
          transitionAudio.currentTime = 0;
          transitionAudio.volume = 1;
          preloadAudioRef.current = new Audio();
          adoptPreviewPlayback(nextAudio, nextPreviewUrl, nextIdx);
        }
      };
      requestAnimationFrame(step);
    }, overlapDelayMs);
  };

  const requestTransition = async (currentTrack, nextTrack) => {
    const requestId = transitionRequestIdRef.current + 1;
    transitionRequestIdRef.current = requestId;

    try {
      const currentPreviewUrl = getPreviewUrl(currentTrack);
      const nextPreviewUrl = getPreviewUrl(nextTrack);
      if (!currentPreviewUrl || !nextPreviewUrl) {
        throw new Error("Missing preview audio for analysis");
      }

      const response = await fetch(
        `${API_BASE}/api/transitions/generate-from-previews`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            aUrl: currentPreviewUrl,
            bUrl: nextPreviewUrl,
            seconds: 12,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to generate transition");
      }

      const data = await response.json();
      if (!data.url) {
        throw new Error("Missing transition url");
      }

      if (transitionRequestIdRef.current !== requestId) {
        return;
      }

      const fullUrl = `${API_BASE}${data.url}`;
      updateTransitionMeta({ url: fullUrl });
      if (scheduleCrossfade(fullUrl)) {
        updateTransitionMeta({ pendingPlayback: false });
        return;
      }

      if (transitionMetaRef.current.pendingPlayback) {
        updateTransitionMeta({ pendingPlayback: false });
        playTransition();
      }
    } catch (err) {
      console.error(err);
      updateTransitionMeta({ failed: true });
      if (transitionMetaRef.current.pendingPlayback) {
        updateTransitionMeta({ pendingPlayback: false });
        playNextPreviewFromMeta();
      }
    }
  };

  const playTransition = () => {
    const meta = transitionMetaRef.current;
    if (!meta.url || !transitionAudioRef.current) {
      return;
    }

    const transitionAudio = transitionAudioRef.current;
    transitionAudio.pause();
    transitionAudio.currentTime = 0;
    transitionAudio.src = meta.url;
    transitionAudio.play();

    transitionAudio.onended = () => {
      const meta = transitionMetaRef.current;
      if (meta.handoff) {
        return;
      }
      playNextPreviewFromMeta();
    };

    startNextOverlap(transitionAudio);
  };

  const startPreview = async (track, index, options = {}) => {
    const previewUrl = getPreviewUrl(track);
    if (!previewUrl || !audioRef.current) {
      setError("No preview available for this track.");
      return;
    }

    if (options.skipStop) {
      if (transitionAudioRef.current) {
        transitionAudioRef.current.pause();
        transitionAudioRef.current.currentTime = 0;
        transitionAudioRef.current.onended = null;
        transitionAudioRef.current.volume = 1;
      }
    } else {
      stopAllAudio();
    }
    setError("");

    const previewAudio = audioRef.current;
    previewAudio.src = previewUrl;
    previewAudio.play();
    setNowPlayingUrl(previewUrl);

    const nextIndex = findNextPlayableIndex(index);
    const nextTrack =
      nextIndex >= 0 ? tracksRef.current[nextIndex]?.track : null;
    const nextPreviewUrl = nextTrack ? getPreviewUrl(nextTrack) : "";
    if (nextPreviewUrl && preloadAudioRef.current) {
      preloadAudioRef.current.src = nextPreviewUrl;
      preloadAudioRef.current.preload = "auto";
      preloadAudioRef.current.load();
    }

    updateTransitionMeta({
      nextIndex,
      url: "",
      pendingPlayback: false,
      failed: false,
      crossfading: false,
    });

    previewAudio.onended = () => {
      setNowPlayingUrl("");

      const meta = transitionMetaRef.current;
      if (meta.crossfading) {
        return;
      }
      if (!meta.url || !transitionAudioRef.current) {
        if (meta.failed) {
          playNextPreviewFromMeta();
          return;
        }

        updateTransitionMeta({ pendingPlayback: true });
        return;
      }

      playTransition();
    };

    if (nextTrack) {
      requestTransition(track, nextTrack);
    }
  };

  const togglePreview = (track, index) => {
    const previewUrl = getPreviewUrl(track);
    if (!previewUrl) {
      setError("No preview available for this track.");
      return;
    }

    if (nowPlayingUrl === previewUrl) {
      stopAllAudio();
      return;
    }

    startPreview(track, index);
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
            <p className="label">Signed in</p>
            <p className="value">{profile?.display_name || "Spotify listener"}</p>
          </div>
          <button className="button ghost" onClick={clearSession}>
            Sign out
          </button>
        </div>
      </header>

      {error ? <div className="alert">{error}</div> : null}

      <section className="dashboard">
        <div className="panel">
          <div className="panel-header">
            <h2>Your playlists</h2>
            <p>{playlists.length} available</p>
          </div>
          <div className="playlist-grid">
            {playlists.map((playlist) => {
              const image = playlist.images?.[0]?.url;
              const active = selectedPlaylist?.id === playlist.id;
              return (
                <button
                  key={playlist.id}
                  className={`playlist-card ${active ? "active" : ""}`}
                  onClick={() => handleSelectPlaylist(playlist)}
                >
                  <div className="playlist-thumb">
                    {image ? (
                      <img src={image} alt={playlist.name} />
                    ) : (
                      <div className="playlist-fallback" />
                    )}
                  </div>
                  <div className="playlist-meta">
                    <span className="playlist-name">{playlist.name}</span>
                    <span className="playlist-count">
                      {playlist.tracks?.total || 0} tracks
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{selectedPlaylist ? selectedPlaylist.name : "Select a playlist"}</h2>
            <p>
              {selectedPlaylist
                ? `${tracks.length} loaded`
                : "Choose a playlist to preview its tracks."}
            </p>
          </div>
          <div className="track-list">
            {tracks.length === 0 && !loading ? (
              <div className="empty">No tracks loaded yet.</div>
            ) : null}
            {tracks.map((item, index) => {
              const track = item.track;
              if (!track) {
                return null;
              }
              const previewUrl =
                track.preview_url || previewMap[track.id] || "";
              const key = track.id || track.uri || `${track.name}-${index}`;
              return (
                <div className="track-row" key={key}>
                  <div className="track-info">
                    <span className="track-title">{track.name}</span>
                    <span className="track-artist">
                      {track.artists?.map((artist) => artist.name).join(", ")}
                    </span>
                  </div>
                  <div className="track-actions">
                    <span className="track-tag">
                      {previewUrl ? "Preview" : "No preview"}
                    </span>
                    <button
                      className="button small"
                      disabled={!previewUrl}
                      onClick={() => togglePreview(track, index)}
                    >
                      {nowPlayingUrl === previewUrl ? "Stop" : "Play"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {loading ? <div className="loading">Loading Spotify data...</div> : null}
    </div>
  );
}

export default DashboardPage;
