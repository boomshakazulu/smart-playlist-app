const SETTINGS_KEY = "transition_track_settings_v1";

const safeParse = (value, fallback) => {
  if (!value) {
    return fallback;
  }
  try {
    return JSON.parse(value);
  } catch (err) {
    return fallback;
  }
};

export const loadTrackSettings = () =>
  safeParse(localStorage.getItem(SETTINGS_KEY), {});

export const saveTrackSettings = (settings) => {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
};
