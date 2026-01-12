const fetchPlaylistPreviewMap = async (playlistId, apiBase = "") => {
  try {
    const response = await fetch(
      `${apiBase}/api/previews/playlist/${playlistId}`
    );
    if (!response.ok) {
      throw new Error("Failed to fetch preview map");
    }
    const data = await response.json();
    return data?.previews || {};
  } catch (error) {
    console.error("Error fetching playlist preview data:", error);
    return {};
  }
};

export { fetchPlaylistPreviewMap };
