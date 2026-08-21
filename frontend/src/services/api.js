const DEFAULT_PRODUCTION_BACKEND = "https://voice-rag-goa-2026-jhpr.onrender.com";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? DEFAULT_PRODUCTION_BACKEND : "");

export async function checkBackendHealth(retries = 2, delayMs = 2500) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      if (attempt === retries) throw err;
    }
    if (attempt < retries) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw new Error("Backend health check failed after retries.");
}
export async function sendTextQuery(query, language = "en") {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      language,
    }),
  });

  if (!response.ok) {
    throw new Error(`Query failed: ${response.status}`);
  }

  return response.json();
}
export async function sendVoiceQuery(audioBlob, languageHint = "en-IN") {
  const formData = new FormData();

  formData.append("file", audioBlob, "voice-query.wav");
  formData.append("language_hint", languageHint);

  const response = await fetch(`${API_BASE_URL}/api/voice/query`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Voice query failed: ${response.status}`);
  }

  return response.json();
}

export { API_BASE_URL };
