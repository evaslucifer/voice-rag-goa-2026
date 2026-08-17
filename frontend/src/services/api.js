// const API_BASE_URL =
//   import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_BASE_URL = "";
export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Backend health check failed: ${response.status}`);
  }

  return response.json();
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

export { API_BASE_URL };