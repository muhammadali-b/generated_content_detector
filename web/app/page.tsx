"use client";

import { useState } from "react";

export default function Home() {
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const callBackend = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch("http://localhost:8000/ping");
      if (!res.ok) {
        throw new Error(`Status ${res.status}`);
      }
      const data = await res.json();
      setResponse(JSON.stringify(data));
    } catch (err: any) {
      setError(err.message || "Something went wrong");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4">
      <h1 className="text-2xl font-bold">AI Content Detector (Demo)</h1>

      <button
        onClick={callBackend}
        className="px-4 py-2 rounded bg-blue-600 text-white"
        disabled={loading}
      >
        {loading ? "Calling backend..." : "Test backend connection"}
      </button>

      {response && (
        <p className="mt-4 text-green-700 break-all">Response: {response}</p>
      )}

      {error && <p className="mt-4 text-red-600">Error: {error}</p>}
    </main>
  );
}
