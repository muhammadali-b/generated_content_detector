"use client";

import { useState } from "react";

type DetectionResult = {
  label: "real" | "ai";
  confidence: number;
};

const API_URL = "http://localhost:8000/detect";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  /**
   * Handle file selection from the file input.
   * Stores the selected file in state and creates a preview URL
   * so the user can see the image before uploading.
   */
  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setErrorMsg(null);

    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  }

  /**
   * Clear the currently selected image and any previous result or error.
   */
  function handleClear() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
  }

  /**
   * Send the selected image file to the FastAPI backend `/detect` endpoint.
   * The file is sent as multipart/form-data using FormData.
   * Once the response arrives, we store the predicted label and confidence.
   */
  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();

    if (!selectedFile) {
      setErrorMsg("Please select an image first.");
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const message =
          errorData?.detail || `Request failed with status ${response.status}`;
        throw new Error(message);
      }

      const data: DetectionResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || "Something went wrong while detecting.");
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * Convert a probability in [0, 1] to a percentage string like "93.7%".
   */
  function formatConfidence(confidence: number): string {
    return `${(confidence * 100).toFixed(1)}%`;
  }

  /**
   * Return Tailwind classes for the label badge based on prediction.
   */
  function labelClasses(label: DetectionResult["label"]): string {
    if (label === "ai") {
      return "bg-purple-500/20 text-purple-300 border border-purple-500/60";
    }
    return "bg-emerald-500/20 text-emerald-300 border border-emerald-500/60";
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-xl bg-slate-900 rounded-2xl shadow-xl border border-slate-700 p-6 md:p-8">
        <h1 className="text-2xl md:text-3xl font-semibold mb-2">
          AI Content Detector
        </h1>
        <p className="text-sm text-slate-300 mb-6">
          Upload an image and the system will predict whether it is{" "}
          <span className="font-semibold">AI-generated</span> or a{" "}
          <span className="font-semibold">real image</span>, together with a
          confidence score.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Select image
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-slate-200
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-full file:border-0
                         file:text-sm file:font-semibold
                         file:bg-indigo-600 file:text-white
                         hover:file:bg-indigo-500"
            />
          </div>

          {previewUrl && (
            <div className="mt-4">
              <p className="text-sm text-slate-300 mb-2">Preview:</p>
              <div className="border border-slate-700 rounded-lg bg-slate-800 flex items-center justify-center">
                {/* 
                  Use max-h and max-w with object-contain so the *full*
                  image is visible without cropping, even for tall images.
                */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Selected preview"
                  className="max-h-80 max-w-full object-contain"
                />
              </div>
            </div>
          )}

          <div className="flex gap-3 mt-2">
            <button
              type="submit"
              disabled={isLoading || !selectedFile}
              className="inline-flex items-center justify-center px-4 py-2
                         rounded-lg bg-indigo-600 text-sm font-medium
                         disabled:opacity-50 disabled:cursor-not-allowed
                         hover:bg-indigo-500 transition-colors"
            >
              {isLoading ? "Analyzing..." : "Detect AI vs Real"}
            </button>

            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading && !selectedFile}
              className="inline-flex items-center justify-center px-4 py-2
                         rounded-lg border border-slate-600 text-sm font-medium
                         text-slate-200 hover:bg-slate-800
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Clear
            </button>
          </div>
        </form>

        {errorMsg && (
          <div className="mt-4 text-sm text-red-400">
            Error: {errorMsg}
          </div>
        )}

        {result && (
          <div className="mt-6 p-4 rounded-lg bg-slate-800 border border-slate-700 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Prediction:</p>
                <p className="text-lg font-semibold">
                  {result.label === "ai" ? "AI-generated" : "Real image"}
                </p>
              </div>
              <span
                className={
                  "text-xs px-3 py-1 rounded-full font-medium " +
                  labelClasses(result.label)
                }
              >
                {result.label.toUpperCase()}
              </span>
            </div>

            <div>
              <p className="text-sm text-slate-300 mb-1">
                Confidence: {formatConfidence(result.confidence)}
              </p>
              {/* Confidence bar */}
              <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden">
                <div
                  className="h-full rounded-full bg-indigo-500"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
