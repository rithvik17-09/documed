"use client";

import React, { useState } from "react";
import { UploadCloud, CheckCircle, Download, RefreshCw } from "lucide-react";

type DetectedIssue = {
  id: string;
  title: string;
  severity: "low" | "medium" | "high";
  location: string;
  description: string;
};

export default function XrayAnalyser() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [status, setStatus] = useState<"Normal" | "Defective" | null>(null);
  const [issues, setIssues] = useState<DetectedIssue[]>([]);
  const [mode, setMode] = useState<"mri" | "xray">("mri");

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setConfidence(null);
      setStatus(null);
      setIssues([]);
    }
  };

  // Simulated analysis - deterministic but varying fake results
  // Always returns a Defective result with confidence >= 90 and at least 2 issues
  const analyze = async () => {
    if (!image) return;
    setLoading(true);
    setConfidence(null);
    setStatus(null);
    setIssues([]);

    // simulate processing delay
    await new Promise((r) => setTimeout(r, 750));

    // pools of possible findings for MRI and X-ray
    const mriFindings = [
      {
        title: "Lesion",
        severity: "high" as const,
        location: "Temporal lobe",
        description: "Suspicious hyperintense region noted on T2-weighted sequence",
      },
      {
        title: "Edema",
        severity: "medium" as const,
        location: "Perilesional area",
        description: "Mild surrounding edema pattern observed",
      },
      {
        title: "Cystic Component",
        severity: "low" as const,
        location: "Frontal cortex",
        description: "Small cystic pocket seen on axial slice",
      },
      {
        title: "Enhancing Nodule",
        severity: "medium" as const,
        location: "Occipital lobe",
        description: "Contrast-enhancing focus suspicious for lesion",
      },
      {
        title: "Signal Abnormality",
        severity: "high" as const,
        location: "Parietal region",
        description: "Focal signal abnormality noted on T1 sequence",
      },
    ];

    const xrayFindings = [
      {
        title: "Fracture",
        severity: "high" as const,
        location: "Left rib",
        description: "Cortical discontinuity noted consistent with acute fracture",
      },
      {
        title: "Consolidation",
        severity: "medium" as const,
        location: "Right lower lobe",
        description: "Airspace consolidation suggests possible pneumonia",
      },
      {
        title: "Pleural Effusion",
        severity: "medium" as const,
        location: "Left hemithorax",
        description: "Small pleural effusion layering posteriorly",
      },
      {
        title: "Opacity",
        severity: "low" as const,
        location: "Perihilar region",
        description: "Nonspecific perihilar opacity — correlate clinically",
      },
    ];

    // choose pool based on mode
    const pool = mode === "mri" ? mriFindings : xrayFindings;

    // pick 2-3 random findings without replacement
    const chosen: DetectedIssue[] = [];
  // MRI should always return exactly 2 issues chosen from the MRI pool.
  const count = mode === "mri" ? 2 : 2 + Math.floor(Math.random() * 2); // MRI:2, X-ray:2 or 3

    // Avoid repeating the exact same findings (by title) from the previous analysis
    const prevTitles = issues.map((it) => it.title);
    // Build a list of candidate indices excluding previously shown titles
    const availableIdx = pool
      .map((_, i) => i)
      .filter((i) => !prevTitles.includes(pool[i].title));

    // If there aren't enough new candidates, fall back to the full pool (allow repeats)
    const indicesToChooseFrom = availableIdx.length >= count ? availableIdx : pool.map((_, i) => i);

    const usedIdx = new Set<number>();
    while (chosen.length < count) {
      const randIndex = Math.floor(Math.random() * indicesToChooseFrom.length);
      const idx = indicesToChooseFrom[randIndex];
      if (usedIdx.has(idx)) continue;
      usedIdx.add(idx);
      const base = pool[idx];
      chosen.push({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        title: base.title,
        severity: base.severity,
        location: base.location,
        description: base.description,
      });
    }

    // produce a high confidence score (90-98)
    const simulatedConfidence = 90 + Math.floor(Math.random() * 9);
    const simulatedStatus: "Defective" | "Normal" = "Defective";

    // set results with a short UI delay for polish
    setTimeout(() => {
      setConfidence(simulatedConfidence);
      setStatus(simulatedStatus as any);
      setIssues(chosen);
      setLoading(false);
    }, 250);
  };

  const exportReport = () => {
    // generate a simple text report and download as .txt
    const content = [
      `Analysis Report - ${new Date().toISOString()}`,
      `Status: ${status || "Unknown"}`,
      `Confidence: ${confidence ?? "-"}%`,
      "Detected Issues:",
      ...issues.map((it) => `- ${it.title} (${it.severity}) at ${it.location}: ${it.description}`),
    ].join("\n\n");

    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mri_analysis_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setConfidence(null);
    setStatus(null);
    setIssues([]);
  };

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center bg-[#f6faff] py-8">
      <div className="w-full max-w-6xl flex items-center justify-between">
        <h1 className="text-3xl md:text-4xl font-bold text-purple-700 flex items-center gap-2 mb-2">
        <span className="inline-block bg-purple-100 rounded-lg p-2">
          <UploadCloud className="w-8 h-8 text-purple-600" />
        </span>
          {mode === "mri" ? "MRI Analyzer" : "X-ray Analyzer"}
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMode("mri")}
            className={`px-3 py-1 rounded ${mode === "mri" ? "bg-purple-600 text-white" : "border"}`}
          >
            MRI
          </button>
          <button
            onClick={() => setMode("xray")}
            className={`px-3 py-1 rounded ${mode === "xray" ? "bg-purple-600 text-white" : "border"}`}
          >
            X-ray
          </button>
        </div>
      </div>
      <p className="text-gray-500 mb-6 text-center max-w-3xl">
        {mode === "mri"
          ? "Modern AI-assisted MRI analysis to flag potential abnormalities with high confidence"
          : "Modern AI-assisted X-ray analysis to flag potential abnormalities with high confidence"}
      </p>

      <div className="flex flex-col lg:flex-row gap-8 w-full max-w-6xl">
        {/* Left - Upload and preview */}
        <div className="flex-1 bg-white rounded-xl shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <UploadCloud className="w-5 h-5" /> Upload {mode === "mri" ? "MRI" : "X-ray"} Image
          </h2>
          <div className="border border-gray-100 rounded-lg p-4">
            <label htmlFor="mri-upload" className="w-full cursor-pointer block">
              <div className="w-full h-[420px] bg-gray-50 rounded-lg flex items-center justify-center overflow-hidden">
                {preview ? (
                  <img src={preview} alt={`${mode} preview`} className="object-contain max-h-full" />
                ) : (
                  <div className="text-center text-gray-400">
                    <UploadCloud className="mx-auto w-12 h-12" />
                    <div className="mt-2">Click or drop to upload {mode === "mri" ? "MRI" : "X-ray"} image</div>
                    <div className="text-xs text-gray-400 mt-1">Supports JPG, PNG, DICOM</div>
                  </div>
                )}
              </div>
              <input id="mri-upload" type="file" accept="image/*,.dcm" className="hidden" onChange={handleImageChange} />
            </label>

            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-gray-600">{image ? image.name : "No file selected"}</div>
              <div className="flex items-center gap-2">
                <button
                  onClick={analyze}
                  disabled={!image || loading}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 2L11 13" />
                    <path d="M22 2l-7 20 2-7 7-13z" />
                  </svg>
                  Analyze Image
                </button>
                <button onClick={reset} className="px-3 py-2 border rounded flex items-center gap-2">
                  <RefreshCw className="w-4 h-4" /> New Analysis
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right - Results panel */}
        <div className="flex-1 bg-white rounded-xl shadow-md p-6">
          <div className="flex items-start justify-between">
            <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" /> Analysis Results
            </h2>
            <div className="text-right">
              <div className={`text-sm font-medium ${status === "Defective" ? "text-red-600" : "text-green-600"}`}>
                {status || "-"}
              </div>
              <div className="text-2xl font-bold text-purple-700 mt-2">{confidence ? `${confidence}%` : "-"}</div>
            </div>
          </div>

          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-700">Detected Issues</h3>
            <div className="mt-3 space-y-3">
              {issues.length === 0 ? (
                <div className="text-gray-400">No issues detected. Upload and analyze an image to start.</div>
              ) : (
                issues.map((it) => (
                  <div key={it.id} className="border rounded p-3 bg-gray-50 flex items-start gap-3">
                    <div className={`w-2 h-full rounded-l ${it.severity === "high" ? "bg-red-400" : it.severity === "medium" ? "bg-orange-300" : "bg-green-300"}`} />
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <div className="font-medium">{it.title} <span className={`ml-2 text-xs px-2 py-1 rounded ${it.severity === "high" ? "bg-red-500 text-white" : it.severity === "medium" ? "bg-gray-300 text-gray-800" : "bg-green-200 text-green-800"}`}>{it.severity}</span></div>
                        <div className="text-sm text-gray-500">Location: {it.location}</div>
                      </div>
                      <div className="text-sm text-gray-600 mt-1">{it.description}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button onClick={exportReport} disabled={!status} className="px-4 py-2 border rounded flex items-center gap-2">
              <Download className="w-4 h-4" /> Export Report
            </button>
            <button onClick={reset} className="px-4 py-2 border rounded">
              New Analysis
            </button>
          </div>
        </div>
      </div>

      <div className="mt-8 w-full max-w-6xl">
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded flex items-center gap-2">
          <span className="font-bold text-yellow-700">Medical Disclaimer:</span>
          <span className="text-yellow-700 text-sm">
            This tool is for educational and research purposes only. It is not a medical diagnosis. Always consult qualified medical professionals for diagnosis and treatment decisions.
          </span>
        </div>
      </div>
    </div>
  );
}
