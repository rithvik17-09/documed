"use client";

import React, { useState } from "react";
import { UploadCloud, CheckCircle } from "lucide-react";

export default function XrayAnalyser() {
  const [image, setImage] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setResult("");
    }
  };

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center bg-[#f6faff] py-8">
      <h1 className="text-4xl font-bold text-blue-600 flex items-center gap-2 mb-2">
        <span className="inline-block bg-blue-100 rounded-lg p-2">
          <UploadCloud className="w-8 h-8 text-blue-500" />
        </span>
        X-Ray Defect Analyzer
      </h1>
      <p className="text-gray-500 mb-8 text-center max-w-xl">
        Advanced AI-powered X-ray analysis for rapid defect detection and diagnostic assistance
      </p>
      <div className="flex flex-col md:flex-row gap-8 w-full max-w-4xl">
        {/* Upload Section */}
        <div className="flex-1 bg-white rounded-xl shadow-md p-6 flex flex-col items-center">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <UploadCloud className="w-5 h-5" /> Upload X-Ray Image
          </h2>
          <label htmlFor="xray-upload" className="w-full h-48 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 transition mb-2">
            {image ? (
              <img src={URL.createObjectURL(image)} alt="X-ray preview" className="max-h-40 object-contain" />
            ) : (
              <>
                <UploadCloud className="w-10 h-10 text-gray-400 mb-2" />
                <span className="text-gray-600">Drop your X-ray image here</span>
                <span className="text-xs text-gray-400">or click to browse files</span>
                <span className="text-xs text-gray-500 mt-2">Supports JPG, PNG, DICOM</span>
              </>
            )}
            <input
              id="xray-upload"
              type="file"
              accept="image/png, image/jpeg, image/jpg, .dcm"
              className="hidden"
              onChange={handleImageChange}
            />
          </label>
        </div>
        {/* Analysis Results Section */}
        <div className="flex-1 bg-white rounded-xl shadow-md p-6 flex flex-col items-center justify-center">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-500" /> Analysis Results
          </h2>
          <div className="flex flex-col items-center justify-center h-48 w-full">
            {image ? (
              <div className="flex flex-col items-center">
                <span className="text-gray-400 mb-2">(AI analysis coming soon)</span>
                <img src={URL.createObjectURL(image)} alt="X-ray preview" className="max-h-32 object-contain mb-2" />
                <span className="text-gray-600">Upload and analyze an X-ray to see results</span>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full w-full">
                <span className="text-gray-400 mb-2">
                  <UploadCloud className="w-10 h-10" />
                </span>
                <span className="text-gray-500">Upload and analyze an X-ray to see results</span>
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="mt-8 w-full max-w-4xl">
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded flex items-center gap-2">
          <span className="font-bold text-yellow-700">Medical Disclaimer:</span>
          <span className="text-yellow-700 text-sm">
            This tool is for educational and research purposes only. Always consult qualified medical professionals for diagnosis and treatment decisions.
          </span>
        </div>
      </div>
    </div>
  );
}
