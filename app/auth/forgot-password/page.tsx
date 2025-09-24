"use client";
import { useState } from "react";

export default function ForgotPassword() {
  const [gmail, setGmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [step, setStep] = useState(1);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleRequestReset(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
  const res = await fetch("/api/auth/forgot-password/request", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gmail })
    });
    const data = await res.json();
    if (data.success) {
      setStep(2);
      setMessage("An OTP has been sent to your Gmail.");
    } else {
      setError(data.error || "Request failed");
    }
  }

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
  const res = await fetch("/api/auth/forgot-password/reset", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gmail, otp, newPassword })
    });
    const data = await res.json();
    if (data.success) {
      setMessage("Password reset successful! You can now log in.");
      setStep(1);
      setGmail("");
      setOtp("");
      setNewPassword("");
    } else {
      setError(data.error || "Reset failed");
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-purple-800 to-blue-900">
      <form onSubmit={step === 1 ? handleRequestReset : handleResetPassword} className="w-full max-w-md bg-white rounded-xl shadow-lg p-8 relative">
        <h2 className="text-2xl font-bold text-center mb-6">Forgot Password</h2>
        {error && <div className="text-red-600 text-center mb-2">{error}</div>}
        {message && <div className="text-green-600 text-center mb-2">{message}</div>}
        <div className="mb-4">
          <input name="gmail" type="email" required placeholder="Enter your Gmail address" className="w-full outline-none border-b border-gray-300 py-2" value={gmail} onChange={e => setGmail(e.target.value)} />
        </div>
        {step === 2 && (
          <>
            <div className="mb-4">
              <input name="otp" type="text" required placeholder="Enter OTP" className="w-full outline-none border-b border-gray-300 py-2" value={otp} onChange={e => setOtp(e.target.value)} />
            </div>
            <div className="mb-4">
              <input name="newPassword" type="password" required placeholder="Enter new password" className="w-full outline-none border-b border-gray-300 py-2" value={newPassword} onChange={e => setNewPassword(e.target.value)} />
            </div>
          </>
        )}
        <button type="submit" className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 rounded transition">
          {step === 1 ? "Request Reset" : "Reset Password"}
        </button>
      </form>
    </div>
  );
}
