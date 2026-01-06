"use client";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useRouter } from "next/navigation";

export default function ForgotPassword() {
  const router = useRouter();
  const [gmail, setGmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [step, setStep] = useState(1);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resetSuccess, setResetSuccess] = useState(false);

  async function handleRequestReset(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
    setLoading(true);
    try {
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
    } finally {
      setLoading(false);
    }
  }

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/forgot-password/reset", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gmail, otp, newPassword })
      });
      const data = await res.json();
      if (data.success) {
        setMessage("Password reset successful!");
        setResetSuccess(true);
        setStep(1);
        setGmail("");
        setOtp("");
        setNewPassword("");
      } else {
        setError(data.error || "Reset failed");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-purple-800 to-blue-900 dark:from-gray-900 dark:to-black transition-colors duration-300">
      <div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 relative transition-colors duration-300">
        <h2 className="text-2xl font-bold text-center mb-6 text-gray-900 dark:text-white transition-colors">Forgot Password</h2>
        {error && <div className="text-red-600 dark:text-red-400 text-center mb-2 transition-colors">{error}</div>}
        {message && <div className="text-green-600 dark:text-green-400 text-center mb-2 transition-colors">{message}</div>}
        
        {resetSuccess ? (
          <button
            onClick={() => router.push("/auth/signin")}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 rounded transition"
          >
            Back to Login
          </button>
        ) : (
          <form onSubmit={step === 1 ? handleRequestReset : handleResetPassword}>
            <div className="mb-4">
              <input 
                name="gmail" 
                type="email" 
                required 
                placeholder="Enter your Gmail address" 
                className="w-full outline-none border-b border-gray-300 dark:border-gray-600 py-2 bg-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors" 
                value={gmail} 
                onChange={e => setGmail(e.target.value)} 
              />
            </div>
            {step === 2 && (
              <>
                <div className="mb-4">
                  <input 
                    name="otp" 
                    type="text" 
                    required 
                    placeholder="Enter OTP" 
                    className="w-full outline-none border-b border-gray-300 dark:border-gray-600 py-2 bg-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors" 
                    value={otp} 
                    onChange={e => setOtp(e.target.value)} 
                  />
                </div>
                <div className="mb-4 relative">
                  <input
                    name="newPassword"
                    type={showNewPassword ? "text" : "password"}
                    required
                    placeholder="Enter new password"
                    className="w-full outline-none border-b border-gray-300 dark:border-gray-600 py-2 pr-10 bg-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors"
                    value={newPassword}
                    onChange={e => setNewPassword(e.target.value)}
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    onClick={() => setShowNewPassword((v) => !v)}
                    aria-label={showNewPassword ? "Hide password" : "Show password"}
                  >
                    {showNewPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </>
            )}
            <button
              type="submit"
              className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 rounded transition disabled:opacity-60"
              disabled={loading}
            >
              {loading ? (step === 1 ? "Requesting..." : "Resetting...") : (step === 1 ? "Request Reset" : "Reset Password")}
            </button>
          </form>
        )}
        
        {!resetSuccess && (
          <div className="mt-4 text-center">
            <button
              onClick={() => router.push("/auth")}
              className="text-purple-600 dark:text-purple-400 hover:underline text-sm"
            >
              Back to Login
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
