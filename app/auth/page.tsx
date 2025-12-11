"use client";

import { useState, useRef } from "react";
import { Eye, EyeOff } from "lucide-react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";

export default function AuthPage() {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [gmail, setGmail] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const gmailRef = useRef<HTMLInputElement | null>(null);
  const router = useRouter();

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const res = await signIn("credentials", {
      redirect: false,
      username,
      password,
    });
    setLoading(false);
    if (res && !res.error) {
      // Redirect to the app's main page after successful login
      window.location.href = "/";
    } else {
      setError("Invalid credentials");
    }
  }

  async function handleSignUp(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, gmail })
    });
    let data: any = {};
    try {
      data = await res.json();
    } catch {
      setLoading(false);
      setError("Unexpected server error. Please try again.");
      return;
    }
    setLoading(false);
    if (data.success) {
      setMode('signin');
      setUsername("");
      setPassword("");
      setError("");
      alert("Signup successful! Please log in.");
    } else {
      // If the backend reports the gmail already exists, focus the gmail input so user can correct it
      if (data.error === "gmail exists please login") {
        setError(data.error);
        gmailRef.current?.focus();
      } else {
        setError(data.error || "Signup failed");
      }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#3a2066] to-[#1e085a] dark:from-gray-900 dark:to-black flex items-center justify-center transition-colors duration-300">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-8 w-full max-w-md relative transition-colors duration-300">
        <button className="absolute right-4 top-4 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 text-2xl font-bold transition-colors">&times;</button>
        <h2 className="text-2xl font-bold text-center mb-6 text-gray-900 dark:text-white transition-colors">{mode === 'signin' ? 'Login' : 'Sign Up'}</h2>
        {error && <div className="text-red-600 dark:text-red-400 text-center mb-2 transition-colors">{error}</div>}
        {mode === 'signin' ? (
          <form onSubmit={handleSignIn}>
            <div className="mb-4 flex items-center border-b border-gray-300 dark:border-gray-600 py-2 transition-colors">
              <span className="text-gray-400 dark:text-gray-500 mr-2 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 20.25v-1.5A2.25 2.25 0 016.75 16.5h10.5a2.25 2.25 0 012.25 2.25v1.5" />
                </svg>
              </span>
              <input
                name="username"
                type="text"
                required
                placeholder="Enter your email"
                className="w-full outline-none border-none bg-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors"
                value={username}
                onChange={e => setUsername(e.target.value)}
              />
            </div>
            <div className="mb-4 flex items-center border-b border-gray-300 py-2">
              <span className="text-gray-400 mr-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75A4.5 4.5 0 008 6.75v3.75m8.25 0a2.25 2.25 0 01-4.5 0m4.5 0h-4.5m-2.25 0a2.25 2.25 0 01-4.5 0m4.5 0h-4.5" />
                </svg>
              </span>
              <div className="relative w-full">
                <input
                  name="password"
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="Enter you password"
                  className="w-full outline-none border-none bg-transparent pr-10"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  tabIndex={-1}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between mb-4 text-sm">
              <label className="flex items-center">
                <input type="checkbox" className="mr-1" /> Remember me
              </label>
              <a href="/auth/forgot-password" className="text-purple-700 hover:underline">Forgot password?</a>
            </div>
            <button
              type="submit"
              className="w-full bg-[#7c3aed] hover:bg-[#6d28d9] text-white font-semibold py-2 rounded transition mb-2 mt-2"
              disabled={loading}
            >
              {loading ? 'Logging In...' : 'Login Now'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleSignUp}>
            <div className="mb-4 flex items-center border-b border-gray-300 py-2">
              <span className="text-gray-400 mr-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 20.25v-1.5A2.25 2.25 0 016.75 16.5h10.5a2.25 2.25 0 012.25 2.25v1.5" />
                </svg>
              </span>
              <input
                name="username"
                type="text"
                required
                placeholder="Enter your username"
                className="w-full outline-none border-none bg-transparent"
                value={username}
                onChange={e => setUsername(e.target.value)}
              />
            </div>
            <div className="mb-4 flex items-center border-b border-gray-300 py-2">
              <span className="text-gray-400 mr-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75A4.5 4.5 0 008 6.75v3.75m8.25 0a2.25 2.25 0 01-4.5 0m4.5 0h-4.5m-2.25 0a2.25 2.25 0 01-4.5 0m4.5 0h-4.5" />
                </svg>
              </span>
              <div className="relative w-full">
                <input
                  name="password"
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="Enter your password"
                  className="w-full outline-none border-none bg-transparent pr-10"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  tabIndex={-1}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
            <div className="mb-4 flex items-center border-b border-gray-300 py-2">
              <span className="text-gray-400 mr-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 7.5v9a2.25 2.25 0 01-2.25 2.25h-15A2.25 2.25 0 012.25 16.5v-9A2.25 2.25 0 014.5 5.25h15A2.25 2.25 0 0121.75 7.5z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 7.5l9.75 6.75L21.75 7.5" />
                </svg>
              </span>
              <input
                name="gmail"
                type="email"
                required
                placeholder="Enter your Gmail address"
                ref={gmailRef}
                className="w-full outline-none border-none bg-transparent"
                value={gmail}
                onChange={e => setGmail(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="w-full bg-[#7c3aed] hover:bg-[#6d28d9] text-white font-semibold py-2 rounded transition mb-2 mt-2"
              disabled={loading}
            >
              {loading ? 'Signing Up...' : 'Sign Up Now'}
            </button>
          </form>
        )}
        <div className="text-center mt-4 text-sm">
          {mode === 'signin' ? (
            <>
              Not a member?{' '}
              <button className="text-purple-700 hover:underline" onClick={() => { setMode('signup'); setError(""); }}>Signup Now</button>
            </>
          ) : (
            <>
              Already a member?{' '}
              <button className="text-purple-700 hover:underline" onClick={() => { setMode('signin'); setError(""); }}>Sign In</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
