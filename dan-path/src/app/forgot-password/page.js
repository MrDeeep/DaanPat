"use client"
import React, { useState } from "react";

// The component is designed to reuse the dark background and centered card style
const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (email) {
      // Mock password reset logic
      setMessage(
        `A password reset link has been sent to ${email}. Please check your inbox.`
      );
      console.log(`Password reset requested for: ${email}`);
    } else {
      setMessage("Please enter a valid email address.");
    }
  };

  // Helper function to handle links using standard <a> tags (for demonstration)
  const NavLink = ({ href, children, className }) => (
    <a href={href} className={className}>
      {children}
    </a>
  );

  return (
    // Outer Container: min-h-screen and flex centering
    <div
      className="flex items-center justify-center min-h-screen w-full p-4 
                   bg-neutral-950 relative overflow-hidden"
    >
      {/* Background Glow Effect (similar to login page) */}
      <div className="absolute top-0 left-0 w-80 h-80 bg-indigo-500/30 rounded-full blur-3xl opacity-50 animate-pulse-slow"></div>
      <div className="absolute bottom-0 right-0 w-64 h-64 bg-fuchsia-500/30 rounded-full blur-3xl opacity-50 animate-pulse-slow delay-1000"></div>

      <style>{`
        /* Minimal custom animation for the background glow */
        @keyframes pulse-slow {
          0%, 100% { transform: translate(0, 0); }
          50% { transform: translate(20px, 20px); }
        }
        .animate-pulse-slow {
          animation: pulse-slow 15s infinite alternate;
        }
      `}</style>

      {/* Forgot Password Card Container */}
      <div
        className="w-full max-w-md p-8 space-y-6 
                   bg-white/5 backdrop-blur-lg rounded-xl shadow-2xl 
                   border border-white/10 text-white z-10"
      >
        <h2 className="text-3xl font-extrabold text-center text-white">
          Reset Password
        </h2>
        <p className="text-center text-gray-300">
          Enter your email address below to receive a password reset link.
        </p>

        {/* Dynamic Message Display */}
        {message && (
          <div className={`p-3 rounded-lg text-sm ${message.includes("sent") ? "bg-green-600/30 text-green-300" : "bg-red-600/30 text-red-300"}`}>
            {message}
          </div>
        )}

        <form className="space-y-6" onSubmit={handleSubmit}>
          {/* Email Input Field */}
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-300"
            >
              Email address
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="you@example.com"
            />
          </div>

          {/* Reset Button */}
          <div>
            <button
              type="submit"
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150"
            >
              Send Reset Link
            </button>
          </div>
        </form>

        {/* Back to Login Link */}
        <div className="text-center text-sm text-gray-400">
          Remember your password?{" "}
          <NavLink
            href="/log-in" // Assuming this is your login page route
            className="font-medium text-indigo-400 hover:text-indigo-300 transition"
          >
            Sign In
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
