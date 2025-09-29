"use client";
// import Link from "next/link"; // Removed to fix the import error
import React from "react";
import Link from "next/link";

const LogIn = () => {
  const handleSubmit = (e) => {
    e.preventDefault();
    // Replaced alert with console log to adhere to best practices
    console.log("Sign-in attempt submitted! Data capture logic goes here.");
  };

  // Helper function to handle links using standard <a> tags
  const NavLink = ({ href, children, className }) => (
    <a href={href} className={className}>
      {children}
    </a>
  );

  return (
    // Outer Container: min-h-screen and flex centering ensures content is vertically centered on any screen size.
    <div
      className="flex items-center justify-center min-h-screen w-full p-4 
                    bg-neutral-950 
                    bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))]"
    >
      {/* Sign-In Card Container: w-full makes it fluid on mobile; max-w-md prevents excessive width on desktop. */}
      <div
        className="w-full max-w-md p-8 space-y-6 
                    bg-white/5 backdrop-blur-md rounded-xl shadow-2xl 
                    border border-white/10 text-white z-10"
      >
        <h2 className="text-3xl font-extrabold text-center text-white">
          Welcome Back
        </h2>
        <p className="text-center text-gray-300">
          Sign in to your account to continue.
        </p>

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
              // Input width is w-full to fill its parent container
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Email"
            />
          </div>

          {/* Password Input Field */}
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-300"
            >
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Password"
            />
          </div>

          {/* Forgot Password Link */}
          <div className="flex items-center justify-between">
            <div className="text-sm">
              <Link
                href="/forgot-password"
                className="font-medium text-indigo-400 hover:text-indigo-300 transition"
              >
                Forgot your password?
              </Link>
            </div>
          </div>

          {/* Sign In Button: w-full ensures button spans the card width */}
          <div>
            <button
              type="submit"
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150"
            >
              Sign In
            </button>
          </div>
        </form>

        {/* Sign Up Prompt: Using NavLink component */}
        <div className="text-center text-sm text-gray-400">
          Dont have an account?{" "}
          <NavLink
            href="/sign-up"
            className="font-medium text-indigo-400 hover:text-indigo-300 transition"
          >
            Sign up
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default LogIn;
