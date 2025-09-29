"use client";
import React, { useState } from "react";
import Link from "next/link";

const SignUp = () => {
  const [userType, setUserType] = useState("normal"); // 'normal' or 'ngo'
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    organizationName: "",
    ngoRegistrationId: "",
    ngoAddress: "", // ✅ New field
  });
  const [passwordError, setPasswordError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleUserTypeChange = (type) => {
    setUserType(type);
    if (type === "normal") {
      setFormData((prevData) => ({
        ...prevData,
        organizationName: "",
        ngoRegistrationId: "",
        ngoAddress: "",
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setPasswordError("");

    if (formData.password !== formData.confirmPassword) {
      setPasswordError("Passwords do not match.");
      return;
    }

    console.log("Sign Up Data:", { userType, ...formData });
    console.log(`Sign Up successful as a ${userType} user!`);
  };

  return (
    <div className="flex items-center justify-center min-h-screen w-full p-4 
                    bg-neutral-950 
                    bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))]">
      <div className="w-full max-w-lg p-8 space-y-6 
                    bg-white/5 backdrop-blur-md rounded-xl shadow-2xl 
                    border border-white/10 text-white z-10">
        
        <h2 className="text-3xl font-extrabold text-center text-white">
          Create Your Account
        </h2>
        <p className="text-center text-gray-300">
          Join us by signing up as a normal user or an NGO.
        </p>

        {/* User Type Selection */}
        <div className="flex justify-center space-x-4 mb-6">
          <button
            type="button"
            onClick={() => handleUserTypeChange("normal")}
            className={`px-6 py-2 rounded-lg text-sm font-medium transition duration-200
              ${userType === "normal"
                ? "bg-indigo-600 text-white shadow-lg"
                : "bg-white/10 text-gray-300 hover:bg-white/20"
              }`}
          >
            Normal User
          </button>
          <button
            type="button"
            onClick={() => handleUserTypeChange("ngo")}
            className={`px-6 py-2 rounded-lg text-sm font-medium transition duration-200
              ${userType === "ngo"
                ? "bg-indigo-600 text-white shadow-lg"
                : "bg-white/10 text-gray-300 hover:bg-white/20"
              }`}
          >
            NGO / Organization
          </button>
        </div>

        <form className="space-y-6" onSubmit={handleSubmit}>
          {/* Common Fields */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-300">
              Full Name
            </label>
            <input
              id="name"
              name="name"
              type="text"
              required
              value={formData.name}
              onChange={handleChange}
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Name"
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-300">
              Email address
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              value={formData.email}
              onChange={handleChange}
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Email"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              value={formData.password}
              onChange={handleChange}
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Password"
            />
          </div>

          <div>
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300">
              Confirm Password
            </label>
            <input
              id="confirmPassword"
              name="confirmPassword"
              type="password"
              required
              value={formData.confirmPassword}
              onChange={handleChange}
              className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
              placeholder="Confirm Password"
            />
            {passwordError && (
              <p className="mt-2 text-sm text-red-400">{passwordError}</p>
            )}
          </div>

          {/* NGO Specific Fields */}
          {userType === "ngo" && (
            <>
              <div className="border-t border-white/10 pt-6">
                <h3 className="text-xl font-bold text-white mb-4">NGO Details</h3>
                
                <div>
                  <label htmlFor="organizationName" className="block text-sm font-medium text-gray-300">
                    Organization Name
                  </label>
                  <input
                    id="organizationName"
                    name="organizationName"
                    type="text"
                    required={userType === "ngo"}
                    value={formData.organizationName}
                    onChange={handleChange}
                    className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
                    placeholder="Your Organization's Name"
                  />
                </div>

                <div className="mt-4">
                  <label htmlFor="ngoRegistrationId" className="block text-sm font-medium text-gray-300">
                    NGO Registration ID
                  </label>
                  <input
                    id="ngoRegistrationId"
                    name="ngoRegistrationId"
                    type="text"
                    required={userType === "ngo"}
                    value={formData.ngoRegistrationId}
                    onChange={handleChange}
                    className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
                    placeholder="e.g., Reg12345"
                  />
                </div>

                {/* ✅ NGO Address Field */}
                <div className="mt-4">
                  <label htmlFor="ngoAddress" className="block text-sm font-medium text-gray-300">
                    NGO Address
                  </label>
                  <textarea
                    id="ngoAddress"
                    name="ngoAddress"
                    rows="3"
                    required={userType === "ngo"}
                    value={formData.ngoAddress}
                    onChange={handleChange}
                    className="w-full mt-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 placeholder-gray-400 text-white outline-none transition"
                    placeholder="Enter your NGO's full address"
                  ></textarea>
                </div>
              </div>
            </>
          )}

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150"
            >
              Sign Up
            </button>
          </div>
        </form>

        {/* Login Prompt */}
        <div className="text-center text-sm text-gray-400">
          Already have an account?{" "}
          <Link href="/log-in" className="font-medium text-indigo-400 hover:text-indigo-300 transition">
            Log in
          </Link>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
