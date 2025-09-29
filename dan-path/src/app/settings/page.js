"use client"
import React, { useState } from "react";

// --- Icons (Using Lucide-React inspired SVGs) ---
const SettingsIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);
const UserIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);
const ZapIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);
const LockIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);
const ChevronRight = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <polyline points="9 18 15 12 9 6" />
  </svg>
);
const BellIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
  </svg>
);

// --- Component for a single setting option ---
const SettingOption = ({ title, description, icon: Icon, onClick, isAccent = false }) => (
  <div 
    className="flex items-center justify-between p-4 rounded-xl cursor-pointer transition hover:bg-neutral-800/70"
    onClick={onClick}
  >
    <div className="flex items-center space-x-4">
      <div className={`p-2 rounded-full ${isAccent ? 'bg-indigo-600' : 'bg-neutral-700'}`}>
        <Icon className={`w-5 h-5 ${isAccent ? 'text-white' : 'text-gray-300'}`} />
      </div>
      <div>
        <p className={`font-semibold ${isAccent ? 'text-indigo-300' : 'text-white'}`}>{title}</p>
        <p className="text-sm text-gray-400">{description}</p>
      </div>
    </div>
    <ChevronRight className="w-5 h-5 text-gray-500" />
  </div>
);

// --- Component for a toggle switch setting ---
const ToggleSetting = ({ title, description, isChecked, onToggle }) => (
  <div className="flex items-center justify-between p-4 transition hover:bg-neutral-800/70 rounded-xl">
    <div className="flex-1">
      <p className="font-semibold text-white">{title}</p>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
    <label className="relative inline-flex items-center cursor-pointer ml-4">
      <input type="checkbox" value="" className="sr-only peer" checked={isChecked} onChange={onToggle} />
      <div 
        className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300/50 rounded-full peer 
        peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
        after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"
      ></div>
    </label>
  </div>
);


const SettingsPage = () => {
  // --- State for toggle settings ---
  const [darkTheme, setDarkTheme] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [privateProfile, setPrivateProfile] = useState(false);

  // Helper function for demo console logging
  const handleOptionClick = (title) => {
    console.log(`Navigating to ${title} details...`);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans p-4 sm:p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white flex items-center mb-8">
          <SettingsIcon className="w-8 h-8 mr-3 text-indigo-400" />
          Settings
        </h1>
        
        {/* --- Settings Container (Card) --- */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl shadow-xl space-y-8 p-4 sm:p-8">

          {/* --- 1. Account Settings Section --- */}
          <div>
            <h2 className="text-2xl font-semibold text-indigo-400 border-b border-indigo-500/30 pb-2 mb-4">Account</h2>
            <div className="space-y-2">
              <SettingOption
                title="Edit Profile"
                description="Change your name, bio, and profile picture."
                icon={UserIcon}
                onClick={() => handleOptionClick("Edit Profile")}
                isAccent={true}
              />
              <SettingOption
                title="Change Password"
                description="Update your current password securely."
                icon={LockIcon}
                onClick={() => handleOptionClick("Change Password")}
              />
              <SettingOption
                title="Linked Accounts"
                description="Manage connections to Google, Apple, etc."
                icon={ZapIcon}
                onClick={() => handleOptionClick("Linked Accounts")}
              />
            </div>
          </div>

          {/* --- 2. Notifications Settings Section --- */}
          <div>
            <h2 className="text-2xl font-semibold text-indigo-400 border-b border-indigo-500/30 pb-2 mb-4">Notifications</h2>
            <div className="space-y-2">
              <ToggleSetting
                title="Email Notifications"
                description="Receive updates about activity and news via email."
                isChecked={emailNotifications}
                onToggle={() => setEmailNotifications(!emailNotifications)}
              />
              <ToggleSetting
                title="Push Notifications"
                description="Get instant alerts for new messages and likes."
                isChecked={true} // Mocked as always true for demo
                onToggle={() => console.log("Toggle Push Notifications")}
              />
              <SettingOption
                title="Notification Preferences"
                description="Customize notification sounds and types."
                icon={BellIcon}
                onClick={() => handleOptionClick("Notification Preferences")}
              />
            </div>
          </div>

          {/* --- 3. Privacy Settings Section --- */}
          <div>
            <h2 className="text-2xl font-semibold text-indigo-400 border-b border-indigo-500/30 pb-2 mb-4">Privacy & Security</h2>
            <div className="space-y-2">
              <ToggleSetting
                title="Private Profile"
                description="Require approval for new followers."
                isChecked={privateProfile}
                onToggle={() => setPrivateProfile(!privateProfile)}
              />
              <SettingOption
                title="Blocked Users"
                description="Manage the list of users you have blocked."
                icon={LockIcon}
                onClick={() => handleOptionClick("Blocked Users")}
              />
            </div>
          </div>
          
          {/* --- 4. Danger Zone (Sign Out) --- */}
          <div className="pt-4 border-t border-neutral-800">
            <button
              className="w-full py-3 px-4 bg-red-600 rounded-xl font-bold hover:bg-red-700 transition shadow-md shadow-red-500/20"
              onClick={() => handleOptionClick("Sign Out")}
            >
              Sign Out
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
