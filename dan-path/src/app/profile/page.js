"use client"

import React, { useState } from "react";

// =================================================================
// 1. ICONS
// =================================================================
const UserIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);
const GridIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <line x1="12" y1="3" x2="12" y2="21" />
    <line x1="3" y1="12" x2="21" y2="12" />
  </svg>
);
const HeartIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);

// =================================================================
// 2. MOCK DATA
// =================================================================
const mockUser = {
  name: "Jane Doe",
  handle: "@janedoe_dev",
  avatar: "https://i.pravatar.cc/150?u=janedoe_dev",
  banner: "https://placehold.co/1000x200/1e293b/94a3b8?text=Profile+Banner",
  bio: "Frontend Developer | React enthusiast | Building beautiful and responsive UIs with a passion for clean code and great user experience.",
  stats: {
    posts: 158,
    followers: "12.5K",
    following: 305,
  },
};

const mockPosts = [
  { id: 1, type: 'image', likes: 231, comments: 42, url: 'https://placehold.co/300x300/4f46e5/ffffff?text=React+Grid' },
  { id: 2, type: 'video', likes: 502, comments: 88, url: 'https://placehold.co/300x300/10b981/ffffff?text=Tailwind+CSS' },
  { id: 3, type: 'image', likes: 98, comments: 15, url: 'https://placehold.co/300x300/f97316/ffffff?text=State+Management' },
  { id: 4, type: 'image', likes: 345, comments: 60, url: 'https://placehold.co/300x300/3b82f6/ffffff?text=Dark+Mode+Design' },
  { id: 5, type: 'video', likes: 121, comments: 20, url: 'https://placehold.co/300x300/eab308/ffffff?text=Component+Tree' },
  { id: 6, type: 'image', likes: 678, comments: 102, url: 'https://placehold.co/300x300/ef4444/ffffff?text=UI%2FUX+Tips' },
];

// =================================================================
// 3. POST CARD
// =================================================================
const PostCard = ({ post }) => (
  <div 
    className="relative group aspect-square bg-neutral-800 rounded-xl overflow-hidden 
               cursor-pointer hover:scale-[1.03] transition-transform duration-300 shadow-lg"
  >
    <img 
      src={post.url} 
      alt={`Post ${post.id}`} 
      className="w-full h-full object-cover transition duration-500 group-hover:opacity-70"
    />
    
    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center space-x-6 transition duration-300">
      <div className="flex items-center text-white font-semibold text-lg">
        <HeartIcon className="w-5 h-5 mr-2 fill-white" />
        {post.likes}
      </div>
      <div className="flex items-center text-white font-semibold text-lg">
        <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        {post.comments}
      </div>
    </div>
  </div>
);

// =================================================================
// 4. PROFILE HEADER
// =================================================================
const ProfileHeader = ({ user }) => (
  <div>
    <div className="h-44 bg-gray-800 relative">
      <img src={user.banner} alt="Profile Banner" className="w-full h-full object-cover" />
    </div>

    <div className="p-6 relative">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between -mt-16">
        <div className="flex items-center space-x-4">
          <img
            src={user.avatar}
            alt={user.name}
            className="w-32 h-32 rounded-full border-4 border-neutral-900 object-cover shadow-lg"
          />
          <div>
            <h1 className="text-3xl font-bold text-white">{user.name}</h1>
            <p className="text-gray-400 text-lg">{user.handle}</p>
          </div>
        </div>

        <div className="mt-4 md:mt-0 flex space-x-3">
          <button className="px-4 py-2 bg-neutral-800 text-gray-300 rounded-full text-sm font-medium hover:bg-neutral-700 transition">
            Edit Profile
          </button>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-full text-sm font-medium hover:bg-indigo-700 transition shadow-md shadow-indigo-500/30">
            Share
          </button>
        </div>
      </div>

      <p className="mt-4 text-gray-300 max-w-2xl">{user.bio}</p>

      <div className="flex space-x-8 mt-4 text-white">
        <div><span className="font-bold text-lg">{user.stats.posts}</span> <span className="text-gray-400">Posts</span></div>
        <div><span className="font-bold text-lg">{user.stats.followers}</span> <span className="text-gray-400">Followers</span></div>
        <div><span className="font-bold text-lg">{user.stats.following}</span> <span className="text-gray-400">Following</span></div>
      </div>
    </div>
  </div>
);

// =================================================================
// 5. MAIN PROFILE PAGE
// =================================================================
const ProfilePage = () => {
  const [activeTab, setActiveTab] = useState("posts");

  const getTabClass = (tabName) => {
    const base = "flex items-center space-x-2 px-6 py-3 font-medium transition cursor-pointer";
    return activeTab === tabName
      ? `${base} text-indigo-400 border-b-2 border-indigo-500`
      : `${base} text-gray-500 hover:text-white hover:border-b-2 hover:border-gray-700`;
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans p-4">
      <div className="max-w-4xl mx-auto bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl overflow-hidden">
        <ProfileHeader user={mockUser} />

        {/* Tabs */}
        <div className="mt-6 border-t border-neutral-800">
          <div className="flex">
            <div className={getTabClass("posts")} onClick={() => setActiveTab("posts")}>
              <GridIcon className="w-5 h-5" />
              <span>Posts</span>
            </div>
            <div className={getTabClass("likes")} onClick={() => setActiveTab("likes")}>
              <HeartIcon className="w-5 h-5" />
              <span>Likes</span>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {activeTab === "posts" ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-5">
              {mockPosts.map((post) => <PostCard key={post.id} post={post} />)}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <HeartIcon className="w-10 h-10 mx-auto mb-3" />
              <p>You haven't liked any posts yet.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// =================================================================
// 6. EXPORT
// =================================================================
export default ProfilePage;
