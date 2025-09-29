"use client";
import React, { useState, useRef, useEffect } from "react";
import { usePathname } from "next/navigation";
import Image from "next/image";
import Link from "next/link";

// --- Icons ---
const HomeIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    <polyline points="9 22 9 12 15 12 15 22" />
  </svg>
);
const SearchIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);
const MessageSquareIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);
const BellIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
  </svg>
);
const MenuIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <line x1="3" y1="12" x2="21" y2="12" />
    <line x1="3" y1="6" x2="21" y2="6" />
    <line x1="3" y1="18" x2="21" y2="18" />
  </svg>
);
const XIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// Navigation links
const socialNavigation = [
  { name: "Feed", href: "/", icon: HomeIcon },
  { name: "Messages", href: "/messages", icon: MessageSquareIcon },
  { name: "Notifications", href: "/notifications", icon: BellIcon },
];

const NavItem = ({ item, currentPath }) => {
  const isActive = currentPath === item.href;
  return (
    <Link
      href={item.href}
      className={`flex items-center justify-center p-2 rounded-full transition ${
        isActive
          ? "bg-indigo-600 text-white"
          : "text-gray-300 hover:bg-gray-800 hover:text-white"
      }`}
    >
      <item.icon className="h-6 w-6" />
    </Link>
  );
};

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);
  const pathname = usePathname();

  const user = {
    photoURL: "https://i.pravatar.cc/100?u=user",
  };

  const handleSearch = (e) => {
    e.preventDefault();
    console.log("Search:", searchQuery);
  };

  const handleSignOut = () => {
    console.log("User signed out");
    setOpen(false);
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <nav className="sticky top-0 z-50 bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))] shadow-md text-white">
      <div className="max-w-6xl mx-auto px-4 flex justify-between items-center h-16">
        {/* Logo */}
        <Link href="/" className="text-xl font-bold">
          DAAN PATH
        </Link>

        {/* Desktop Search Bar */}
        <form
          onSubmit={handleSearch}
          className="hidden md:flex flex-1 mx-4 max-w-md"
        >
          <div className="relative w-full">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="w-full pl-10 pr-4 py-2 rounded-full bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
          </div>
        </form>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center space-x-4">
          {socialNavigation.map((item) => (
            <NavItem key={item.name} item={item} currentPath={pathname} />
          ))}

          {/* User Profile + Dropdown */}
          {user && (
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setOpen(!open)}
                className="rounded-full focus:outline-none"
              >
                <Image
                  src={user.photoURL}
                  width={36}
                  height={36}
                  alt="avatar"
                  className="rounded-full border-2 border-indigo-500 object-cover"
                />
              </button>

              {open && (
                <div className="absolute right-0 mt-3 w-56 rounded-xl bg-neutral-800/90 backdrop-blur-lg shadow-xl border border-gray-700 overflow-hidden">
                  <ul className="py-2 text-sm text-gray-200">
                    <li>
                      <Link
                        href="/profile"
                        className="block px-4 py-2 hover:bg-neutral-700 transition"
                        onClick={() => setOpen(false)}
                      >
                        My Profile
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/settings"
                        className="block px-4 py-2 hover:bg-neutral-700 transition"
                        onClick={() => setOpen(false)}
                      >
                        Settings
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/help"
                        className="block px-4 py-2 hover:bg-neutral-700 transition"
                        onClick={() => setOpen(false)}
                      >
                        Help
                      </Link>
                    </li>
                    <li>
                      <button
                        onClick={handleSignOut}
                        className="w-full text-left px-4 py-2 text-red-400 hover:bg-neutral-700 transition"
                      >
                        Sign out
                      </button>
                    </li>
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="md:hidden p-2 rounded hover:bg-gray-800"
        >
          {isOpen ? <XIcon className="h-6 w-6" /> : <MenuIcon className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden px-2 pb-3 space-y-2">
          {/* Mobile Search Bar */}
          <form onSubmit={handleSearch} className="px-4 mb-2">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search..."
                className="w-full pl-10 pr-4 py-2 rounded-full bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
            </div>
          </form>

          {/* Mobile Profile */}
          <Link
            href="/profile"
            className="flex items-center gap-2 px-4 py-2 rounded hover:bg-gray-800"
            onClick={() => setIsOpen(false)}
          >
            <Image
              width={36}
              height={36}
              src="https://i.pravatar.cc/40"
              alt="User"
              className="rounded-full border-2 border-indigo-500 object-cover"
            />
            <span>Profile</span>
          </Link>

          {socialNavigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-2 px-4 py-2 rounded transition ${
                  isActive
                    ? "bg-indigo-600 text-white"
                    : "text-gray-300 hover:bg-gray-800 hover:text-white"
                }`}
                onClick={() => setIsOpen(false)}
              >
                <item.icon className="h-5 w-5" /> {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
