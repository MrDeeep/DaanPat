"use client";
import Stories from "../components/Stories";
import PostsFeed from "../components/PostsFeed";
import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      <main className="max-w-6xl mx-auto py-8 px-4">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Sidebar → Removed Weather Forecast */}
          <aside className="hidden lg:block lg:col-span-3">
            <div className="sticky top-20 flex flex-col h-[500px] bg-gray-900 rounded-xl shadow-lg border border-gray-800">
              <p className="font-bold text-lg border-b border-gray-800 p-4 bg-gray-800/50 rounded-t-xl text-center">
                {/* Empty for now */}
              </p>
            </div>
          </aside>

          {/* Main Feed Column */}
          <section className="lg:col-span-6 space-y-6">
            <Stories />
            <PostsFeed />
          </section>

          {/* Right Sidebar → Friends Online */}
          <aside className="hidden xl:block xl:col-span-3">
            <div className="sticky top-20 space-y-6 text-gray-300">
              <p className="font-bold text-lg border-b border-gray-800 pb-2">
                Friends Online
              </p>

              {["f1", "f2"].map((id, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 hover:bg-gray-800 rounded-lg transition cursor-pointer"
                >
                  <div className="relative">
                    <Image
                      src={`https://i.pravatar.cc/32?u=${id}`}
                      alt="Friend"
                      width={32}
                      height={32}
                      className="rounded-full"
                    />
                    <span className="absolute bottom-0 right-0 h-2.5 w-2.5 bg-green-500 rounded-full ring-2 ring-neutral-950"></span>
                  </div>
                  <span>Friend {idx + 1} Name</span>
                </div>
              ))}
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
