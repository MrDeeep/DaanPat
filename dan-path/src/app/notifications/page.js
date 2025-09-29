import React from "react";

// Keeping necessary icons for the NotificationItem component
const MessageSquareIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);
const HeartIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);
const MessageCircleIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.42 8.42 0 0 1 7.4 5.1v.2z" />
  </svg>
);
const UserPlusIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="8.5" cy="7" r="4" />
    <line x1="18" y1="10" x2="18" y2="16" />
    <line x1="21" y1="13" x2="15" y2="13" />
  </svg>
);

// =================================================================
// 1. DUMMY DATA
// =================================================================
const dummyNotifications = [
  {
    id: 1,
    type: "like",
    user: "Alice Johnson",
    avatar: "https://i.pravatar.cc/40?u=n1",
    time: "5m ago",
    content: "liked your latest photo.",
    icon: HeartIcon,
    iconColor: "text-red-500",
  },
  {
    id: 2,
    type: "comment",
    user: "Bob Williams",
    avatar: "https://i.pravatar.cc/40?u=n2",
    time: "1h ago",
    content: "commented on your post: \"Great view!\"",
    icon: MessageCircleIcon,
    iconColor: "text-blue-500",
  },
  {
    id: 3,
    type: "follow",
    user: "Charli Davis",
    avatar: "https://i.pravatar.cc/40?u=n3",
    time: "2h ago",
    content: "started following you.",
    icon: UserPlusIcon,
    iconColor: "text-indigo-500",
  },
  {
    id: 4,
    type: "mention",
    user: "David Smith",
    avatar: "https://i.pravatar.cc/40?u=n4",
    time: "4h ago",
    content: "mentioned you in a story.",
    icon: MessageSquareIcon,
    iconColor: "text-yellow-500",
  },
  {
    id: 5,
    type: "like",
    user: "Erica Foster",
    avatar: "https://i.pravatar.cc/40?u=n5",
    time: "1d ago",
    content: "liked your comment.",
    icon: HeartIcon,
    iconColor: "text-red-500",
  },
  {
    id: 6,
    type: "follow",
    user: "Frank Green",
    avatar: "https://i.pravatar.cc/40?u=n6",
    time: "2d ago",
    content: "started following you.",
    icon: UserPlusIcon,
    iconColor: "text-indigo-500",
  },
  {
    id: 7,
    type: "comment",
    user: "Grace Hill",
    avatar: "https://i.pravatar.cc/40?u=n7",
    time: "3d ago",
    content: "replied to your thread: \"I agree 100%.\"",
    icon: MessageCircleIcon,
    iconColor: "text-blue-500",
  },
  // Adding more notifications for scrolling test
  { id: 8, type: "like", user: "Henry Jones", avatar: "https://i.pravatar.cc/40?u=n8", time: "4d ago", content: "liked your reel.", icon: HeartIcon, iconColor: "text-red-500" },
  { id: 9, type: "mention", user: "Ivy King", avatar: "https://i.pravatar.cc/40?u=n9", time: "5d ago", content: "mentioned you in a caption.", icon: MessageSquareIcon, iconColor: "text-yellow-500" },
  { id: 10, type: "follow", user: "Jack Lee", avatar: "https://i.pravatar.cc/40?u=n10", time: "1w ago", content: "started following you.", icon: UserPlusIcon, iconColor: "text-indigo-500" },
];


// =================================================================
// 2. NOTIFICATION ITEM COMPONENT
// =================================================================
const NotificationItem = ({ notification }) => {
  const Icon = notification.icon;

  return (
    <div className="flex items-start gap-4 p-4 rounded-xl transition hover:bg-neutral-800 cursor-pointer border-b border-gray-800 last:border-b-0">
      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-neutral-800 ${notification.iconColor} p-1.5`}>
        <Icon className="h-5 w-5 fill-current" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <p className="text-sm font-semibold text-white truncate">
            {notification.user}
          </p>
          <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
            {notification.time}
          </span>
        </div>
        <p className="text-gray-400 text-sm">
          <span className="font-medium text-white mr-1">{notification.user}</span>
          {notification.content}
        </p>
      </div>
      
      {/* User Avatar */}
      <img
        src={notification.avatar}
        alt={notification.user}
        width={40}
        height={40}
        className="rounded-full border-2 border-indigo-500 object-cover flex-shrink-0"
      />
    </div>
  );
};

// =================================================================
// 3. NOTIFICATIONS PAGE COMPONENT (The core content)
// =================================================================
const NotificationsPage = () => {
  const unreadNotifications = dummyNotifications.filter(n => n.id <= 3); // Example unread
  const readNotifications = dummyNotifications.filter(n => n.id > 3);

  return (
    <div className="flex-1 min-h-full p-4 md:p-8 bg-neutral-900 overflow-y-auto scrollbar-hide">
      <h1 className="text-4xl font-extrabold text-white mb-8 border-b border-indigo-700 pb-3">
        Activity Notifications
      </h1>

      <div className="space-y-6">
        {/* Unread Section */}
        {unreadNotifications.length > 0 && (
          <section>
            <h2 className="text-xl font-bold text-indigo-400 mb-4">New (Unread)</h2>
            <div className="bg-neutral-950 rounded-xl shadow-lg border border-indigo-800/50">
              {unreadNotifications.map((n) => (
                <NotificationItem key={n.id} notification={n} />
              ))}
            </div>
          </section>
        )}

        {/* Read Section */}
        <section>
          <h2 className="text-xl font-bold text-gray-400 mb-4">Earlier</h2>
          <div className="bg-neutral-950 rounded-xl shadow-lg border border-gray-800/50">
            {readNotifications.map((n) => (
              <NotificationItem key={n.id} notification={n} />
            ))}
          </div>
        </section>

        {dummyNotifications.length === 0 && (
          <div className="text-center py-10 text-gray-500 text-lg">
            You are all caught up! No new notifications.
          </div>
        )}
      </div>
    </div>
  );
};


// =================================================================
// 4. MAIN APPLICATION COMPONENT
// =================================================================
export default function App() {
  return (
    // Centering the notification page content
    <div className="min-h-screen bg-neutral-950 font-sans p-4">
      <div className="max-w-3xl mx-auto flex min-h-[95vh] border border-neutral-800 rounded-xl shadow-2xl overflow-hidden w-full">
        <NotificationsPage />
      </div>
    </div>
  );
}
