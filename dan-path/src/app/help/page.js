"use client"
import React, { useState } from "react";

// --- Icons (Using Lucide-React inspired SVGs) ---
const HelpCircleIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <circle cx="12" cy="12" r="10" />
    <path d="M9.09 9a3 3 0 0 1 5.86 1c0 2-3 3-3 3" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const MailIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
    <polyline points="22,6 12,13 2,6" />
  </svg>
);
const BookOpenIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path d="M2 19C4.8 17.5 7.2 16 12 16s7.2 1.5 10 3" />
    <path d="M12 2C6.8 2 4 4.8 4 8v10c0 1 2.8 2 8 2s8-1 8-2V8c0-3.2-2.8-6-8-6z" />
  </svg>
);
const ClockIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);
const ChevronDown = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// --- Component for a single FAQ item with toggle ---
const FAQItem = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-b border-neutral-700">
      <div 
        className="flex items-center justify-between p-4 cursor-pointer transition hover:bg-neutral-800/70 rounded-t-xl"
        onClick={() => setIsOpen(!isOpen)}
      >
        <p className="font-semibold text-white">{question}</p>
        <ChevronDown 
          className={`w-5 h-5 text-indigo-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : 'rotate-0'}`} 
        />
      </div>
      {isOpen && (
        <div className="px-4 pb-4 text-gray-400 text-sm animate-fade-in">
          {answer}
        </div>
      )}
    </div>
  );
};

// --- Dummy Data for FAQ ---
const faqData = [
  { 
    q: "How do I reset my password?", 
    a: "Go to the Login page, click 'Forgot Password,' and follow the instructions sent to your registered email address." 
  },
  { 
    q: "How can I change my profile picture?", 
    a: "Navigate to Settings > Account > Edit Profile. You can upload a new photo from your device there." 
  },
  { 
    q: "What are the supported payment methods?", 
    a: "We currently accept Visa, MasterCard, PayPal, and Apple Pay." 
  },
  { 
    q: "How long does it take to get a response from support?", 
    a: "Our typical response time for email support is 24-48 hours. For urgent issues, please check our help documentation." 
  },
];

const HelpPage = () => {
  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans p-4 sm:p-8">
      {/* FIX: Removed the proprietary 'jsx global' attributes from the style tag.
        This resolves the console error warning about non-boolean attributes.
      */}
      <style>{`
        /* Minimal custom keyframe for smooth FAQ reveal */
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
      
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white flex items-center mb-8">
          <HelpCircleIcon className="w-8 h-8 mr-3 text-indigo-400" />
          Help & Support
        </h1>
        
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl shadow-xl space-y-8 p-4 sm:p-8">
          
          {/* --- 1. Frequently Asked Questions (FAQ) --- */}
          <div>
            <h2 className="text-2xl font-semibold text-indigo-400 mb-4">FAQs</h2>
            <div className="bg-neutral-800 rounded-xl overflow-hidden">
              {faqData.map((item, index) => (
                <FAQItem key={index} question={item.q} answer={item.a} />
              ))}
            </div>
          </div>

          {/* --- 2. Resources and Contact --- */}
          <div>
            <h2 className="text-2xl font-semibold text-indigo-400 mb-4">Resources & Contact</h2>
            <div className="space-y-3">
              
              {/* Documentation */}
              <a href="#" className="flex items-center justify-between p-4 rounded-xl transition bg-neutral-800 hover:bg-neutral-700 group">
                <div className="flex items-center space-x-4">
                  <div className="p-2 rounded-full bg-indigo-600 group-hover:bg-indigo-500">
                    <BookOpenIcon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-white">View Documentation</p>
                    <p className="text-sm text-gray-400">In-depth guides and feature tutorials.</p>
                  </div>
                </div>
                <ChevronDown className="w-5 h-5 text-gray-400 rotate-[-90deg]" />
              </a>

              {/* Contact Support */}
              <a href="mailto:support@daanpath.com" className="flex items-center justify-between p-4 rounded-xl transition bg-neutral-800 hover:bg-neutral-700 group">
                <div className="flex items-center space-x-4">
                  <div className="p-2 rounded-full bg-indigo-600 group-hover:bg-indigo-500">
                    <MailIcon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-white">Contact Email Support</p>
                    <p className="text-sm text-gray-400">Get personalized assistance from our team.</p>
                  </div>
                </div>
                <ChevronDown className="w-5 h-5 text-gray-400 rotate-[-90deg]" />
              </a>

              {/* Support Hours/Info */}
              <div className="flex items-center space-x-4 p-4 rounded-xl border border-neutral-700/50">
                <div className="p-2 rounded-full bg-neutral-700">
                  <ClockIcon className="w-5 h-5 text-gray-400" />
                </div>
                <div>
                  <p className="font-semibold text-white">Support Hours</p>
                  <p className="text-sm text-gray-400">Monday - Friday, 9:00 AM - 5:00 PM EST</p>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default HelpPage;
