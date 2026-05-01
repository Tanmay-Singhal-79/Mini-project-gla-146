export const mockResources = [
  {
    id: "r1",
    title: "Cracking Placements: The 2024 System Design Guide for Indian MNCs",
    upvotes: 245,
    comments_count: 18,
    creator_name: "Ishaan_Bengaluru",
    link: "https://roadmap.sh/system-design"
  },
  {
    id: "r2",
    title: "UPI 2.0 Integration: Handling Webhooks in Node.js (Indian Fintech)",
    upvotes: 189,
    comments_count: 9,
    creator_name: "Priya_Dev_Delhi",
    link: "https://razorpay.com/docs"
  },
  {
    id: "r3",
    title: "MERN vs Next.js: What Indian Startups are Hiring for in 2024?",
    upvotes: 156,
    comments_count: 24,
    creator_name: "Aryan_Mumbai_99",
    link: "https://nextjs.org/docs"
  },
  {
    id: "r4",
    title: "Low-Cost Cloud Hosting: Deploying your Portfolio from India",
    upvotes: 112,
    comments_count: 7,
    creator_name: "Sneha_Kolkata",
    link: "https://www.hostinger.in"
  }
];

export const mockComments = {
  "r1": [
    { id: "c1", content: "The section on Load Balancing is exactly what they asked me in the Amazon India interview!", author: "Vikram_S", timestamp: "2h ago" },
    { id: "c2", content: "Does this also cover HLD for apps like Zomato/Swiggy?", author: "HungryCoder", timestamp: "1h ago" },
    { id: "c3", content: "Yes, look at page 14 for the delivery tracking logic.", author: "Ishaan_Bengaluru", timestamp: "45m ago" }
  ],
  "r2": [
    { id: "c4", content: "Extremely helpful! Finding clear docs for UPI intent flows is so hard.", author: "Fintech_Guru", timestamp: "5h ago" },
    { id: "c5", content: "Make sure to handle the checksum verification correctly or the transaction will fail silently.", author: "Priya_Dev_Delhi", timestamp: "3h ago" }
  ],
  "r3": [
    { id: "c6", content: "In Bengaluru, almost every YC-backed startup is moving to Next.js App Router.", author: "Indiranagar_Dev", timestamp: "6h ago" },
    { id: "c7", content: "True, but for service-based companies, MERN is still the king.", author: "JobSeeker_2024", timestamp: "4h ago" }
  ]
};
