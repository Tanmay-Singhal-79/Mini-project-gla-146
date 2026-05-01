export const mockLessons = {
  "Modern HTML5 Mastery": {
    title: "Modern HTML5 Mastery",
    introduction: "Semantic HTML is the backbone of the modern web. It provides meaning to your markup, ensuring that both browsers and assistive technologies can interpret your content correctly.",
    sections: [
      {
        heading: "1. The Semantic Revolution",
        content: "HTML5 introduced a suite of semantic elements like `<article>`, `<section>`, and `<main>`. These tags tell the browser exactly what kind of content they contain, which is vital for SEO and accessibility."
      },
      {
        heading: "2. Accessibility (ARIA)",
        content: "When semantic tags aren't enough, we use ARIA roles. ARIA (Accessible Rich Internet Applications) attributes provide extra context to screen readers, such as `aria-expanded` for dropdowns."
      }
    ],
    tip: "Always use the most specific tag possible. Use `<nav>` for navigation, not just a `<div>` with a class."
  },
  "JavaScript ES2024+": {
    title: "JavaScript ES2024+",
    introduction: "JavaScript has evolved into a high-performance, multi-paradigm language. Mastering the latest ES2024+ features is essential for modern fullstack development.",
    sections: [
      {
        heading: "1. Asynchronous Mastery",
        content: "The `async/await` syntax transformed how we handle side effects. Combined with `Promise.allSettled()`, you can manage complex concurrent operations with ease."
      },
      {
        heading: "2. Optional Chaining & Nullish Coalescing",
        content: "Features like `?.` and `??` allow you to write much cleaner code by safely accessing deeply nested object properties without crashing."
      }
    ],
    tip: "Use `const` by default. Only use `let` if you explicitly need to reassign the variable."
  }
};

export const getFallbackLesson = (title) => {
  const key = Object.keys(mockLessons).find(k => 
    title.toLowerCase().includes(k.toLowerCase()) || 
    k.toLowerCase().includes(title.toLowerCase())
  );
  
  if (key) return mockLessons[key];

  // Deep & God-Level Content Generation for 100+ roadmaps
  return {
    title: title,
    introduction: `Welcome to the high-intensity module on **${title}**. This is not just a tutorial; it's a deep-dive into the architectural nuances, performance bottlenecks, and industrial-grade patterns that define modern technical excellence. By the end of this module, you will have moved beyond basic syntax into the realm of professional mastery.`,
    sections: [
      {
        heading: "1. Theoretical Underpinnings",
        content: `Understanding **${title}** starts with grasping the 'First Principles'. We analyze the core logic that makes this technology indispensable. Whether it's resource management, state propagation, or cryptographic integrity, this section strips away the abstraction to reveal the fundamental mechanics.`,
      },
      {
        heading: "2. Architectural Pattern Analysis",
        content: `Modern systems are built on patterns, not just code. For **${title}**, we focus on modularity and scalability. How does this fit into a microservices architecture? How do we handle race conditions? We explore the design decisions that lead to robust, fail-safe implementations.`,
        language: "Architecture Logic",
        code: `// High-level conceptual implementation of ${title}
class ${title.replace(/\s+/g, '')}Service {
  constructor(config) {
    this.config = config;
    this.state = 'INITIALIZED';
  }

  async executeSequence(payload) {
    try {
      console.log('Initiating ${title} logic chain...');
      // Implement advanced validation and processing here
      const result = await this.process(payload);
      this.state = 'COMPLETED';
      return result;
    } catch (error) {
      this.state = 'FAILED';
      throw new Error(\`Sequence interrupted: \${error.message}\`);
    }
  }
}`
      },
      {
        heading: "3. Hyper-Performance Optimization",
        content: `In production, every millisecond counts. We look at **${title}** through the lens of performance. This involves minimizing memory footprints, optimizing garbage collection cycles, and ensuring that our execution logic doesn't block the main event loop or exceed gas limits.`,
      },
      {
        heading: "4. Security & Edge-Case Resilience",
        content: `A master knows that 'it works' is not enough. We must ensure it 'cannot break'. We dive into security audits for **${title}**, handling unexpected inputs, preventing injection attacks, and implementing robust fallback mechanisms that maintain system integrity during partial failures.`,
      },
      {
        heading: "5. Industrial Deployment Strategies",
        content: `Finally, we look at the 'DevOps' of **${title}**. Containerization, CI/CD pipelines, and real-time monitoring. We discuss how to observe this logic in the wild using telemetry and how to perform blue-green deployments without user-facing downtime.`,
      }
    ],
    tip: `The difference between a senior and a lead developer is the ability to anticipate how **${title}** will behave under 100x load. Always build for scale.`
  };
};
