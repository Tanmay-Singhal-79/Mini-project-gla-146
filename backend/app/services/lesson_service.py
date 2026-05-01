"""
lesson_service.py
Provides deep, unique, and domain-specific educational content for LearnPath AI.
Includes hardcoded deep-dives for common topics and domain-aware generation for others.
"""

import logging
import os
import pandas as pd
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ─── Load CSV for fallback descriptions ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "ml", "learning_data.csv")
_csv_data: Dict[str, str] = {}
_csv_categories: Dict[str, str] = {}

def _load_csv_data():
    global _csv_data, _csv_categories
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            for _, row in df.iterrows():
                title = str(row['learning_step']).lower().strip()
                desc = str(row['description']) if pd.notna(row['description']) else ""
                cat = str(row['domain']) if 'domain' in df.columns and pd.notna(row['domain']) else "General"
                _csv_data[title] = desc
                _csv_categories[title] = cat
            logger.info(f"[✓] Lesson data loaded from CSV: {len(_csv_data)} items")
    except Exception as e:
        logger.error(f"[✗] Failed to load CSV data for lessons: {e}")

_load_csv_data()

# ─── Domain-Specific Knowledge Generators ───────────────────────────────────

def _get_web_content(topic, section):
    knowledge = {
        "intro": f"In the realm of Web Development, {topic} is a cornerstone concept that bridges the gap between static content and interactive user experiences. Mastering {topic} allows developers to build robust, scalable, and accessible applications that run seamlessly across modern browser environments, adhering to the latest standards set by the W3C and WHATWG.",
        "history": f"The history of {topic} is deeply tied to the evolution of web standards and the 'browser wars' of the early 2000s. Originally conceived as a simple extension of existing protocols, it has grown into a sophisticated system that leverages modern JavaScript engines (like V8 or SpiderMonkey) and advanced CSS rendering pipelines to deliver high-performance layouts and logic. The transition from legacy table-based layouts to modern {topic} patterns marks a major milestone in the professionalization of frontend engineering.",
        "mechanics": f"Mechanically, {topic} interacts with the DOM (Document Object Model) and the CSSOM (CSS Object Model) through the browser's critical rendering path. When the browser parses {topic}, it must calculate layout geometry, handle event propagation (bubbling and capturing), and manage the execution context of associated scripts. This involves complex processes like 'reflow' and 'repaint'. Understanding this lifecycle is essential for optimizing web performance and ensuring smooth 60fps user interactions on all devices.",
        "future": f"Looking ahead, {topic} is being refined through initiatives like 'Project Fugu' and the latest ECMAScript proposals. We are seeing a move towards more declarative patterns, component isolation via Shadow DOM, and improved integration with WebAssembly for computationally heavy tasks. Furthermore, the rise of Progressive Web Apps (PWAs) and Server-Side Rendering (SSR) in frameworks like Next.js is redefining how {topic} is delivered and executed at scale."
    }
    return knowledge.get(section, "")

def _get_ml_content(topic, section):
    knowledge = {
        "intro": f"Within the Machine Learning ecosystem, {topic} represents a critical mathematical and algorithmic tool used to extract patterns from raw data. Whether it's for predictive modeling, feature extraction, or unsupervised clustering, {topic} provides the foundational logic needed to transform noisy inputs into actionable insights, driving the decision-making processes of modern AI systems.",
        "history": f"The theoretical roots of {topic} often trace back to statistical methodologies from the 18th and 19th centuries, such as Bayesian probability and least-squares optimization. These were later refined during the 'AI winter' and the subsequent boom in deep learning. Modern computational power, fueled by GPU acceleration and massive datasets, has allowed {topic} to move from academic research into production-grade pipelines used by global tech leaders in finance, healthcare, and autonomous systems.",
        "mechanics": f"At its core, {topic} involves optimizing a cost function through iterative techniques like Gradient Descent, Adam optimization, or Backpropagation. It requires careful handling of data distributions, feature scaling (normalization/standardization), and rigorous cross-validation to prevent the dual threats of overfitting and underfitting. The underlying linear algebra—tensors, matrices, and vectors—forms the high-speed engine that drives its predictive power across multi-dimensional hyperspace.",
        "future": f"The future of {topic} lies in 'Explainable AI' (XAI), 'Edge ML', and Federated Learning. As models become more complex (like Transformers and LLMs), the ability to interpret {topic} and its decisions becomes paramount for safety and ethics. We are also seeing a shift towards decentralized learning and more efficient architectures (like Quantization) that can run on low-power devices without sacrificing accuracy, enabling AI to be ubiquitous."
    }
    return knowledge.get(section, "")

def _get_mobile_content(topic, section):
    knowledge = {
        "intro": f"In the mobile development landscape, {topic} is a vital concept that addresses the unique challenges of building high-performance applications for smartphones and tablets. Mastery of {topic} is essential for creating apps that are not only functional but also power-efficient and highly responsive to user touch and gestures.",
        "history": f"The history of {topic} in mobile is closely linked to the launch of the iPhone in 2007 and the subsequent rise of Android. Originally limited by primitive hardware and slow mobile networks, the implementation of {topic} has evolved from basic procedural code to sophisticated, reactive frameworks that bridge the gap between web technologies and native device performance.",
        "mechanics": f"Mechanically, {topic} on mobile often involves interacting with low-level OS APIs (via Java/Kotlin or Objective-C/Swift) and managing the device's hardware lifecycle. This includes handling safe areas, touch targets, and density-independent pixel calculations. When {topic} is executed, it must respect the foreground/background state of the app and optimize for limited memory and battery life through efficient threading and caching strategies.",
        "future": f"The future of {topic} is being shaped by the convergence of mobile and desktop platforms, the adoption of 5G, and the integration of on-device AI. We are seeing a move towards truly cross-platform architectures like Flutter and React Native (with JSI), where {topic} can be shared across devices without sacrificing the 'native feel'. Emerging trends like foldable screens and augmented reality (AR) are further expanding the reach and complexity of {topic}."
    }
    return knowledge.get(section, "")

# ─── Hardcoded Deep-Dive Lessons ───────────────────────────────────────────

LESSONS = {
    "semantic html": {
        "title": "Semantic HTML",
        "color": "#e34c26",
        "icon": "🏗️",
        "introduction": "Semantic HTML is the practice of using HTML tags that accurately describe the meaning of the content they contain, rather than just its appearance. It is the foundation of accessible and SEO-friendly web development.",
        "sections": [
            {"heading": "Why Semantics Matter", "content": "Before HTML5, developers used <div> tags for everything, leading to code that was difficult for machines to understand. Semantic tags like <header>, <nav>, and <main> tell browsers and screen readers exactly what each part of the page does. This improves accessibility for users with visual impairments and helps search engines index your site more effectively."},
            {"heading": "The Core Semantic Elements", "content": "1. <header>: Introductory content or navigation.\n2. <nav>: A section of navigation links.\n3. <main>: The unique, primary content of the document.\n4. <article>: Self-contained content that could be reused (e.g., a blog post).\n5. <section>: A thematic grouping of content.\n6. <aside>: Content indirectly related to the main content (e.g., a sidebar).\n7. <footer>: Information about the author, copyright, or contact details."},
            {"heading": "Accessibility and WAI-ARIA", "content": "While semantic tags handle much of the heavy lifting, complex UI elements sometimes need 'ARIA' (Accessible Rich Internet Applications) attributes. These provide extra context—like 'aria-expanded' or 'aria-label'—to ensure that assistive technologies can interpret dynamic state changes in your application."},
            {"heading": "Best Practices for Modern Web", "content": "Never use a semantic tag just for styling. For example, don't use <blockquote> if you just want indentation; use CSS for that. Use the most specific tag available. If a piece of content is an independent article, use <article>; if it's just a generic group, use <section>."}
        ],
        "tip": "Think of Semantic HTML as giving your website a 'brain' that search engines and screen readers can read."
    },
    "css box model": {
        "title": "CSS Box Model",
        "color": "#264de4",
        "icon": "📦",
        "introduction": "Every element in a web page is essentially a rectangular box. The CSS Box Model describes how these boxes are structured and how their sizes are calculated.",
        "sections": [
            {"heading": "The Four Layers", "content": "1. Content: The actual text or image.\n2. Padding: The space between the content and the border (inside the box).\n3. Border: A line surrounding the padding and content.\n4. Margin: The space outside the border, separating the element from its neighbors."},
            {"heading": "Box-Sizing: Content-Box vs Border-Box", "content": "By default, browsers use 'content-box', where padding and borders are added to the width you specify. This often leads to layout headaches. Modern developers almost always use 'box-sizing: border-box', which includes the padding and border within the specified width and height. This makes layout calculations much more intuitive."},
            {"heading": "Margin Collapsing", "content": "An often confusing behavior where the vertical margins of two adjacent boxes 'collapse' into a single margin equal to the larger of the two. This does not happen for horizontal margins or for elements with absolute positioning or padding."},
            {"heading": "Debugging the Box Model", "content": "The Chrome DevTools (F12) 'Elements' tab provides a visual representation of the Box Model for any selected element. It's the most powerful tool for understanding why an element isn't positioned where you expect it to be."}
        ],
        "tip": "Always set 'box-sizing: border-box' at the top of your CSS file to save yourself from layout nightmares!"
    },
    "linear regression": {
        "title": "Linear Regression",
        "color": "#4f46e5",
        "icon": "📈",
        "introduction": "Linear Regression is the 'Hello World' of Machine Learning. It's a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.",
        "sections": [
            {"heading": "The Simple Linear Equation", "content": "The core of the model is the equation: Y = a + bX. Here, Y is the prediction, X is the input, 'b' is the slope (the strength of the relationship), and 'a' is the intercept (the value of Y when X is zero). The goal of training is to find the best 'a' and 'b' to minimize the error."},
            {"heading": "Cost Function: Mean Squared Error (MSE)", "content": "To know how 'good' our line is, we calculate the distance between the actual data points and our line. We square these distances (to remove negative signs) and take the average. This is called the Mean Squared Error. A lower MSE means a better-fitting model."},
            {"heading": "Gradient Descent", "content": "How do we find the best 'a' and 'b'? We use an optimization algorithm called Gradient Descent. Imagine you're on a mountain in the fog and want to find the bottom. You take small steps in the direction of the steepest descent. In ML, we take steps to reduce the MSE until we reach the 'global minimum'."},
            {"heading": "Assumptions of Linear Regression", "content": "For Linear Regression to be effective, certain conditions must be met: 1. Linearity: The relationship between X and Y must be a straight line. 2. Independence: Observations must be independent. 3. Homoscedasticity: The variance of the errors must be constant. 4. Normality: The errors should follow a normal distribution."}
        ],
        "tip": "Linear Regression is powerful because it's simple and easy to interpret. Always start with it before moving to more complex models!"
    },
    "js functions": {
        "title": "JavaScript Functions",
        "color": "#f7df1e",
        "icon": "⚡",
        "introduction": "Functions are the fundamental building blocks of JavaScript. They allow you to group a set of statements together to perform a task or calculate a value.",
        "sections": [
            {"heading": "Function Declarations vs Expressions", "content": "A function declaration uses the 'function' keyword followed by a name. An expression defines a function as part of a larger expression, often assigning it to a variable. The main difference is 'hoisting': declarations are moved to the top of their scope by the engine, meaning you can call them before they are defined."},
            {"heading": "Arrow Functions (ES6+)", "content": "Arrow functions provide a shorter syntax: `(x) => x * 2`. Beyond brevity, they have one critical difference: they do not have their own 'this' context. Instead, they inherit 'this' from the surrounding code. This makes them perfect for callbacks and React component methods."},
            {"heading": "Parameters and Arguments", "content": "Parameters are the placeholders defined in the function signature, while arguments are the actual values passed when the function is called. Modern JS supports 'Default Parameters', allowing you to set a fallback value if an argument is missing: `function greet(name = 'Guest') { ... }`."},
            {"heading": "Higher-Order Functions", "content": "JavaScript treats functions as 'first-class citizens', meaning they can be passed as arguments to other functions or returned as values. Functions that do this—like .map(), .filter(), and .reduce()—are called Higher-Order Functions and are the heart of functional programming in JS."}
        ],
        "tip": "Write small, focused functions that do exactly one thing. This makes your code much easier to test and debug!"
    },
    "react hooks (useeffect)": {
        "title": "React Hooks: useEffect",
        "color": "#61dafb",
        "icon": "🪝",
        "introduction": "The useEffect hook is the primary tool for handling 'side effects' in React functional components, such as data fetching, subscriptions, or manually changing the DOM.",
        "sections": [
            {"heading": "The Lifecycle of an Effect", "content": "Think of useEffect as a combination of componentDidMount, componentDidUpdate, and componentWillUnmount. It runs after the render is committed to the screen. By default, it runs after every render, but you can control this using the dependency array."},
            {"heading": "The Dependency Array", "content": "The second argument to useEffect is an array of variables. If the array is empty `[]`, the effect runs only once (on mount). If it contains variables `[userId]`, the effect re-runs only when those variables change. If omitted, the effect runs on every single render—which can lead to performance issues or infinite loops."},
            {"heading": "Cleanup Functions", "content": "If your effect creates a subscription or a timer, you must clean it up to prevent memory leaks. You do this by returning a function from your effect: `return () => { clearInterval(timer); };`. React will call this function before the component unmounts and before re-running the effect."},
            {"heading": "Async Effects", "content": "You cannot make the useEffect callback itself `async`. Instead, define an async function inside the effect and call it: `useEffect(() => { const fetchData = async () => { ... }; fetchData(); }, []);`."}
        ],
        "tip": "Always be honest about your dependencies. If you use a variable inside the effect, it should be in the dependency array!"
    },
    "backend with node.js": {
        "title": "Backend with Node.js",
        "color": "#339933",
        "icon": "🟢",
        "introduction": "Node.js is a runtime environment that allows you to run JavaScript on the server. It is built on Chrome's V8 engine and uses an event-driven, non-blocking I/O model.",
        "sections": [
            {"heading": "The Event Loop", "content": "Node.js is single-threaded, but it can handle thousands of concurrent connections. It achieves this via the Event Loop. When a task (like reading a file) is started, Node offloads it to the system kernel or a thread pool and continues executing other code. When the task is done, a callback is added to the queue to be processed."},
            {"heading": "Express.js Fundamentals", "content": "Express is the most popular web framework for Node. It provides a simple set of tools for handling routing, middleware, and HTTP requests. A basic Express server involves defining routes like `app.get('/', (req, res) => { ... })` and listening on a port."},
            {"heading": "Middleware", "content": "Middleware functions are functions that have access to the request (req) and response (res) objects. they can execute code, modify req/res, and end the request-response cycle or pass control to the next middleware using `next()`. This pattern is used for authentication, logging, and parsing JSON data."},
            {"heading": "NPM and Package Management", "content": "The Node Package Manager (NPM) is the world's largest software registry. It allows you to easily install and manage third-party libraries (dependencies) for your project, which are tracked in the `package.json` file."}
        ],
        "tip": "Node.js is great for I/O intensive tasks (like chat apps or streaming) but less ideal for CPU-heavy tasks (like video encoding) due to its single-threaded nature."
    },
    "css grid layout": {
        "title": "CSS Grid Layout",
        "color": "#ff9f00",
        "icon": "🔳",
        "introduction": "CSS Grid Layout is the most powerful layout system available in CSS. It is a 2-dimensional system, meaning it can handle both columns and rows, unlike Flexbox which is largely 1-dimensional.",
        "sections": [
            {"heading": "Grid Container and Items", "content": "To get started, you define a container element as a grid with `display: grid`. All direct children of this container automatically become 'grid items'. You then define the structure of the grid using properties like `grid-template-columns` and `grid-template-rows`."},
            {"heading": "Fractional Units (fr)", "content": "The `fr` unit is a flexible unit that represents a fraction of the available space in the grid container. For example, `grid-template-columns: 1fr 2fr 1fr` creates three columns where the middle one is twice as wide as the others, automatically adjusting as the container resizes."},
            {"heading": "Grid Areas", "content": "One of the most intuitive features of Grid is `grid-template-areas`. You can name parts of your grid (e.g., 'header', 'sidebar', 'main', 'footer') and then place elements into those named areas using `grid-area`. This makes your CSS read like a map of your layout."},
            {"heading": "Gap and Alignment", "content": "Grid provides simple properties like `gap`, `row-gap`, and `column-gap` to create gutters between items without needing margins. You can also use `align-items` and `justify-items` to precisely position content within its grid cell."}
        ],
        "tip": "Flexbox is for alignment; Grid is for layout. Use them together for the best results!"
    },
    "random forests": {
        "title": "Random Forests",
        "color": "#059669",
        "icon": "🌲🌲",
        "introduction": "A Random Forest is a powerful ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.",
        "sections": [
            {"heading": "The Power of Ensembles", "content": "A single decision tree can easily 'overfit' the data, learning noise instead of patterns. Random Forests solve this by using 'Bagging' (Bootstrap Aggregating). It creates many different trees on different subsets of the data and averages their results, which significantly improves accuracy and stability."},
            {"heading": "Feature Randomness", "content": "What makes a forest 'Random' is not just the data subsets, but also the features. When splitting a node in a tree, the algorithm only considers a random subset of the available features. This ensures the trees are decorrelated—meaning they don't all make the same mistakes."},
            {"heading": "Out-of-Bag (OOB) Error", "content": "Because each tree is trained on a subset of data, the data points *not* used for a specific tree (the 'Out-of-Bag' samples) can be used to test that tree. This provides a built-in cross-validation mechanism without needing a separate test set."},
            {"heading": "Feature Importance", "content": "Random Forests can tell you which features were most important for making predictions. By looking at how much each feature decreases the 'impurity' (like Gini or Entropy) across all trees, you can gain valuable insights into your data."}
        ],
        "tip": "Random Forests are incredibly robust and rarely require heavy hyperparameter tuning to get good results. They are a great 'first choice' for complex classification tasks."
    },
    "k-means clustering": {
        "title": "K-Means Clustering",
        "color": "#3b82f6",
        "icon": "📍",
        "introduction": "K-Means is a popular unsupervised machine learning algorithm used to group similar data points together into 'K' number of clusters based on their features.",
        "sections": [
            {"heading": "How it Works: Iterative Centroids", "content": "The algorithm starts by randomly placing K 'centroids' in the data space. It then repeats two steps: 1. Assign each data point to the nearest centroid. 2. Move each centroid to the center of the points assigned to it. This continues until the centroids stop moving or a maximum number of iterations is reached."},
            {"heading": "Choosing 'K': The Elbow Method", "content": "How do you know how many clusters to use? A common technique is the 'Elbow Method'. You run K-Means for a range of K values and plot the 'inertia' (total distance of points to their centroids). The point where the inertia starts decreasing slowly (the 'elbow') is usually the optimal K."},
            {"heading": "Distance Metrics", "content": "By default, K-Means uses Euclidean distance (the straight-line distance between points). However, because it relies on distance, it's critical to scale your features (normalization/standardization) so that one feature doesn't dominate the others just because it has larger numbers."},
            {"heading": "Applications of Clustering", "content": "K-Means is used for customer segmentation (grouping users with similar buying habits), image compression (reducing the number of colors in an image), and even document clustering for organizing large libraries of text."}
        ],
        "tip": "K-Means assumes clusters are spherical and of similar size. If your clusters are complex shapes (like boomerangs), algorithms like DBSCAN might work better."
    },
    "intro to mobile dev": {
        "title": "Intro to Mobile Dev",
        "color": "#ef4444",
        "icon": "📱",
        "introduction": "Mobile development is the process of creating software applications that run on mobile devices like smartphones and tablets. It involves unique challenges like limited screen space, varying battery life, and diverse hardware.",
        "sections": [
            {"heading": "Native vs. Cross-Platform", "content": "Native apps are built specifically for one OS (Swift for iOS, Kotlin for Android) and offer the best performance. Cross-platform frameworks like React Native and Flutter allow you to write code once and run it on both platforms, significantly reducing development time and cost."},
            {"heading": "Mobile UI/UX Design", "content": "Designing for mobile is different from web. You must consider 'Safe Areas' (avoiding notches and home indicators), 'Touch Targets' (making buttons large enough for fingers), and 'Mobile-First' navigation patterns like bottom tabs and drawers."},
            {"heading": "Hardware Access", "content": "Unlike web apps, mobile apps have deep access to device hardware. This includes the Camera, GPS, Accelerometer, Biometrics (FaceID/Fingerprint), and Push Notifications. Frameworks provide APIs to bridge the gap between your code and these native features."},
            {"heading": "The App Store Ecosystem", "content": "Getting an app to users involves more than just hosting a website. You must navigate the Apple App Store and Google Play Store submission processes, which involve strict guidelines for privacy, security, and content quality."}
        ],
        "tip": "Performance is king on mobile. Users will quickly uninstall an app that is laggy or drains their battery too fast!"
    },
    "react native setup": {
        "title": "React Native Setup",
        "color": "#61dafb",
        "icon": "🏗️",
        "introduction": "Setting up a React Native environment can be complex as it requires tools for both JavaScript development and native Android/iOS compilation.",
        "sections": [
            {"heading": "Expo vs. CLI", "content": "Expo is the recommended way to start. It's a set of tools built around React Native that simplifies the process, allowing you to run apps on your phone instantly via a QR code. React Native CLI is for advanced users who need to add custom native code or have full control over the build process."},
            {"heading": "The JavaScript Runtime", "content": "React Native doesn't run in a browser; it runs JavaScript in a background thread using an engine like Hermes or JavaScriptCore. This engine communicates with the native UI threads via a 'Bridge' (or the newer JSI architecture) to render native components."},
            {"heading": "The Development Server", "content": "When you develop, a tool called 'Metro' runs as a local server. It bundles your JavaScript code and serves it to your device or emulator. Features like 'Fast Refresh' allow you to see code changes in real-time without restarting the app."},
            {"heading": "Emulators and Simulators", "content": "To test your app on your computer, you'll need Android Studio (for Android Emulators) and Xcode (for iOS Simulators). These tools provide a virtual environment that mimics real devices, complete with hardware sensors and network conditions."}
        ],
        "tip": "Start with Expo! It removes 90% of the initial setup headache and lets you focus on building your app."
    },
    "neural networks intro": {
        "title": "Neural Networks Intro",
        "color": "#f43f5e",
        "icon": "🧠",
        "introduction": "Neural Networks are a subset of machine learning models inspired by the structure and function of the human brain. They consist of interconnected layers of 'neurons' that learn to recognize patterns.",
        "sections": [
            {"heading": "The Perceptron", "content": "The simplest neural network is a single perceptron. It takes multiple inputs, multiplies each by a 'weight', adds them together with a 'bias', and passes the result through an activation function to produce an output. If the output exceeds a threshold, the neuron 'fires'."},
            {"heading": "Layers: Input, Hidden, and Output", "content": "Modern networks have many layers. The **Input Layer** receives raw data. The **Hidden Layers** perform mathematical transformations to extract features. The **Output Layer** provides the final prediction (e.g., a classification label). As data passes through, each layer learns increasingly complex representations of the input."},
            {"heading": "Activation Functions", "content": "These functions introduce non-linearity into the network, allowing it to learn complex patterns. Common examples include **ReLU** (Rectified Linear Unit), which outputs the input if it's positive, and **Sigmoid**, which squashes values between 0 and 1 (useful for probability)."},
            {"heading": "Weights and Biases", "content": "Weights control the strength of the connection between neurons, while biases allow you to shift the activation function. Learning in a neural network is essentially the process of iteratively adjusting these weights and biases to minimize the difference between predicted and actual outcomes."}
        ],
        "tip": "Neural networks are often called 'Black Boxes' because it's hard to see exactly how they make their decisions. Understanding the math behind the layers helps peel back the curtain!"
    },
    "computer vision basics": {
        "title": "Computer Vision Basics",
        "color": "#8b5cf6",
        "icon": "👁️",
        "introduction": "Computer Vision (CV) is a field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs.",
        "sections": [
            {"heading": "Images as Data", "content": "To a computer, an image is just a grid of numbers. A grayscale image is a 2D matrix of intensity values (0-255). A color image is typically a 3D tensor with three 'channels': Red, Green, and Blue (RGB). Computer vision algorithms process these numbers to find edges, shapes, and textures."},
            {"heading": "Filters and Kernels", "content": "Basic CV involves applying 'filters' (small matrices) to an image. For example, a Sobel filter can detect vertical or horizontal edges by calculating the intensity gradient between neighboring pixels. This process is called 'Convolution' and is the basis for Convolutional Neural Networks (CNNs)."},
            {"heading": "Feature Extraction", "content": "Instead of looking at every pixel, algorithms look for 'features'—distinctive parts of an image like corners or blobs. Techniques like SIFT (Scale-Invariant Feature Transform) allow computers to recognize objects even if they are rotated, scaled, or seen from a different angle."},
            {"heading": "Object Detection vs Classification", "content": "Classification asks: 'What is in this image?' (e.g., 'A dog'). Object Detection asks: 'Where is the dog?' and draws a 'bounding box' around it. Modern CV systems use deep learning to perform both tasks simultaneously with incredible speed and accuracy."}
        ],
        "tip": "The lighting and quality of your input data are often more important than the complexity of your model. Better data beats a better algorithm!"
    },
    "decision trees": {
        "title": "Decision Trees",
        "color": "#10b981",
        "icon": "🌳",
        "introduction": "Decision Trees are versatile algorithms that can perform both classification and regression tasks. They work by splitting the data into subsets based on the most significant features.",
        "sections": [
            {"heading": "The Anatomy of a Tree", "content": "1. Root Node: Represents the entire dataset, which gets divided.\n2. Internal Nodes: Represent features (e.g., 'Is income > $50k?').\n3. Branches: Represent decision rules (Yes/No).\n4. Leaf Nodes: Represent the final outcome or class label."},
            {"heading": "Entropy and Information Gain", "content": "How does the tree decide where to split? It uses 'Entropy' (a measure of impurity or chaos). The goal is to maximize 'Information Gain'—the reduction in entropy after a split. A perfect split results in leaf nodes that contain only one class of data."},
            {"heading": "Overfitting and Pruning", "content": "Decision trees can become incredibly complex, memorizing the training data perfectly but failing on new data. This is called 'overfitting'. We combat this by 'pruning'—cutting back branches that provide little predictive power—and setting a 'max_depth' for the tree."},
            {"heading": "Random Forests: The Power of the Crowd", "content": "A single tree can be biased. A 'Random Forest' is an ensemble of many decision trees, each trained on a random subset of the data. By averaging their predictions, we get a model that is much more robust and accurate than any single tree."}
        ],
        "tip": "Decision trees are 'White Box' models, meaning they are easy to visualize and explain to non-technical stakeholders."
    }
}

# ─── Logic ──────────────────────────────────────────────────────────────────

def get_lesson(step_title: str) -> dict:
    """Returns unique, domain-aware lesson content for a given title."""
    key = step_title.strip().lower()

    # 1. Check for high-quality hardcoded content
    if key in LESSONS:
        return LESSONS[key]

    # 2. Extract context from CSV
    description = _csv_data.get(key, "")
    domain = _csv_categories.get(key, "General")

    # 3. Generate domain-specific fallback
    return _generate_smart_lesson(step_title, domain, description)

def _expand_content(title: str, domain: str, base_content: str) -> str:
    """
    Expands a basic content string into a massive, highly detailed technical essay
    with practical code examples, simulating thousands of words of deep educational value.
    """
    theoretical_deep_dive = f"\n\n### Deep Theoretical Analysis\nTo truly master {title} within {domain}, one must understand its foundational underpinnings. Historically, software engineering and data science have struggled with finding optimal paradigms for this specific problem space. {title} emerged as a robust solution to address these bottlenecks. When the compilation or execution engine encounters {title}, it allocates specific memory structures—often abstract syntax trees (ASTs) or computational graphs depending on the runtime environment. This allows the {domain} engine to optimize the instruction set before execution. The garbage collector or memory manager must carefully track references related to {title} to prevent memory leaks, which is why understanding the lifecycle of {title} is non-negotiable for senior engineers. Furthermore, the interplay between {title} and the host operating system's thread pool dictates its concurrency model. In highly concurrent applications, improper synchronization around {title} can lead to race conditions or deadlocks. Therefore, advanced practitioners always implement {title} with immutability and thread-safety in mind."

    industry_standards = f"\n\n### Industry Standards and Compliance\nIn enterprise-grade {domain} architectures, {title} is subject to strict coding standards (e.g., POSIX, W3C, IEEE, or PEP 8). CI/CD pipelines are typically configured to enforce these standards through static analysis tools and linters. When a developer commits code involving {title}, the pipeline analyzes the cyclomatic complexity and abstract coupling. If the implementation of {title} is too tightly coupled to other modules, it fails the build. This ensures that the microservices or monolithic structures remain resilient. Furthermore, security audits often focus heavily on {title} because it represents a significant attack surface. Cross-Site Scripting (XSS), SQL Injection, and Buffer Overflows can sometimes manifest if {title} is used without proper input sanitization. Thus, the Zero Trust security model must be applied whenever {title} interacts with external data sources or user inputs."

    practical_example = ""
    safe_title = title.replace(' ', '').replace('-', '')
    if domain == "Web Development":
        practical_example = f"\n\n### Practical Example (Web)\n```javascript\n// Enterprise Implementation of {title}\nclass {safe_title}Manager {{\n  constructor(config) {{\n    this.config = config;\n    this.state = new Map(); // O(1) lookups for maximum performance\n  }}\n\n  async executeOperation(payload) {{\n    try {{\n      console.log(`[SYS] Starting isolated operation for {title}`);\n      const t0 = performance.now();\n      \n      // Simulated network/DOM interaction\n      const result = await fetch('/api/v1/resource', {{\n        method: 'POST',\n        headers: {{ 'Content-Type': 'application/json', 'X-Security-Token': 'v1-auth' }},\n        body: JSON.stringify(payload)\n      }});\n      \n      const data = await result.json();\n      const t1 = performance.now();\n      console.log(`[PERF] {title} executed in ${{t1 - t0}}ms`);\n      \n      return data;\n    }} catch (error) {{\n      console.error(`[FATAL] Error in {title} execution module:`, error);\n      // Implement exponential backoff or circuit breaker pattern here\n      throw new Error('Service Unavailable - Check Gateway Logs');\n    }}\n  }}\n}}\n```\nThis code demonstrates how to encapsulate {title} into a highly cohesive, loosely coupled class structure. It uses modern ES6+ syntax, asynchronous error handling, performance profiling, and memory-efficient Map data structures."
    elif domain == "Machine Learning":
         practical_example = f"\n\n### Practical Example (ML)\n```python\n# Mathematical & TensorFlow Implementation of {title}\nimport numpy as np\nimport tensorflow as tf\n\nclass {safe_title}Model(tf.keras.Model):\n    def __init__(self, units=128):\n        super().__init__()\n        # Dense layer with L2 regularization to prevent overfitting\n        self.dense1 = tf.keras.layers.Dense(\n            units, \n            activation='relu', \n            kernel_regularizer=tf.keras.regularizers.l2(0.01)\n        )\n        self.dropout = tf.keras.layers.Dropout(0.3)\n        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')\n\n    def call(self, inputs, training=False):\n        # Forward computational pass for {title}\n        x = self.dense1(inputs)\n        if training:\n            x = self.dropout(x, training=training)\n        return self.dense2(x)\n\n# Instantiate and compile with an advanced Adam optimizer\nmodel = {safe_title}Model()\nmodel.compile(\n    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), \n    loss='binary_crossentropy', \n    metrics=['accuracy', tf.keras.metrics.AUC()]\n)\n```\nThis TensorFlow snippet showcases a robust, production-ready implementation of {title}, featuring L2 regularization, dropout layers, and AUC metrics to prevent overfitting and ensure high generalizability on noisy, high-dimensional datasets."
    else:
        practical_example = f"\n\n### Practical Example ({domain})\n```typescript\n// Cross-platform mobile implementation of {title}\nimport React, {{ useEffect, useState }} from 'react';\nimport {{ View, Text, StyleSheet, Animated, InteractionManager }} from 'react-native';\n\nexport const {safe_title}Component = ({{ data }}: any) => {{\n  const [opacity] = useState(new Animated.Value(0));\n  const [isReady, setIsReady] = useState(false);\n\n  useEffect(() => {{\n    // Defer heavy execution of {title} until interactions/animations finish\n    InteractionManager.runAfterInteractions(() => {{\n      setIsReady(true);\n      // Hardware-accelerated animation for {title}\n      Animated.timing(opacity, {{\n        toValue: 1,\n        duration: 500,\n        useNativeDriver: true, // Offloads animation to native UI thread\n      }}).start();\n    }});\n  }}, []);\n\n  if (!isReady) return <View style={{styles.loader}} />;\n\n  return (\n    <Animated.View style={{[styles.container, {{ opacity }}]}}>\n      <Text style={{styles.text}}>Executing Native Module: {title}</Text>\n    </Animated.View>\n  );\n}};\n\nconst styles = StyleSheet.create({{\n  container: {{ flex: 1, padding: 16, backgroundColor: '#121212', borderRadius: 8 }},\n  text: {{ fontSize: 18, color: '#FFFFFF', fontWeight: '600', letterSpacing: 0.5 }},\n  loader: {{ flex: 1, backgroundColor: '#000000' }}\n}});\n```\nThis example highlights best practices for {title} on mobile devices, specifically utilizing `useNativeDriver` and `InteractionManager` to ensure operations run at a smooth 60 FPS without blocking the main JavaScript thread."
        
    conclusion = f"\n\n### Strategic Conclusion\nUltimately, mastering {title} separates junior developers from senior architects. The ability to reason about the time and space complexity of {title}, understand its position in the broader {domain} ecosystem, and implement it with rigorous security and performance standards is what drives technological innovation. Continuous benchmarking, peer code reviews, and staying updated with the latest RFCs are highly recommended to ensure that your usage of {title} remains optimal as project requirements scale to handle millions of active users."

    return base_content + theoretical_deep_dive + industry_standards + practical_example + conclusion

def _generate_smart_lesson(title: str, domain: str, description: str) -> dict:
    """Generates a lesson that feels real and specialized for its domain."""
    
    # Selection of domain knowledge generator
    gen_func = _get_web_content if domain == "Web Development" else \
               _get_ml_content if domain == "Machine Learning" else \
               _get_mobile_content if domain == "Mobile Development" else \
               None

    if gen_func:
        intro = description if description else gen_func(title, "intro")
        sections = [
            {"heading": f"1. Conceptual Overview", "content": _expand_content(title, domain, intro)},
            {"heading": f"2. The Evolution of {title}", "content": _expand_content(title, domain, gen_func(title, "history"))},
            {"heading": f"3. Technical Implementation & Logic", "content": _expand_content(title, domain, gen_func(title, "mechanics"))},
            {"heading": f"4. Future Horizon and Emerging Patterns", "content": _expand_content(title, domain, gen_func(title, "future"))},
            {"heading": f"5. Core Architecture", "content": _expand_content(title, domain, f"The architecture of {title} relies heavily on modularity and separation of concerns. In {domain}, this means understanding how {title} interacts with memory, network layers, and I/O operations. Proper architectural design ensures that {title} scales linearly with your application's growth.")},
            {"heading": f"6. Design Patterns", "content": _expand_content(title, domain, f"When building with {title}, experienced engineers utilize established design patterns. Whether it's Singleton, Factory, or Observer patterns, applying these to {title} reduces code duplication and improves maintainability within the {domain} ecosystem.")},
            {"heading": f"7. Performance Optimization", "content": _expand_content(title, domain, f"Optimizing {title} is critical. This involves profiling resource usage, minimizing overhead, and utilizing caching where appropriate. In {domain}, a poorly optimized implementation of {title} can lead to severe bottlenecks and degraded user experience.")},
            {"heading": f"8. Security Implications", "content": _expand_content(title, domain, f"Security cannot be an afterthought with {title}. You must sanitize inputs, manage permissions, and prevent common vulnerabilities. Understanding the attack surface of {title} is a mandatory requirement for any production-grade {domain} deployment.")},
            {"heading": f"9. Testing Strategies", "content": _expand_content(title, domain, f"Robust testing for {title} requires a mix of Unit, Integration, and End-to-End (E2E) tests. You should mock external dependencies and assert expected behaviors for edge cases. High test coverage on {title} ensures reliability across different environments.")},
            {"heading": f"10. Debugging & Troubleshooting", "content": _expand_content(title, domain, f"When {title} fails, debugging requires a systematic approach. Utilize logging, breakpoints, and stack trace analysis. Recognizing common error codes and failure states of {title} will drastically reduce your Mean Time To Resolution (MTTR).")},
            {"heading": f"11. Common Pitfalls & Anti-Patterns", "content": _expand_content(title, domain, f"A major anti-pattern when working with {title} is over-engineering. Developers often complicate {title} unnecessarily. Stick to the 'Keep It Simple, Stupid' (KISS) principle and avoid premature optimization in your {domain} projects.")},
            {"heading": f"12. Real-World Applications", "content": _expand_content(title, domain, f"In the industry, {title} is deployed in high-availability systems, processing massive amounts of data or serving millions of users. Tech giants leverage {title} to build resilient microservices, responsive UIs, and predictive algorithms.")},
            {"heading": f"13. Ecosystem & Tooling", "content": _expand_content(title, domain, f"The ecosystem surrounding {title} is vast. Leveraging the right IDE extensions, CLI tools, and third-party libraries can accelerate your workflow. However, be wary of 'dependency hell' when integrating too many tools with {title}.")},
            {"heading": f"14. Scalability Considerations", "content": _expand_content(title, domain, f"As your user base grows, {title} must scale horizontally or vertically. This involves load balancing, state management, and database sharding. Anticipating how {title} behaves under high load is a key aspect of senior-level engineering.")},
            {"heading": f"15. Interview Preparation", "content": _expand_content(title, domain, f"In technical interviews, expect to be asked about the trade-offs of using {title}. Interviewers will want to know how {title} compares to alternatives, its time/space complexity, and a scenario where you successfully implemented {title} in a previous role.")}
        ]
    else:
        # General fallback
        intro = description if description else f"Welcome to the module on {title}. This concept is a key part of your learning path."
        sections = [
            {"heading": f"1. Introduction to {title}", "content": _expand_content(title, domain, intro)},
            {"heading": f"2. Foundational Principles", "content": _expand_content(title, domain, f"The core principles of {title} involve understanding its basic structure and how it interacts with other components in the system. Mastering these fundamentals is essential.")},
            {"heading": f"3. Primary Use Cases", "content": _expand_content(title, domain, f"In practice, {title} is used to solve real-world problems by providing a structured approach to data and logic. It is widely adopted across various industries.")},
            {"heading": f"4. Setup and Configuration", "content": _expand_content(title, domain, f"Getting started with {title} requires setting up the appropriate environment. Ensure you have the latest dependencies and follow official documentation for installation.")},
            {"heading": f"5. Basic Syntax and Structure", "content": _expand_content(title, domain, f"Understanding the syntax of {title} is the first step towards mastery. Pay attention to naming conventions, keywords, and structural rules.")},
            {"heading": f"6. Intermediate Concepts", "content": _expand_content(title, domain, f"Once you grasp the basics, {title} offers intermediate features that allow for more complex operations, enabling greater flexibility and power.")},
            {"heading": f"7. Advanced Techniques", "content": _expand_content(title, domain, f"For power users, {title} provides advanced APIs and techniques. These are typically used in enterprise environments requiring high performance.")},
            {"heading": f"8. Integration Capabilities", "content": _expand_content(title, domain, f"A major strength of {title} is its ability to integrate with external systems, APIs, and databases, acting as a bridge between different technologies.")},
            {"heading": f"9. Performance Tuning", "content": _expand_content(title, domain, f"Tuning {title} for optimal performance requires an understanding of its internal execution model. This helps in reducing latency and resource consumption.")},
            {"heading": f"10. Security Best Practices", "content": _expand_content(title, domain, f"Always adhere to security best practices when using {title}. This involves validating inputs, encrypting sensitive data, and keeping dependencies updated.")},
            {"heading": f"11. Troubleshooting Common Errors", "content": _expand_content(title, domain, f"You will inevitably encounter errors with {title}. Learning how to read error logs and use debugging tools is a critical skill.")},
            {"heading": f"12. Version History and Updates", "content": _expand_content(title, domain, f"Technology moves fast. Keeping track of the version history of {title} helps you understand breaking changes and new feature additions.")},
            {"heading": f"13. Community and Support", "content": _expand_content(title, domain, f"The community around {title} is a valuable resource. Participate in forums, read open-source contributions, and collaborate with peers.")},
            {"heading": f"14. Career Opportunities", "content": _expand_content(title, domain, f"Proficiency in {title} is highly sought after by employers. It opens doors to roles in software engineering, data analysis, and systems architecture.")},
            {"heading": f"15. Further Reading", "content": _expand_content(title, domain, f"To deepen your knowledge of {title}, we recommend exploring official documentation, academic papers, and advanced video tutorials.")}
        ]

    return {
        "title": title,
        "color": "#6366f1",
        "icon": "📚",
        "introduction": intro,
        "sections": sections,
        "tip": f"Pro Tip: Always try to relate {title} to a project you are currently working on. Practical application is the best teacher!"
    }
