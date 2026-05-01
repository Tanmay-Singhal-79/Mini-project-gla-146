import os
import pandas as pd
import random

# Configuration
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "learning_data.csv")

# Core Curriculum Definitions (First 25 steps per domain)
CURRICULUM = {
    "Web Development": [
        ("Intro to HTML5", "Learn the skeleton of the web, doctypes, and the <html>, <head>, <body> structure."),
        ("Semantic HTML", "Using <header>, <footer>, <article>, and <section> for better SEO and accessibility."),
        ("HTML Forms & Validations", "Creating inputs, labels, and using native HTML5 validation like required and pattern."),
        ("HTML Media & Graphics", "Embedding images, videos, audio, and an introduction to <canvas> and <svg>."),
        ("HTML APIs", "Introduction to Geolocation, Drag & Drop, and Web Storage APIs (localStorage/sessionStorage)."),
        ("CSS Basics & Selectors", "Mastering colors, fonts, and the various ways to target elements."),
        ("CSS Box Model", "Understanding margin, border, padding, and how elements take up space."),
        ("CSS Positioning", "Mastering static, relative, absolute, fixed, and sticky positioning."),
        ("Flexbox Layout", "Building responsive rows and columns using display: flex and alignment properties."),
        ("CSS Grid Layout", "Creating complex 2D layouts with grid-template-areas and fractional units."),
        ("Responsive Web Design", "Using @media queries, viewport settings, and fluid layouts for all devices."),
        ("CSS Variables & Functions", "Advanced techniques like Custom Properties, calc(), min(), max(), and clamp()."),
        ("CSS Animations", "Creating smooth transitions and keyframe animations for better UX."),
        ("JS Intro & Syntax", "The basics of JavaScript: variables (let/const), statements, and basic operators."),
        ("JS Data Types", "Working with Strings, Numbers, Booleans, Null, Undefined, and BigInt."),
        ("JS Functions", "Defining reusable code blocks, arrow functions, and understanding parameters."),
        ("JS Objects", "Creating and manipulating objects, properties, and methods."),
        ("JS Events", "Handling user interaction with click, submit, mouseover, and keyboard events."),
        ("JS DOM Navigation", "Traversing the DOM tree using parentNode, children, and nextSibling."),
        ("JS DOM Modification", "Changing HTML content, attributes, and styles dynamically."),
        ("JS Arrays", "Mastering array methods like push, pop, shift, unshift, and length."),
        ("JS Array Iteration", "Using forEach, map, filter, reduce, and every/some for data processing."),
        ("JS Async & Await", "Handling asynchronous operations with Promises and the modern async/await syntax."),
        ("JS JSON & Fetch", "Fetching data from APIs and parsing JSON responses."),
        ("JS Web APIs", "Working with Fetch, History, and Web Workers for advanced performance."),
        ("React Intro & JSX", "Understanding the component-based architecture and JSX syntax."),
        ("React Props & State", "Managing data flow between components and local state with useState."),
        ("React Hooks (useEffect)", "Handling side effects, data fetching, and component lifecycle."),
        ("React Router", "Implementing client-side routing for multi-page feeling apps."),
        ("State Management (Redux)", "Centralized state management for complex applications."),
        ("Formik & Yup", "Advanced form handling and schema-based validation in React."),
        ("API Integration", "Connecting React apps to backend services using Axios and best practices."),
        ("Tailwind CSS Basics", "Utility-first CSS for rapid UI development."),
        ("Material UI Components", "Using pre-built component libraries for consistent design."),
        ("Next.js Fundamentals", "Server-side rendering, static site generation, and file-based routing."),
        ("Backend with Node.js", "Building servers with Express.js and handling HTTP requests."),
        ("MongoDB & Mongoose", "Working with NoSQL databases and defining data schemas."),
        ("JWT Authentication", "Securing applications with JSON Web Tokens and middleware."),
        ("Deployment (Vercel/Netlify)", "Deploying frontend applications and configuring CI/CD pipelines."),
        ("Web Security (CORS/XSS)", "Understanding and preventing common web vulnerabilities."),
    ],
    "Machine Learning": [
        ("Intro to AI/ML", "Understanding the difference between Artificial Intelligence and Machine Learning."),
        ("Machine Learning Pipeline", "The end-to-end process: collection, cleaning, processing, and model selection."),
        ("ML Data Types", "Understanding Numerical, Categorical, and Ordinal data types in ML context."),
        ("Basic Statistics for ML", "Calculating Mean, Median, Mode, Standard Deviation, and Percentiles."),
        ("Probability Distributions", "Normal distribution, Uniform distribution, and their importance in ML."),
        ("Data Preprocessing", "Handling missing values, scaling features (Standardization/Normalization)."),
        ("Linear Regression", "Predicting continuous values using best-fit lines and R-squared evaluation."),
        ("Multiple Regression", "Using multiple input features to predict a single continuous target."),
        ("Logistic Regression", "Binary classification for predicting categories like Yes/No using the sigmoid function."),
        ("Decision Trees", "Flowchart-like models for classification and regression tasks."),
        ("Random Forests", "Ensemble learning using multiple trees to reduce overfitting and improve accuracy."),
        ("K-Nearest Neighbors (KNN)", "Classifying data based on similarity to neighboring data points."),
        ("Support Vector Machines (SVM)", "Finding optimal hyperplanes to separate different classes in high-dimensional space."),
        ("Naïve Bayes", "Probabilistic classifier based on Bayes' Theorem with independence assumptions."),
        ("K-Means Clustering", "Unsupervised learning for grouping similar data points into K clusters."),
        ("Hierarchical Clustering", "Building a hierarchy of clusters using agglomerative or divisive methods."),
        ("Principal Component Analysis (PCA)", "Dimensionality reduction technique to simplify complex datasets."),
        ("Cross-Validation", "Using K-Fold and other techniques to ensure model stability and generalizability."),
        ("Model Evaluation Metrics", "Understanding Accuracy, Precision, Recall, F1-Score, and Confusion Matrices."),
        ("Neural Networks Intro", "The basics of artificial neurons, weights, biases, and activation functions."),
        ("Backpropagation", "How neural networks learn by calculating gradients and updating weights."),
        ("Deep Learning Basics", "Introduction to multi-layer perceptrons and deep architectures."),
        ("Convolutional Neural Nets (CNN)", "Specialized networks for image recognition and spatial data processing."),
        ("Recurrent Neural Nets (RNN)", "Sequence modeling for time-series data and natural language processing."),
        ("Natural Language Processing (NLP)", "Tokenization, lemmatization, and sentiment analysis techniques."),
        ("Word Embeddings", "Understanding Word2Vec, GloVe, and vector representations of text."),
        ("LSTMs & GRUs", "Advanced RNN architectures for long-term dependencies in sequences."),
        ("Computer Vision Basics", "Image processing techniques using OpenCV and scikit-image."),
        ("Transfer Learning", "Using pre-trained models like ResNet or VGG for custom tasks."),
        ("Generative AI Intro", "Understanding GANs and Variational Autoencoders (VAEs)."),
        ("Transformers Architecture", "The foundation of modern LLMs: Self-attention and Encoder-Decoder blocks."),
        ("BERT & Fine-tuning", "Using pre-trained Transformer models for NLP tasks."),
        ("Model Deployment", "Serving models via FastAPI, Flask, or TensorFlow Serving."),
        ("MLOps Basics", "Version control for data and models using DVC and MLflow."),
        ("Feature Engineering", "Creating new features from raw data to improve model performance."),
        ("Hyperparameter Tuning", "Optimizing model parameters using Grid Search and Random Search."),
        ("Reinforcement Learning", "Learning from environment feedback using Q-learning and Policy Gradients."),
        ("Time Series Analysis", "Predicting future trends from historical sequential data."),
        ("Recommender Systems", "Building engines for personalized suggestions based on user behavior."),
        ("AI Ethics & Fairness", "Identifying and mitigating bias in machine learning models."),
    ],
    "Mobile Development": [
        ("Intro to Mobile Dev", "Overview of native vs hybrid vs cross-platform app development."),
        ("UI/UX Mobile Principles", "Design patterns for mobile: touch targets, safe areas, and bottom navigation."),
        ("React Native Setup", "Setting up the development environment for iOS and Android."),
        ("JSX & Core Components", "Building screens with View, Text, Image, and ScrollView."),
        ("StyleSheet & Flexbox", "Mobile-specific layout and styling techniques."),
        ("Props & State", "Managing data flow and local component state with hooks."),
        ("TextInput & Forms", "Handling user input, keyboards, and form validation on mobile."),
        ("List Rendering", "Efficiently displaying data with FlatList and SectionList."),
        ("React Navigation", "Implementing Stack, Tab, and Drawer navigation patterns."),
        ("Button & Pressables", "Handling touch events and providing haptic feedback."),
        ("Icons & Images", "Using Vector Icons and optimized image assets."),
        ("Fetching Data", "Connecting to REST APIs using the Fetch API or Axios."),
        ("AsyncStorage", "Persisting small amounts of data locally on the device."),
        ("Device Features", "Accessing Camera, Gallery, and Location services."),
        ("Native Modules", "Introduction to bridging native code with JavaScript."),
        ("Animations in RN", "Using the Animated API for smooth transitions and interactions."),
        ("Debugging Mobile", "Using Chrome Debugger, React DevTools, and Flipper."),
        ("App Performance", "Optimizing renders, memoization, and image loading."),
        ("Dark Mode Support", "Implementing theme switching and dynamic color schemes."),
        ("Push Notifications", "Setting up local and remote notifications with Firebase."),
        ("App Icons & Splash", "Configuring platform-specific assets for app branding."),
        ("Building for Production", "Generating APKs and IPAs for testing and deployment."),
        ("App Store Submission", "Preparing metadata and assets for the Apple App Store."),
        ("Google Play Submission", "Publishing process for the Google Play Store."),
        ("Over-the-Air Updates", "Implementing OTA updates with services like Expo Updates."),
    ],
    "Blockchain": [
        ("Intro to Blockchain", "Understanding decentralization, blocks, hashes, and the basic architecture of a ledger."),
        ("Bitcoin & Digital Gold", "Analyzing the original whitepaper and the concept of peer-to-peer electronic cash."),
        ("Ethereum & Smart Contracts", "Introduction to programmable money and the Ethereum Virtual Machine (EVM)."),
        ("Solidity Fundamentals", "Learning the basics of Solidity: variables, types, and contract structure."),
        ("Web3.js & Ethers.js", "Connecting frontend applications to the blockchain using JavaScript libraries."),
        ("IPFS & Decentralized Storage", "Storing files off-chain using InterPlanetary File System."),
        ("DeFi & Liquidity Pools", "Understanding decentralized finance, swapping, and yield farming."),
        ("NFTs & ERC-721", "Building and deploying non-fungible tokens on Ethereum."),
        ("DAOs & Governance", "Creating decentralized autonomous organizations with voting mechanisms."),
        ("Layer 2 Scaling", "Understanding Optimism, Arbitrum, and Polygon for faster transactions."),
    ],
    "Cybersecurity": [
        ("Intro to Cybersecurity", "The CIA triad: Confidentiality, Integrity, and Availability."),
        ("Networking Basics", "Understanding TCP/IP, DNS, and how data moves across the internet."),
        ("Linux for Hackers", "Mastering the command line, permissions, and bash scripting."),
        ("Web Vulnerabilities (OWASP)", "Learning about SQL Injection, XSS, and Cross-Site Request Forgery."),
        ("Cryptography Basics", "Understanding symmetric vs asymmetric encryption and hashing."),
        ("Ethical Hacking Tools", "Introduction to Nmap, Metasploit, and Burp Suite."),
        ("Network Security", "Configuring firewalls, VPNs, and intrusion detection systems."),
        ("Social Engineering", "Understanding phishing, baiting, and psychological manipulation."),
        ("Incident Response", "How to react and recover from a security breach."),
        ("Cloud Security", "Securing AWS, Azure, and Google Cloud environments."),
    ],
    "Cloud Computing": [
        ("Intro to Cloud", "Understanding IaaS, PaaS, and SaaS models."),
        ("AWS Core Services", "Mastering EC2, S3, and RDS for basic cloud infrastructure."),
        ("Azure Fundamentals", "Introduction to Microsoft's cloud platform and services."),
        ("Google Cloud Platform", "Building and deploying apps on GCP."),
        ("Serverless Architecture", "Using AWS Lambda and Cloud Functions for event-driven computing."),
        ("Containers & Docker", "Packaging applications for consistent deployment across environments."),
        ("Kubernetes Orchestration", "Managing containerized applications at scale."),
        ("Cloud Storage Patterns", "Choosing between object, block, and file storage."),
        ("Identity & Access Management", "Managing users and permissions with IAM roles."),
        ("DevOps & CI/CD", "Automating deployments using cloud-native tools."),
    ],
    "Game Development": [
        ("Intro to Game Dev", "Understanding game loops, rendering, and physics engines."),
        ("C# for Unity", "Learning the primary language for professional game development."),
        ("Unity Interface", "Navigating the scene view, inspector, and project structure."),
        ("2D Game Mechanics", "Building platformers, sprites, and 2D physics."),
        ("3D Modeling & Texturing", "Basics of creating 3D assets for games."),
        ("Unreal Engine & C++", "High-performance game development with Unreal."),
        ("Game AI Basics", "Implementing pathfinding and state machines for NPCs."),
        ("Multiplayer Networking", "Synchronizing game state across multiple players."),
        ("Mobile Game Optimization", "Ensuring smooth performance on low-power devices."),
        ("Game Publishing", "Distributing games on Steam, itch.io, and app stores."),
    ],
    "Data Science": [
        ("Python for Data Science", "Mastering NumPy, Pandas, and Matplotlib for data manipulation."),
        ("Exploratory Data Analysis", "Identifying patterns and outliers in datasets using visualization."),
        ("Statistical Inference", "Hypothesis testing, p-values, and confidence intervals."),
        ("Data Wrangling", "Cleaning and transforming raw data into usable formats."),
        ("SQL for Data Science", "Querying complex databases and aggregating large datasets."),
        ("Predictive Modeling", "Using regression and classification to forecast outcomes."),
        ("Data Visualization", "Creating impactful dashboards with Tableau or PowerBI."),
        ("Big Data Intro", "Handling massive datasets with Spark and Hadoop."),
        ("Feature Selection", "Identifying the most important variables for your models."),
        ("Ethics in Data", "Privacy, bias, and the social impact of data-driven decisions."),
    ],
    "DevOps": [
        ("Intro to DevOps", "Understanding the culture of collaboration between Dev and Ops."),
        ("Version Control (Git)", "Mastering branching strategies and collaborative workflows."),
        ("CI/CD Pipelines", "Automating build, test, and deployment processes."),
        ("Infrastructure as Code", "Managing cloud resources using Terraform and CloudFormation."),
        ("Configuration Management", "Using Ansible and Chef to automate server setup."),
        ("Monitoring & Logging", "Tracking system health with Prometheus and ELK stack."),
        ("Container Security", "Hardening Docker images and Kubernetes clusters."),
        ("Cloud Native Patterns", "Designing systems for high availability and scalability."),
        ("SRE Principles", "Error budgets, SLIs, and SLOs for reliable systems."),
        ("Disaster Recovery", "Planning for and recovering from system failures."),
    ],
    "UI/UX Design": [
        ("Design Thinking", "Understanding empathy, ideation, and the user-centric process."),
        ("Wireframing Basics", "Creating low-fidelity sketches of application layouts."),
        ("Typography & Color", "Mastering visual hierarchy and emotional design."),
        ("Prototyping (Figma)", "Building interactive mockups for user testing."),
        ("User Research", "Conducting interviews and surveys to gather user insights."),
        ("Usability Testing", "Identifying friction points in your interface design."),
        ("Accessibility in UI", "Designing for users with varying abilities (WCAG)."),
        ("Design Systems", "Creating reusable component libraries for consistency."),
        ("Mobile vs Desktop UX", "Adapting designs for different screen sizes and contexts."),
        ("Interface Micro-interactions", "Adding delight with subtle animations and feedback."),
    ],
}

# Sub-topic banks for generating steps 26-100
GENERIC_SUBTOPICS = {
    "Web Development": [
        "WebSockets Real-time", "Progressive Web Apps (PWA)", "Server-Side Rendering (SSR)", "Static Site Generation (SSG)",
        "Web Accessibility (WCAG)", "BEM Methodology", "SASS/SCSS Advanced", "Tailwind CSS Workflow",
        "React Hooks (useEffect/useMemo)", "Redux Toolkit", "Context API Patterns", "Next.js App Router",
        "TypeScript Interfaces", "GraphQL Queries", "Apollo Client", "Node.js Express Middleware",
        "RESTful API Best Practices", "JWT Authentication", "OAuth2 & OpenID Connect", "Jest Unit Testing",
        "Cypress E2E Testing", "Docker Containerization", "CI/CD GitHub Actions", "Web Security (CORS/XSS)",
        "Service Workers Caching", "Web Components Custom Elements", "WebAssembly (WASM)", "Chrome DevTools Profiling",
        "Lighthouse Performance", "SEO Schema Markup", "Monorepo Management", "Micro-frontends Architecture",
        "Serverless Functions (AWS/Vercel)", "Edge Runtime", "Redis Caching", "PostgreSQL Optimization",
        "MongoDB Aggregations", "Prisma ORM", "Storybook Component Dev", "Framer Motion Animations"
    ],
    "Machine Learning": [
        "LSTMs & Sequential Models", "Transformers Architecture", "BERT & Fine-tuning", "GPT & Large Language Models",
        "Object Detection (YOLO/R-CNN)", "Image Segmentation (U-Net)", "Generative Adversarial Nets (GANs)", "Autoencoders for Denoising",
        "XGBoost & Gradient Boosting", "LightGBM High Performance", "Bayesian Optimization", "Hyperparameter Tuning (Optuna)",
        "Time Series Forecasting", "Anomaly Detection Patterns", "Reinforcement Learning (Q-Learning)", "Policy Gradient Methods",
        "MLOps MLflow Integration", "Model Deployment (FastAPI/Docker)", "TFX Pipelines", "TensorFlow Extended",
        "PyTorch Lightning", "Model Quantization", "Edge AI Deployment", "Federated Learning",
        "Explainable AI (SHAP/LIME)", "AI Ethics & Fairness", "Graph Neural Networks", "Recommender Systems (Collaborative Filtering)",
        "Content-Based Filtering", "Deep Q-Networks (DQN)", "Transfer Learning Strategies", "Active Learning",
        "Self-Supervised Learning", "Multi-task Learning", "Word Embeddings (FastText)", "Named Entity Recognition (NER)",
        "Speech Recognition Basics", "Computer Vision with OpenCV", "Audio Processing for ML", "ML Security & Adversarial Attacks"
    ],
    "Mobile Development": [
        "Native Performance", "Platform Channels", "State Management (Riverpod)", "Animations (Lottie)", "Background Tasks", 
        "Deep Linking", "Biometric Auth", "Payment Integration", "Offline Sync", "App Performance Profiling"
    ],
    "Blockchain": [
        "Zero-Knowledge Proofs", "Cross-chain Bridges", "MEV (Miner Extractable Value)", "Consensus Algorithms (PoS/PoW)",
        "Smart Contract Security", "Oracles (Chainlink)", "Stablecoin Mechanics", "DAO Governance Tokens", 
        "EVM Opcode Analysis", "zk-Rollups Architecture"
    ],
    "Cybersecurity": [
        "Pentesting Methodologies", "Malware Analysis", "Forensics Basics", "Kernel Exploitation",
        "Wireless Security", "Active Directory Hacking", "Bug Bounty Hunting", "Cloud Threat Modeling",
        "Zero Trust Architecture", "SOC Operations"
    ],
    "Cloud Computing": [
        "Multi-cloud Strategy", "Cloud Cost Optimization", "Infrastructure Compliance", "Disaster Recovery Testing",
        "Kubernetes Networking", "Service Mesh (Istio)", "Edge Computing Basics", "Hybrid Cloud Connectivity",
        "Global Traffic Management", "Cloud Data Governance"
    ],
    "Game Development": [
        "Shader Programming", "Physics Engine Tuning", "Procedural Content Generation", "Spatial Audio Design",
        "Vulkan API Intro", "Ray Tracing Implementation", "Game State Serialization", "Advanced Particle Systems",
        "VR Locomotion Patterns", "Anti-cheat Systems"
    ],
    "Data Science": [
        "Natural Language Understanding", "Time Series Decomposition", "Causal Inference", "Feature Store Implementation",
        "Model Drift Monitoring", "AB Testing at Scale", "Geospatial Data Analysis", "Reinforcement Learning for Business",
        "High-dimensional Data Visualization", "Pipeline Orchestration (Airflow)"
    ],
    "DevOps": [
        "GitOps Workflow", "Service Level Objectives", "Chaos Engineering", "Secret Management (Vault)",
        "Network Overlay Plugins", "Automated Canary Deployments", "Policy as Code (OPA)", "Blue-Green Strategies",
        "Observability Frameworks", "Infrastructure Drift Detection"
    ],
    "UI/UX Design": [
        "Motion Design Systems", "Voice User Interface (VUI)", "Cognitive Psychology in UX", "Advanced Auto Layout",
        "Design Handoff Optimization", "Inclusive Design Research", "AR Interface Patterns", "Data-driven Design Decisions",
        "Emotional Design Frameworks", "UX Writing Best Practices"
    ]
}

def generate():
    data = []
    
    # Standard interests to support
    interests = [
        "Web Development", "Machine Learning", "Mobile Development", 
        "Cloud Computing", "Cybersecurity", "Data Science", 
        "DevOps", "Game Development", "Blockchain", "UI/UX Design"
    ]
    
    for idx, interest in enumerate(interests):
        domain_id = idx
        
        # Get hardcoded steps
        base_steps = CURRICULUM.get(interest, [])
        
        # Fill steps 1 to 100
        for i in range(1, 101):
            if i <= len(base_steps):
                topic, desc = base_steps[i-1]
            else:
                # Generate realistic step name from banks or pattern
                bank = GENERIC_SUBTOPICS.get(interest, ["Advanced Concept", "Specialized Technique", "Best Practices", "Architectural Patterns", "Industry Case Study", "Scaling Systems", "Performance Tuning"])
                concept = bank[(i - len(base_steps) - 1) % len(bank)]
                
                # Create variation for the name to avoid exact duplicates
                cycle = (i - len(base_steps) - 1) // len(bank)
                if cycle == 0:
                    topic = concept
                elif cycle == 1:
                    topic = f"Deep Dive: {concept}"
                elif cycle == 2:
                    topic = f"Expert {concept} Patterns"
                else:
                    topic = f"Advanced {concept} Implementation"
                
                desc = f"Master {concept} to build robust, industry-level {interest} projects. This module covers advanced techniques and real-world deployment strategies. Level {cycle + 1} specialization."

            data.append({
                "interest": interest,
                "topic": topic,
                "difficulty": "Intermediate" if i > 35 else "Beginner" if i <= 15 else "Foundational",
                "learning_step_order": i,
                "learning_step": topic,
                "description": desc,
                "domain_id": domain_id
            })
            
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Dataset generated: {len(df)} records across {len(interests)} domains.")

if __name__ == "__main__":
    generate()
