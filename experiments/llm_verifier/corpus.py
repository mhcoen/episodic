"""
Synthetic corpus for LLM verifier experiment.

Creates controlled statement clusters designed to test verifier accuracy:
- Hard negatives (must be UNRELATED)
- Hard positives (must be RELATED even without lexical match)
- Near-miss distractors (tempting but should be UNRELATED)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"

# Seed for reproducibility
random.seed(42)

# =============================================================================
# SYNTHETIC STATEMENTS
# =============================================================================

# Format: (id, text, tags) where tags help us build query cases
STATEMENTS = [
    # --- PYTHON (language) cluster ---
    (1, "We discussed Python's GIL and how it affects multithreading. The Global Interpreter Lock prevents true parallel execution of Python bytecode, which is why CPU-bound tasks don't benefit from threading in Python.", ["python", "programming"]),
    (2, "I explained how to use Python list comprehensions for filtering data. The syntax [x for x in items if condition] is more Pythonic than using filter().", ["python", "programming"]),
    (3, "We debugged a Python ImportError together. The issue was a circular import between two modules that referenced each other.", ["python", "programming"]),
    (4, "I showed you how to use Python's asyncio for concurrent I/O operations. The async/await syntax makes it easy to write non-blocking code.", ["python", "programming"]),

    # --- COMPUTER (generic, NOT Python-specific) - HARD NEGATIVE for "python" ---
    (5, "We talked about how computers work at a fundamental level. The CPU fetches instructions from memory, decodes them, and executes them in a cycle.", ["computer", "hardware"]),
    (6, "I explained the difference between RAM and storage. RAM is volatile and fast, while SSDs and HDDs provide persistent but slower storage.", ["computer", "hardware"]),
    (7, "We discussed computer networking basics. Data is sent in packets through routers that forward them toward their destination.", ["computer", "networking"]),
    (8, "I described how operating systems manage resources. The kernel handles memory allocation, process scheduling, and device drivers.", ["computer", "os"]),

    # --- JAVA (language) cluster ---
    (9, "We discussed Java's garbage collection mechanisms. The JVM automatically reclaims memory from objects that are no longer reachable.", ["java", "programming"]),
    (10, "I explained Java interfaces and how they differ from abstract classes. Interfaces define contracts that implementing classes must fulfill.", ["java", "programming"]),
    (11, "We debugged a Java NullPointerException. The issue was accessing a method on an uninitialized object reference.", ["java", "programming"]),

    # --- COFFEE (beverage) - HARD NEGATIVE for "java" ---
    (12, "We talked about coffee brewing methods. Pour-over produces a cleaner cup than French press because the paper filter removes oils.", ["coffee", "beverage"]),
    (13, "I explained the difference between Arabica and Robusta beans. Arabica has more nuanced flavors while Robusta has more caffeine and bitterness.", ["coffee", "beverage"]),
    (14, "We discussed espresso extraction. The ideal shot takes 25-30 seconds and produces a rich crema on top.", ["coffee", "beverage"]),

    # --- APPLE (company) cluster ---
    (15, "We discussed Apple's M-series chips and their ARM architecture. The unified memory design gives significant performance benefits.", ["apple_company", "tech"]),
    (16, "I explained how to use Xcode for iOS development. The Interface Builder lets you design UIs visually.", ["apple_company", "programming"]),
    (17, "We talked about macOS system preferences and security settings. Gatekeeper controls which apps can run on your Mac.", ["apple_company", "os"]),

    # --- APPLE (fruit) - HARD NEGATIVE for "apple" (company) ---
    (18, "We discussed apple varieties and their characteristics. Honeycrisp apples are sweet and crunchy, while Granny Smith are tart and firm.", ["apple_fruit", "food"]),
    (19, "I explained how to make apple pie from scratch. The key is to use a mix of sweet and tart apples for balanced flavor.", ["apple_fruit", "food"]),
    (20, "We talked about growing apple trees. They need cross-pollination from a different variety to produce fruit.", ["apple_fruit", "gardening"]),

    # --- RUST (language) cluster ---
    (21, "We discussed Rust's ownership system and borrowing rules. The borrow checker prevents data races at compile time.", ["rust_lang", "programming"]),
    (22, "I explained Rust's Result and Option types for error handling. Using ? operator propagates errors elegantly.", ["rust_lang", "programming"]),
    (23, "We debugged a Rust lifetime error together. The issue was a reference outliving the data it pointed to.", ["rust_lang", "programming"]),

    # --- RUST (corrosion) - HARD NEGATIVE for "rust" (language) ---
    (24, "We talked about preventing rust on metal surfaces. Galvanization applies a zinc coating that corrodes sacrificially.", ["rust_corrosion", "chemistry"]),
    (25, "I explained the electrochemical process of rusting. Iron oxidizes in the presence of water and oxygen to form iron oxide.", ["rust_corrosion", "chemistry"]),
    (26, "We discussed rust removal techniques. Naval jelly contains phosphoric acid that converts rust to iron phosphate.", ["rust_corrosion", "diy"]),

    # --- MODEL (ML) cluster ---
    (27, "We discussed training neural network models. Gradient descent updates weights to minimize the loss function.", ["ml_model", "ai"]),
    (28, "I explained how transformer models work. Self-attention allows each token to attend to all other tokens in the sequence.", ["ml_model", "ai"]),
    (29, "We talked about model evaluation metrics. Precision measures false positives while recall measures false negatives.", ["ml_model", "ai"]),
    (30, "I showed you how to fine-tune a pretrained language model. LoRA adapters reduce the number of trainable parameters.", ["ml_model", "ai"]),

    # --- MODEL (fashion) - HARD NEGATIVE for "model" (ML) ---
    (31, "We talked about fashion modeling careers. Runway models typically need to be at least 5'9\" for women.", ["fashion_model", "career"]),
    (32, "I explained how fashion photoshoots work. The photographer directs poses while stylists handle wardrobe changes.", ["fashion_model", "photography"]),
    (33, "We discussed model portfolios and comp cards. A strong portfolio shows range across different looks and moods.", ["fashion_model", "career"]),

    # --- SQLITE cluster ---
    (34, "We discussed SQLite's file-based architecture. The entire database is stored in a single cross-platform file.", ["sqlite", "database"]),
    (35, "I explained SQLite's locking mechanism. It uses file-level locking which limits concurrent write access.", ["sqlite", "database"]),
    (36, "We debugged a SQLite database locked error. The issue was an uncommitted transaction holding a write lock.", ["sqlite", "database"]),
    (37, "I showed you how to use SQLite's FTS5 for full-text search. It creates inverted indexes for fast text queries.", ["sqlite", "database"]),

    # --- SQL (generic) - NEAR-MISS for "sqlite" ---
    (38, "We discussed SQL query optimization in general. Indexes speed up WHERE clauses but slow down INSERT operations.", ["sql", "database"]),
    (39, "I explained SQL JOIN types. INNER JOIN returns only matching rows while LEFT JOIN includes all rows from the left table.", ["sql", "database"]),
    (40, "We talked about SQL injection vulnerabilities. Parameterized queries prevent malicious input from being executed.", ["sql", "database"]),

    # --- WAKE WORD cluster ---
    (41, "We discussed wake word detection algorithms. They use small neural networks that run continuously on low-power DSPs.", ["wake_word", "voice"]),
    (42, "I explained how to train a custom wake word model. You need thousands of positive and negative audio samples.", ["wake_word", "voice"]),
    (43, "We debugged false wake word activations. The issue was acoustic similarity between the wake phrase and common speech.", ["wake_word", "voice"]),

    # --- SPEECH-TO-TEXT (generic) - NEAR-MISS for "wake word" ---
    (44, "We talked about speech recognition accuracy. Modern ASR systems use end-to-end neural models trained on thousands of hours.", ["speech", "voice"]),
    (45, "I explained how Whisper transcription works. It's a transformer model trained on diverse multilingual audio data.", ["speech", "voice"]),
    (46, "We discussed real-time transcription latency. Streaming ASR processes audio in chunks to minimize delay.", ["speech", "voice"]),

    # --- PHILOSOPHY cluster (abstract) ---
    (47, "We had a deep discussion about epistemology and justified true belief. The Gettier problem shows that JTB isn't sufficient for knowledge.", ["philosophy", "epistemology"]),
    (48, "I explained different ethical frameworks. Utilitarianism focuses on outcomes while deontology focuses on duties and rules.", ["philosophy", "ethics"]),
    (49, "We discussed the mind-body problem. Dualism posits separate mental and physical substances while physicalism denies this.", ["philosophy", "metaphysics"]),

    # --- AGNOSTICISM/BELIEF (should match "philosophy" as SUBSUMES) ---
    (50, "We talked about agnosticism as an epistemic position. It's the view that the existence of God is unknown or unknowable.", ["agnosticism", "belief"]),
    (51, "I explained the difference between knowledge and belief. Knowledge requires justification and truth, while belief doesn't.", ["epistemology", "belief"]),
    (52, "We discussed epistemic humility and intellectual honesty. Acknowledging uncertainty is epistemically virtuous.", ["epistemology", "belief"]),

    # --- MEMORY SYSTEM cluster ---
    (53, "We discussed how the memory system stores conversation history. Each exchange becomes a node in a directed acyclic graph.", ["memory_system", "architecture"]),
    (54, "I explained the retrieval mechanism for finding relevant memories. Vector embeddings enable semantic similarity search.", ["memory_system", "retrieval"]),
    (55, "We debugged a memory retrieval issue. The similarity threshold was too high, filtering out relevant results.", ["memory_system", "retrieval"]),

    # --- TOPIC SEGMENTATION cluster ---
    (56, "We discussed neural topic segmentation algorithms. TextTiling uses lexical cohesion to detect topic boundaries.", ["topic_segmentation", "nlp"]),
    (57, "I explained coherence-based topic detection. Segments with high internal coherence and low cross-segment coherence indicate boundaries.", ["topic_segmentation", "nlp"]),
    (58, "We talked about evaluating segmentation quality. WindowDiff and Pk are standard metrics for boundary detection.", ["topic_segmentation", "nlp"]),

    # --- PYTORCH cluster ---
    (59, "We discussed PyTorch's dynamic computation graph. Unlike static graphs, it builds the graph during forward pass execution.", ["pytorch", "ml"]),
    (60, "I explained PyTorch DataLoader for batching. It handles shuffling, batching, and parallel data loading automatically.", ["pytorch", "ml"]),
    (61, "We debugged a PyTorch CUDA out of memory error. The issue was accumulating gradients without calling optimizer.zero_grad().", ["pytorch", "ml"]),

    # --- DEEP LEARNING (generic) - NEAR-MISS for "pytorch" ---
    (62, "We talked about deep learning fundamentals. Neural networks learn hierarchical representations through backpropagation.", ["deep_learning", "ml"]),
    (63, "I explained convolutional neural networks. Conv layers detect local patterns that combine into higher-level features.", ["deep_learning", "ml"]),
    (64, "We discussed activation functions in neural networks. ReLU is popular because it avoids vanishing gradients.", ["deep_learning", "ml"]),

    # --- MCP RELAY cluster ---
    (65, "We discussed MCP relay for inter-process communication. It passes messages between Claude Desktop and Claude Code.", ["mcp_relay", "architecture"]),
    (66, "I explained how to configure MCP server endpoints. The relay routes requests to the appropriate handler.", ["mcp_relay", "architecture"]),
    (67, "We debugged an MCP relay connection timeout. The issue was the server not acknowledging the handshake.", ["mcp_relay", "architecture"]),

    # --- LINUX cluster ---
    (68, "We discussed Linux process management. The init system (systemd) manages service lifecycles and dependencies.", ["linux", "os"]),
    (69, "I explained Linux file permissions. The rwx bits control read, write, and execute access for owner, group, and others.", ["linux", "os"]),
    (70, "We talked about Linux shell scripting. Bash scripts automate repetitive tasks with conditionals and loops.", ["linux", "os"]),

    # --- UNIX (generic) - NEAR-MISS for "linux" ---
    (71, "We discussed Unix philosophy and design principles. Small, composable tools that do one thing well.", ["unix", "os"]),
    (72, "I explained Unix pipes and redirection. The pipe operator connects stdout of one process to stdin of another.", ["unix", "os"]),
    (73, "We talked about POSIX standards for Unix compatibility. It defines interfaces that portable programs should use.", ["unix", "os"]),

    # --- RELIGION cluster (abstract) ---
    (74, "We discussed comparative religion and different faith traditions. Buddhism, Christianity, Islam, and Hinduism have distinct metaphysics.", ["religion", "belief"]),
    (75, "I explained theological arguments for God's existence. The cosmological argument posits a first cause.", ["religion", "theology"]),
    (76, "We talked about religious experience and mysticism. Many traditions describe direct encounters with the divine.", ["religion", "spirituality"]),

    # --- POLITICS cluster (abstract) ---
    (77, "We discussed different political ideologies. Liberalism emphasizes individual rights while socialism emphasizes collective ownership.", ["politics", "ideology"]),
    (78, "I explained how democratic institutions work. Separation of powers prevents concentration of authority.", ["politics", "governance"]),
    (79, "We talked about political polarization and tribalism. Social media creates filter bubbles that reinforce existing views.", ["politics", "society"]),

    # --- EMBEDDING cluster ---
    (80, "We discussed word embeddings and semantic similarity. Word2Vec learns vector representations from co-occurrence patterns.", ["embedding", "nlp"]),
    (81, "I explained sentence embedding models. SBERT creates fixed-size vectors that capture sentence meaning.", ["embedding", "nlp"]),
    (82, "We talked about embedding similarity thresholds. Cosine similarity above 0.8 typically indicates strong semantic match.", ["embedding", "nlp"]),

    # --- CHROMADB cluster ---
    (83, "We discussed ChromaDB for vector storage. It's an open-source embedding database optimized for similarity search.", ["chromadb", "database"]),
    (84, "I explained ChromaDB collection management. Collections group related embeddings with optional metadata filtering.", ["chromadb", "database"]),
    (85, "We debugged a ChromaDB query returning no results. The issue was a mismatched embedding dimension.", ["chromadb", "database"]),

    # --- VECTOR DATABASE (generic) - NEAR-MISS for "chromadb" ---
    (86, "We talked about vector database architectures. They use approximate nearest neighbor algorithms like HNSW for fast search.", ["vector_db", "database"]),
    (87, "I explained different ANN indexing strategies. IVF partitions the space while HNSW builds a navigable graph.", ["vector_db", "database"]),
    (88, "We discussed vector database scaling. Sharding distributes vectors across nodes for horizontal scaling.", ["vector_db", "database"]),

    # --- ADDITIONAL FILLER for realistic retrieval mix ---
    (89, "We discussed markdown formatting for documentation. Headers, lists, and code blocks structure content clearly.", ["markdown", "docs"]),
    (90, "I explained git branching strategies. Feature branches keep work isolated until ready to merge.", ["git", "version_control"]),
    (91, "We talked about API design best practices. REST uses HTTP verbs to indicate operations on resources.", ["api", "design"]),
    (92, "I showed you how to write unit tests. Tests should be isolated, fast, and cover edge cases.", ["testing", "practices"]),
    (93, "We discussed code review practices. Reviews catch bugs early and spread knowledge across the team.", ["code_review", "practices"]),
    (94, "I explained dependency injection for testability. It decouples components so you can substitute mocks.", ["testing", "design"]),
    (95, "We talked about continuous integration pipelines. CI runs tests automatically on every commit.", ["ci", "devops"]),
    (96, "I described Docker containerization. Containers package applications with their dependencies for consistent deployment.", ["docker", "devops"]),
    (97, "We discussed Kubernetes orchestration. K8s manages container deployment, scaling, and networking.", ["kubernetes", "devops"]),
    (98, "I explained microservices architecture. Small, independent services communicate over well-defined APIs.", ["microservices", "architecture"]),
    (99, "We talked about database normalization. Third normal form eliminates transitive dependencies.", ["database", "design"]),
    (100, "I showed you how to profile Python code. cProfile identifies which functions consume the most time.", ["profiling", "python"]),

    # --- MORE HARD NEGATIVES ---
    # SHELL vs BASH
    (101, "We discussed seashells and marine biology. Mollusks build their shells from calcium carbonate secretions.", ["shell_marine", "biology"]),
    (102, "I explained how shells protect marine organisms. The spiral shape of snail shells follows the Fibonacci sequence.", ["shell_marine", "biology"]),

    # KERNEL vs LINUX KERNEL
    (103, "We talked about popcorn kernels and how they pop. Moisture inside the kernel turns to steam and bursts the hull.", ["kernel_food", "food"]),
    (104, "I explained corn kernel anatomy. The endosperm contains starch while the germ holds the embryo.", ["kernel_food", "food"]),

    # BRANCH vs GIT BRANCH
    (105, "We discussed tree pruning and branch management. Removing dead branches improves tree health and appearance.", ["branch_tree", "gardening"]),
    (106, "I explained how tree branches grow toward light. Phototropism causes stems to bend toward the sun.", ["branch_tree", "biology"]),

    # --- MORE ABSTRACT POSITIVES ---
    # Should match "epistemology" or "philosophy"
    (107, "We had a debate about whether we can ever truly know anything. Radical skepticism questions even basic perceptions.", ["skepticism", "philosophy"]),
    (108, "I explained foundationalism versus coherentism in justification theory. Do beliefs need basic foundations or just mutual support?", ["justification", "philosophy"]),

    # Should match "ethics" or "philosophy"
    (109, "We discussed trolley problems and moral intuitions. These thought experiments test our ethical reasoning.", ["ethics", "philosophy"]),
    (110, "I explained virtue ethics and character development. Aristotle argued that virtues are habits cultivated through practice.", ["ethics", "philosophy"]),

    # --- PROGRAMMING LANGUAGE CLUSTER (generic) ---
    (111, "We discussed programming language paradigms. Functional, object-oriented, and procedural are major approaches.", ["programming", "languages"]),
    (112, "I explained static versus dynamic typing. Static typing catches errors at compile time while dynamic typing defers to runtime.", ["programming", "languages"]),
    (113, "We talked about memory management strategies. Garbage collection automates cleanup while manual management gives control.", ["programming", "memory"]),

    # --- TENSORFLOW cluster (for contrast with PyTorch) ---
    (114, "We discussed TensorFlow's static computation graphs. The graph is defined first, then executed in a session.", ["tensorflow", "ml"]),
    (115, "I explained TensorFlow Serving for model deployment. It handles model versioning and request batching.", ["tensorflow", "ml"]),
    (116, "We debugged a TensorFlow shape mismatch error. The issue was incompatible tensor dimensions in a matrix multiply.", ["tensorflow", "ml"]),

    # --- DATABASE (generic) cluster ---
    (117, "We discussed ACID properties in databases. Atomicity, consistency, isolation, and durability guarantee reliable transactions.", ["database", "theory"]),
    (118, "I explained database indexing strategies. B-tree indexes are efficient for range queries while hash indexes excel at equality lookups.", ["database", "theory"]),
    (119, "We talked about database replication for high availability. Primary-replica setups allow reads from replicas while writes go to primary.", ["database", "theory"]),

    # --- ADDITIONAL STATEMENTS for density ---
    (120, "We discussed HTTP status codes and their meanings. 200 means success, 404 means not found, 500 means server error.", ["http", "web"]),
    (121, "I explained OAuth 2.0 authentication flows. The authorization code flow is most secure for server-side apps.", ["auth", "security"]),
    (122, "We talked about JWT tokens for stateless auth. The payload contains claims that the server can verify without database lookup.", ["auth", "security"]),
    (123, "I showed you how to use regex for pattern matching. The syntax is powerful but can be hard to read.", ["regex", "programming"]),
    (124, "We discussed async/await patterns for concurrency. It makes asynchronous code read like synchronous code.", ["async", "programming"]),
    (125, "I explained WebSocket connections for real-time communication. Unlike HTTP, WebSockets maintain persistent bidirectional channels.", ["websocket", "web"]),
    (126, "We talked about load balancing strategies. Round-robin distributes requests evenly while least-connections optimizes for busy servers.", ["load_balancing", "devops"]),
    (127, "I described message queues for decoupling services. RabbitMQ and Kafka handle asynchronous communication between components.", ["messaging", "architecture"]),
    (128, "We discussed caching strategies for performance. Redis and Memcached store frequently accessed data in memory.", ["caching", "performance"]),
    (129, "I explained rate limiting for API protection. Token bucket and leaky bucket algorithms control request throughput.", ["rate_limiting", "security"]),
    (130, "We talked about logging best practices. Structured logs with correlation IDs help trace requests across services.", ["logging", "observability"]),
]


# =============================================================================
# QUERY CASES with deterministic candidate lists and gold labels
# =============================================================================

def build_query_cases():
    """
    Build query cases with controlled candidate lists.
    Each case has true positives, hard negatives, and random fillers.
    """
    cases = []

    # Helper to get statement IDs by tag
    def by_tag(tag):
        return [s[0] for s in STATEMENTS if tag in s[2]]

    def by_tags(tags):
        return [s[0] for s in STATEMENTS if any(t in s[2] for t in tags)]

    all_ids = [s[0] for s in STATEMENTS]

    # --- HARD NEGATIVE CASES ---

    # python (language) vs computer (generic)
    cases.append({
        "query": "python",
        "description": "Python programming language - must NOT match generic computer talk",
        "candidates": by_tag("python") + by_tag("computer") + by_tag("programming")[:4] + random.sample([i for i in all_ids if i not in by_tag("python") + by_tag("computer")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("python")} | {str(id): 0 for id in by_tag("computer")},
        "hard_negatives": by_tag("computer"),
    })

    # java (language) vs coffee
    cases.append({
        "query": "java",
        "description": "Java programming language - must NOT match coffee discussion",
        "candidates": by_tag("java") + by_tag("coffee") + by_tag("programming")[:4] + random.sample([i for i in all_ids if i not in by_tag("java") + by_tag("coffee")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("java")} | {str(id): 0 for id in by_tag("coffee")},
        "hard_negatives": by_tag("coffee"),
    })

    # apple (company) vs apple (fruit) - make query unambiguous
    cases.append({
        "query": "Apple computers",
        "description": "Apple the tech company - must NOT match apple fruit discussion",
        "candidates": by_tag("apple_company") + by_tag("apple_fruit") + by_tag("tech")[:3] + random.sample([i for i in all_ids if i not in by_tag("apple_company") + by_tag("apple_fruit")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("apple_company")} | {str(id): 0 for id in by_tag("apple_fruit")},
        "hard_negatives": by_tag("apple_fruit"),
    })

    # rust (language) vs rust (corrosion)
    cases.append({
        "query": "rust",
        "description": "Rust programming language - must NOT match corrosion discussion",
        "candidates": by_tag("rust_lang") + by_tag("rust_corrosion") + by_tag("programming")[:4] + random.sample([i for i in all_ids if i not in by_tag("rust_lang") + by_tag("rust_corrosion")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("rust_lang")} | {str(id): 0 for id in by_tag("rust_corrosion")},
        "hard_negatives": by_tag("rust_corrosion"),
    })

    # model (ML) vs model (fashion)
    cases.append({
        "query": "model",
        "description": "ML/AI models - must NOT match fashion model discussion",
        "candidates": by_tag("ml_model") + by_tag("fashion_model") + by_tag("ai")[:2] + random.sample([i for i in all_ids if i not in by_tag("ml_model") + by_tag("fashion_model")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("ml_model")} | {str(id): 0 for id in by_tag("fashion_model")},
        "hard_negatives": by_tag("fashion_model"),
    })

    # --- NEAR-MISS CASES ---

    # sqlite vs generic SQL
    cases.append({
        "query": "sqlite",
        "description": "SQLite specifically - generic SQL discussion is a near-miss",
        "candidates": by_tag("sqlite") + by_tag("sql") + by_tag("database")[:4] + random.sample([i for i in all_ids if i not in by_tag("sqlite") + by_tag("sql")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("sqlite")} | {str(id): 0 for id in by_tag("sql")},
        "hard_negatives": by_tag("sql"),
    })

    # wake word vs generic speech
    cases.append({
        "query": "wake word",
        "description": "Wake word detection specifically - generic speech recognition is near-miss",
        "candidates": by_tag("wake_word") + by_tag("speech") + by_tag("voice")[:2] + random.sample([i for i in all_ids if i not in by_tag("wake_word") + by_tag("speech")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("wake_word")} | {str(id): 0 for id in by_tag("speech")},
        "hard_negatives": by_tag("speech"),
    })

    # pytorch vs generic deep learning
    cases.append({
        "query": "pytorch",
        "description": "PyTorch specifically - generic deep learning is near-miss",
        "candidates": by_tag("pytorch") + by_tag("deep_learning") + by_tag("ml")[:4] + random.sample([i for i in all_ids if i not in by_tag("pytorch") + by_tag("deep_learning")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("pytorch")} | {str(id): 0 for id in by_tag("deep_learning")},
        "hard_negatives": by_tag("deep_learning"),
    })

    # linux vs generic unix - Unix is actually related (Linux is Unix-like)
    # Keep as near-miss test but expect some overlap to be accepted
    cases.append({
        "query": "Linux kernel",
        "description": "Linux kernel specifically - generic Unix philosophy is near-miss",
        "candidates": by_tag("linux") + by_tag("unix") + by_tag("os")[:4] + random.sample([i for i in all_ids if i not in by_tag("linux") + by_tag("unix")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("linux")} | {str(id): 0 for id in by_tag("unix")},
        "hard_negatives": by_tag("unix"),
    })

    # chromadb vs generic vector db
    cases.append({
        "query": "chromadb",
        "description": "ChromaDB specifically - generic vector DB discussion is near-miss",
        "candidates": by_tag("chromadb") + by_tag("vector_db") + by_tag("database")[:4] + random.sample([i for i in all_ids if i not in by_tag("chromadb") + by_tag("vector_db")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("chromadb")} | {str(id): 0 for id in by_tag("vector_db")},
        "hard_negatives": by_tag("vector_db"),
    })

    # --- ABSTRACT/SEMANTIC MATCH CASES ---

    # philosophy should match epistemology discussions
    cases.append({
        "query": "philosophy",
        "description": "Philosophy - should match epistemology, ethics, agnosticism discussions",
        "candidates": by_tag("philosophy") + by_tags(["epistemology", "agnosticism", "ethics", "belief", "skepticism", "justification"]) + random.sample([i for i in all_ids if i not in by_tags(["philosophy", "epistemology", "agnosticism", "ethics", "belief", "skepticism", "justification"])], 25),
        "gold_relevant": {str(id): 1 for id in by_tags(["philosophy", "epistemology", "agnosticism", "ethics", "belief", "skepticism", "justification"])},
        "hard_negatives": [],
    })

    # epistemology should match related discussions
    cases.append({
        "query": "epistemology",
        "description": "Epistemology - knowledge/belief discussions",
        "candidates": by_tags(["epistemology", "belief", "skepticism", "justification"]) + by_tag("philosophy") + random.sample([i for i in all_ids if i not in by_tags(["epistemology", "belief", "philosophy", "skepticism", "justification"])], 30),
        "gold_relevant": {str(id): 1 for id in by_tags(["epistemology", "belief", "skepticism", "justification", "philosophy"])},
        "hard_negatives": [],
    })

    # --- SPECIFIC TECHNICAL QUERIES ---

    # memory system
    cases.append({
        "query": "memory system",
        "description": "Episodic memory system architecture",
        "candidates": by_tag("memory_system") + by_tag("retrieval")[:2] + random.sample([i for i in all_ids if i not in by_tag("memory_system")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("memory_system")},
        "hard_negatives": [],
    })

    # topic segmentation
    cases.append({
        "query": "topic segmentation",
        "description": "Topic/dialogue segmentation algorithms",
        "candidates": by_tag("topic_segmentation") + by_tag("nlp")[:3] + random.sample([i for i in all_ids if i not in by_tag("topic_segmentation")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("topic_segmentation")},
        "hard_negatives": [],
    })

    # mcp relay
    cases.append({
        "query": "mcp relay",
        "description": "MCP relay for inter-process communication",
        "candidates": by_tag("mcp_relay") + by_tag("architecture")[:3] + random.sample([i for i in all_ids if i not in by_tag("mcp_relay")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("mcp_relay")},
        "hard_negatives": [],
    })

    # embeddings
    cases.append({
        "query": "embeddings",
        "description": "Word/sentence embeddings",
        "candidates": by_tag("embedding") + by_tag("nlp")[:3] + random.sample([i for i in all_ids if i not in by_tag("embedding")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("embedding")},
        "hard_negatives": [],
    })

    # --- MORE POLYSEMY/CONFUSABLE CASES ---

    # shell (bash) vs shell (marine) - using linux/unix as proxy for bash
    cases.append({
        "query": "shell",
        "description": "Command shell (bash/zsh) - must NOT match seashells",
        "candidates": by_tag("linux") + by_tag("unix") + by_tag("shell_marine") + random.sample([i for i in all_ids if i not in by_tag("linux") + by_tag("unix") + by_tag("shell_marine")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("linux") + by_tag("unix")} | {str(id): 0 for id in by_tag("shell_marine")},
        "hard_negatives": by_tag("shell_marine"),
    })

    # kernel (linux) vs kernel (food)
    cases.append({
        "query": "kernel",
        "description": "OS kernel - must NOT match corn kernels",
        "candidates": by_tag("linux") + by_tag("os")[:3] + by_tag("kernel_food") + random.sample([i for i in all_ids if i not in by_tag("linux") + by_tag("kernel_food")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("linux")} | {str(id): 0 for id in by_tag("kernel_food")},
        "hard_negatives": by_tag("kernel_food"),
    })

    # branch (git) vs branch (tree) - make query unambiguous
    cases.append({
        "query": "git branch",
        "description": "Git branches - must NOT match tree branches",
        "candidates": by_tag("git") + by_tag("version_control")[:2] + by_tag("branch_tree") + random.sample([i for i in all_ids if i not in by_tag("git") + by_tag("branch_tree")], 30),
        "gold_relevant": {str(id): 1 for id in by_tag("git")} | {str(id): 0 for id in by_tag("branch_tree")},
        "hard_negatives": by_tag("branch_tree"),
    })

    # --- ADDITIONAL TECHNICAL QUERIES ---

    # tensorflow
    cases.append({
        "query": "tensorflow",
        "description": "TensorFlow ML framework",
        "candidates": by_tag("tensorflow") + by_tag("ml")[:4] + random.sample([i for i in all_ids if i not in by_tag("tensorflow")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("tensorflow")},
        "hard_negatives": [],
    })

    # database
    cases.append({
        "query": "database",
        "description": "Database systems in general",
        "candidates": by_tags(["database", "sqlite", "sql", "chromadb", "vector_db"]) + random.sample([i for i in all_ids if i not in by_tags(["database", "sqlite", "sql", "chromadb", "vector_db"])], 25),
        "gold_relevant": {str(id): 1 for id in by_tags(["database", "sqlite", "sql", "chromadb", "vector_db"])},
        "hard_negatives": [],
    })

    # religion
    cases.append({
        "query": "religion",
        "description": "Religion and theology",
        "candidates": by_tags(["religion", "theology", "spirituality"]) + by_tag("belief")[:2] + random.sample([i for i in all_ids if i not in by_tags(["religion", "theology", "spirituality", "belief"])], 30),
        "gold_relevant": {str(id): 1 for id in by_tags(["religion", "theology", "spirituality"])},
        "hard_negatives": [],
    })

    # politics
    cases.append({
        "query": "politics",
        "description": "Politics and governance",
        "candidates": by_tags(["politics", "ideology", "governance", "society"]) + random.sample([i for i in all_ids if i not in by_tags(["politics", "ideology", "governance", "society"])], 30),
        "gold_relevant": {str(id): 1 for id in by_tags(["politics", "ideology", "governance", "society"])},
        "hard_negatives": [],
    })

    # docker
    cases.append({
        "query": "docker",
        "description": "Docker containerization",
        "candidates": by_tag("docker") + by_tag("devops")[:4] + random.sample([i for i in all_ids if i not in by_tag("docker")], 35),
        "gold_relevant": {str(id): 1 for id in by_tag("docker")},
        "hard_negatives": [],
    })

    # Dedupe candidates and ensure max 50 per case
    for case in cases:
        # Dedupe while preserving order
        seen = set()
        deduped = []
        for cid in case["candidates"]:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        case["candidates"] = deduped[:50]  # Cap at 50

    return cases


def create_database():
    """Create the synthetic database with statements."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE statements (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            text TEXT
        )
    """)

    base_time = datetime(2024, 1, 1, 10, 0, 0)
    for stmt_id, text, tags in STATEMENTS:
        created_at = (base_time + timedelta(hours=stmt_id)).isoformat()
        cursor.execute(
            "INSERT INTO statements (id, created_at, text) VALUES (?, ?, ?)",
            (stmt_id, created_at, text)
        )

    conn.commit()
    conn.close()
    print(f"Created database with {len(STATEMENTS)} statements at {DB_PATH}")


def create_query_cases():
    """Create the query cases JSON file."""
    cases = build_query_cases()

    with open(QUERY_CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Created {len(cases)} query cases at {QUERY_CASES_PATH}")

    # Summary stats
    total_hard_negatives = sum(len(c.get("hard_negatives", [])) for c in cases)
    total_candidates = sum(len(c["candidates"]) for c in cases)
    print(f"  Total candidates across all queries: {total_candidates}")
    print(f"  Total hard negatives to test: {total_hard_negatives}")


if __name__ == "__main__":
    create_database()
    create_query_cases()
