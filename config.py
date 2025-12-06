# ============================================
# config.py — Global config & curriculum v2
# ============================================

from pathlib import Path

# ---- PROJECT ROOT ----
PROJECT_ROOT = Path(r"D:\Projects\skill-extraction")

# ---- MODEL / DATA PATHS ----
# Base model for fine-tuning (JobBERT)
JOBBERT_MODEL_NAME = "jjzha/jobbert-base-cased"  # or any BERT-like model

# Where to save the fine-tuned multi-task model
# MULTITASK_MODEL_DIR = PROJECT_ROOT / "models" / "jobbert_research_v1"

MULTITASK_MODEL_DIR = PROJECT_ROOT / "baseline_versions" / "jobbert_crf" / "outputs" / "multitask_model"

# Where your SkillSpan / NER-style data lives
# You will need to adapt to your actual files later.
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.json"      # TODO: adjust
VAL_PATH   = DATA_DIR / "dev.json"        # TODO: adjust
TEST_PATH  = DATA_DIR / "test.json"       # TODO: adjust

# Real-world job postings (one sentence per line OR JSON; we handle both later)
REAL_WORLD_JOBS_PATH = DATA_DIR / "real_job_sentences.txt"  # TODO: adjust

# Output directory for v2 analysis
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- LABELS ----
# We assume simple BIO: O, B, I for both skills and knowledge.
# If your SkillSpan encoding differs, change these here.
SKILL_LABEL_LIST = ["O", "B", "I"]
KNOWLEDGE_LABEL_LIST = ["O", "B", "I"]

SKILL_LABEL2ID = {lbl: i for i, lbl in enumerate(SKILL_LABEL_LIST)}
SKILL_ID2LABEL = {i: lbl for lbl, i in SKILL_LABEL2ID.items()}

KNOWLEDGE_LABEL2ID = {lbl: i for i, lbl in enumerate(KNOWLEDGE_LABEL_LIST)}
KNOWLEDGE_ID2LABEL = {i: lbl for lbl, i in KNOWLEDGE_LABEL2ID.items()}

# ---- HUGGING FACE / LLAMA ----
# You can export env variable:
#   set HF_API_TOKEN=your_token_here
# or put your token here (less secure).
import os

from pathlib import Path

# Read token from file (once, at import time)
HF_API_TOKEN = Path("api_keys/hf_token.txt").read_text(encoding="utf-8").strip()


# ---- CURRICULUM COMPONENTS (SHORTENED STUB) ----
# # Replace with your full 19-component dict later.
# CURRICULUM_COMPONENTS = {
#     # 1. Berpikir Komputasional / Computational Thinking
#     "computational_thinking": {
#         "understand": [
#             "data structure concepts",
#             "standard algorithms",
#             "von neumann architecture",
#             "operating system roles",
#             "computational models",
#             "input-process-output",
#             "computer architecture",
#             "memory management",
#             "process scheduling",
#             "binary logic",
#             "system logic",
#             "data structures",
#             "standard data structures",
#             "algorithmic thinking",
#             "problem decomposition",
#             "abstraction",
#             "pattern recognition",
#             "control structures",
#             "von neumann model",
#             "von neumann computer",
#             "input process output",
#             "IPO model",
#             "operating system concepts",
#             "operating system fundamentals"
#         ],
#         "apply": [
#             "computational process",
#             "pseudocode",
#             "simulation",
#             "independent computation",
#             "group computation",
#             "algorithm execution",
#             "flowcharting",
#             "command line",
#             "terminal usage",
#             "bash scripting",
#             "shell scripting",
#             "automating tasks",
#             "implement data structures",
#             "implement algorithms",
#             "apply standard algorithms",
#             "write pseudocode",
#             "design pseudocode",
#             "simulate algorithms",
#             "trace algorithms",
#             "debug algorithms"
#         ],
#         "analyze": [
#             "analyze computational problems",
#             "analyze problem complexity",
#             "predict outcomes",
#             "evaluate algorithm choices",
#             "optimize algorithms",
#             "choose appropriate data structures"
#         ],
#         "create": [
#             "program design",
#             "solution design",
#             "problem solving strategy",
#             "system architecture design",
#             "technical specification",
#             "workflow design",
#             "logic design"
#         ]
#     },

#     # 2. Literasi Digital / Digital Literacy
#     "digital_literacy": {
#         "understand": [
#             "search engine mechanisms",
#             "information ecosystem",
#             "digital technology devices",
#             "cloud connectivity",
#             "internet of things",
#             "iot",
#             "saas",
#             "paas",
#             "iaas",
#             "cloud storage concepts",
#             "search engine usage",
#             "advanced search",
#             "search operators",
#             "fact-checking ecosystem",
#             "fake news detection",
#             "digital citizenship",
#             "basic network security",
#             "wifi security",
#             "encryption basics"
#         ],
#         "apply": [
#             "advanced search techniques",
#             "document processing",
#             "spreadsheet utilization",
#             "presentation software",
#             "password management",
#             "two-factor authentication",
#             "privacy configuration",
#             "microsoft office",
#             "excel",
#             "word",
#             "powerpoint",
#             "google workspace",
#             "collaboration tools",
#             "slack",
#             "trello",
#             "jira",
#             "zoom",
#             "teams",
#             "use search engines",
#             "use spreadsheets",
#             "use presentation tools",
#             "word processing",
#             "online collaboration",
#             "manage passwords",
#             "configure wifi",
#             "configure privacy settings"
#         ],
#         "evaluate": [
#             "fact checking",
#             "lateral reading",
#             "information evaluation",
#             "filtering negative content",
#             "data privacy",
#             "gdpr compliance",
#             "phishing awareness",
#             "cyber hygiene",
#             "information security basics",
#             "digital footprint analysis",
#             "evaluate online information",
#             "detect misinformation",
#             "assess credibility of sources",
#             "evaluate digital sources",
#             "verify information",
#             "cross-check information"
#         ],
#         "create": [
#             "digital content creation",
#             "multimedia production",
#             "content dissemination",
#             "blogging",
#             "video editing",
#             "content management systems",
#             "cms",
#             "social media management",
#             "digital branding",
#             "produce digital content",
#             "create multimedia content",
#             "create presentations",
#             "create infographics",
#             "create social media content",
#             "digital storytelling"
#         ]
#     },

#     # 3. Algoritma dan Pemrograman / Algorithms & Programming
#     "algorithms_programming": {
#         "analyze": [
#             "algorithm comparison",
#             "programming logic analysis",
#             "code review",
#             "complexity analysis",
#             "big o notation",
#             "benchmarking",
#             "performance tuning",
#             "compare algorithms",
#             "analyze algorithm complexity",
#             "time complexity",
#             "space complexity",
#             "profiling code",
#             "analyze program logic"
#         ],
#         "apply": [
#             "algorithm implementation",
#             "application programming",
#             "coding",
#             "scripting",
#             "software implementation",
#             "executable creation",
#             "debugging basics",
#             "write programs",
#             "implement algorithms in code",
#             "write functions",
#             "write procedures",
#             "use control structures",
#             "write loops",
#             "implement conditional statements",
#             "implement selection structures"
#         ],
#         "create": [
#             "application generation",
#             "produce applications",
#             "build software",
#             "deploy application",
#             "software engineering",
#             "release management",
#             "design algorithms",
#             "design programs",
#             "develop applications",
#             "develop software solutions",
#             "build prototypes"
#         ]
#     },

#     # 4. Analisis Data / Data Analysis & DB Concepts
#     "data_analysis": {
#         "understand": [
#             "database concepts",
#             "basis data",
#             "data processing concepts",
#             "relational database",
#             "rdbms",
#             "data warehousing",
#             "data mining concepts",
#             "data encoding",
#             "character encoding",
#             "utf-8",
#             "ascii",
#             "database normalization concepts",
#             "keys and constraints"
#         ],
#         "apply": [
#             "data processing",
#             "database management",
#             "data cleaning",
#             "etl",
#             "extract transform load",
#             "excel formulas",
#             "spreadsheets",
#             "data visualization",
#             "tableau",
#             "power bi",
#             "looker",
#             "pandas",
#             "numpy",
#             "sql",
#             "sql queries",
#             "select statements",
#             "join operations",
#             "database queries",
#             "data reporting",
#             "generate reports",
#             "etl pipelines",
#             "data pipelines"
#         ]
#     },

#     # 5. Literasi & Etika AI / AI Literacy & Ethics
#     "ai_literacy_ethics": {
#         "understand": [
#             "artificial intelligence principles",
#             "ai technology work",
#             "social impact of ai",
#             "machine learning concepts",
#             "neural networks",
#             "deep learning",
#             "nlp",
#             "computer vision",
#             "llm",
#             "machine learning",
#             "supervised learning",
#             "unsupervised learning",
#             "classification",
#             "regression",
#             "reinforcement learning",
#             "ai applications",
#             "ai use cases"
#         ],
#         "evaluate": [
#             "ai ethics",
#             "data security in ai",
#             "responsible ai",
#             "bias detection",
#             "model fairness",
#             "explainable ai",
#             "xai",
#             "ai compliance",
#             "data privacy",
#             "ethical ai",
#             "ethical considerations",
#             "responsible data use",
#             "data protection",
#             "data governance",
#             "privacy by design"
#         ]
#     },

#     # 6. Pemanfaatan & Pengembangan AI / AI Utilization & Development
#     "ai_development": {
#         "apply": [
#             "prompt engineering",
#             "generative ai utilization",
#             "chatgpt",
#             "midjourney",
#             "copilot",
#             "gemini",
#             "large language models usage",
#             "ai tools integration",
#             "use generative ai",
#             "use llm tools",
#             "use ai assistants",
#             "ai-assisted coding",
#             "ai-assisted design"
#         ],
#         "create": [
#             "ai project planning",
#             "product design with ai",
#             "prototype development",
#             "ai application development",
#             "fine-tuning models",
#             "rag",
#             "retrieval augmented generation",
#             "building chatbots",
#             "ai solutions",
#             "design ai solutions",
#             "design ai workflows",
#             "integrate ai into products",
#             "build ai-powered features"
#         ]
#     },

#     # 7. Wawasan Dunia Kerja / Workforce Insights
#     "workforce_insights": {
#         "analyze": [
#             "software development steps",
#             "game development steps",
#             "customer needs",
#             "cloud computing trends",
#             "job professions",
#             "market research",
#             "competitor analysis",
#             "agile methodology",
#             "scrum",
#             "kanban",
#             "waterfall",
#             "devops culture",
#             "industry standards",
#             "analyze software development process",
#             "analyze game development process",
#             "analyze customer needs",
#             "analyze technology trends",
#             "analyze job roles",
#             "analyze career paths"
#         ],
#         "evaluate": [
#             "career planning",
#             "entrepreneurship",
#             "personal branding",
#             "project management",
#             "product management",
#             "freelancing",
#             "consulting",
#             "startup ecosystem",
#             "portfolio building",
#             "plan career",
#             "evaluate career options",
#             "entrepreneurship in software",
#             "entrepreneurship in games",
#             "personal branding in tech"
#         ]
#     },

#     # 8. Kecakapan Kerja Dasar & K3 / Basic Job Skills & OHS
#     "basic_job_skills_ohs": {
#         "apply": [
#             "k3",
#             "occupational health and safety",
#             "work culture",
#             "standard procedures",
#             "accident prevention",
#             "tool management",
#             "asset management",
#             "user interface development",
#             "work procedures",
#             "safety procedures",
#             "use development tools",
#             "manage development tools",
#             "osha",
#             "workplace safety",
#             "hse",
#             "standard operating procedures",
#             "sop",
#             "inventory management",
#             "version control",
#             "git",
#             "quality assurance",
#             "qa",
#             "first aid",
#             "cpr",
#             "emergency response",
#             "safety equipment",
#             "ppe",
#             "personal protective equipment",
#             "electrical safety"
#         ],
#         "analyze": [
#             "workflow optimization",
#             "root cause analysis",
#             "incident reporting",
#             "risk assessment",
#             "analyze development process",
#             "analyze programming process"
#         ]
#     },

#     # 9. Teknologi Jaringan Komputer / Network Technology
#     "network_technology": {
#         "apply": [
#             "network basics",
#             "computer network",
#             "network configuration",
#             "connectivity",
#             "cabling",
#             "wireless setup",
#             "lan",
#             "wan",
#             "tcp/ip",
#             "dns",
#             "dhcp",
#             "vpn",
#             "routing",
#             "switching",
#             "cisco",
#             "mikrotik",
#             "juniper",
#             "network administration",
#             "configure networks",
#             "configure routers",
#             "configure switches",
#             "setup lan",
#             "setup wifi",
#             "network installation"
#         ],
#         "analyze": [
#             "hardware analysis",
#             "topology comparison",
#             "network troubleshooting",
#             "packet analysis",
#             "wireshark",
#             "network monitoring",
#             "bandwidth analysis",
#             "latency troubleshooting",
#             "network diagnostics",
#             "troubleshoot network issues",
#             "diagnose network problems",
#             "analyze network performance"
#         ]
#     },

#     # 10. Pemrograman Terstruktur / Structured Programming
#     "structured_programming": {
#         "apply": [
#             "structured programming principles",
#             "logic programming",
#             "procedural programming",
#             "c programming",
#             "pascal",
#             "control structures",
#             "loops",
#             "functions",
#             "arrays",
#             "pointers",
#             "write structured code",
#             "use loops",
#             "use conditionals",
#             "simple console applications"
#         ],
#         "analyze": [
#             "logic analysis",
#             "debugging",
#             "problem solving",
#             "stack trace analysis",
#             "memory management",
#             "code optimization",
#             "error handling",
#             "analyze logic errors",
#             "trace code",
#             "debug procedural code"
#         ],
#         "create": [
#             "effective solution creation",
#             "efficient program",
#             "algorithm design",
#             "embedded systems programming",
#             "utility scripts"
#         ]
#     },

#     # 11. Pemrograman Berorientasi Objek / OOP
#     "oop": {
#         "apply": [
#             "object oriented concepts",
#             "oop",
#             "encapsulation",
#             "inheritance",
#             "polymorphism",
#             "java",
#             "c#",
#             ".net",
#             "python",
#             "c++",
#             "classes",
#             "objects",
#             "instantiation",
#             "object oriented programming",
#             "oop principles",
#             "apply encapsulation",
#             "apply inheritance",
#             "apply polymorphism"
#         ],
#         "create": [
#             "oop solution",
#             "class design",
#             "software architecture",
#             "class hierarchy",
#             "api design",
#             "backend logic",
#             "modular code",
#             "design class diagrams",
#             "design class hierarchy",
#             "design object model"
#         ]
#     },

#     # 12. Basis Data / Database Systems
#     "database_systems": {
#         "create": [
#             "database design",
#             "database creation",
#             "sql utilization",
#             "structured query language",
#             "mysql",
#             "postgresql",
#             "oracle",
#             "sql server",
#             "sqlite",
#             "mongodb",
#             "nosql",
#             "database schema",
#             "erd",
#             "entity relationship diagram",
#             "normalization",
#             "stored procedures",
#             "triggers",
#             "crud",
#             "design database schema",
#             "design erd",
#             "design entity relationship diagram",
#             "design tables",
#             "define primary key",
#             "define foreign key",
#             "create sql database",
#             "create tables",
#             "write ddl",
#             "write dml"
#         ]
#     },

#     # 13. Pemrograman berbasis teks, grafis, dan multimedia / Multimedia Programming
#     "multimedia_programming": {
#         "apply": [
#             "command execution",
#             "library utilization",
#             "gui programming",
#             "graphical user interface",
#             "swing",
#             "javafx",
#             "tkinter",
#             "qt",
#             "external libraries",
#             "dependency management",
#             "maven",
#             "gradle",
#             "pip",
#             "npm",
#             "event-driven programming",
#             "desktop applications",
#             "use graphics libraries"
#         ],
#         "create": [
#             "software modeling",
#             "multimedia application",
#             "desktop application",
#             "graphics rendering",
#             "opengl",
#             "directx",
#             "game mechanics",
#             "uml modeling",
#             "design desktop applications",
#             "design gui",
#             "build multimedia apps"
#         ]
#     },

#     # 14. Pemrograman Web / Web Development
#     "web_development": {
#         "apply": [
#             "server-side programming",
#             "web framework",
#             "web documentation",
#             "html",
#             "css",
#             "javascript",
#             "typescript",
#             "php",
#             "laravel",
#             "codeigniter",
#             "bootstrap",
#             "tailwind",
#             "jquery",
#             "frontend development",
#             "backend development",
#             "full stack developer",
#             "http",
#             "json",
#             "restful api"
#         ],
#         "create": [
#             "static web",
#             "dynamic web",
#             "web application",
#             "react",
#             "angular",
#             "vue",
#             "node.js",
#             "express",
#             "django",
#             "flask",
#             "asp.net",
#             "ruby on rails",
#             "rest api",
#             "graphql",
#             "full stack development",
#             "frontend",
#             "backend",
#             "develop web applications",
#             "build rest api",
#             "implement authentication",
#             "implement authorization",
#             "deploy web application"
#         ]
#     },

#     # 15. Pemrograman Perangkat Bergerak / Mobile Development
#     "mobile_development": {
#         "apply": [
#             "mobile programming",
#             "ide utilization",
#             "mobile framework",
#             "mobile database",
#             "api integration",
#             "android studio",
#             "xcode",
#             "sdk",
#             "firebase",
#             "sqlite",
#             "realm",
#             "rest client",
#             "android developer",
#             "ios developer",
#             "use mobile sdk",
#             "call rest api from mobile app"
#         ],
#         "create": [
#             "mobile application",
#             "android application",
#             "ios application",
#             "kotlin",
#             "java for android",
#             "swift",
#             "objective-c",
#             "flutter",
#             "dart",
#             "react native",
#             "ionic",
#             "cross-platform development",
#             "app deployment",
#             "play store",
#             "app store",
#             "develop mobile applications",
#             "publish app to play store",
#             "publish app to app store"
#         ]
#     },

#     # 16. Pemodelan Gim / Game Modeling
#     "game_modeling": {
#         "analyze": [
#             "game concept",
#             "game design document",
#             "gdd",
#             "game mechanic concept",
#             "game system concept",
#             "game level concept",
#             "game narrative concept",
#             "game user research",
#             "analyze game balancing"
#         ],
#         "create": [
#             "create game prototype",
#             "game design prototype",
#             "design game levels",
#             "design game systems",
#             "design game mechanics",
#             "design game narrative"
#         ]
#     },

#     # 17. Pemrograman Gim / Game Programming
#     "game_programming": {
#         "apply": [
#             "game engine",
#             "unity",
#             "unreal engine",
#             "godot",
#             "gameplay programming",
#             "implement gameplay",
#             "implement ui ux in games",
#             "integrate static assets",
#             "integrate dynamic assets"
#         ],
#         "analyze": [
#             "debug games",
#             "optimize game performance",
#             "debug gameplay",
#             "profile game performance"
#         ],
#         "create": [
#             "develop game",
#             "build game",
#             "implement game mechanics",
#             "implement game features",
#             "implement tools and plugins",
#             "game testing",
#             "game polishing"
#         ]
#     },

#     # 18. Komputer Grafis & Multimedia / Graphics & Multimedia
#     "graphics_multimedia": {
#         "apply": [
#             "concept art",
#             "art design document",
#             "character design",
#             "environment design",
#             "prop design",
#             "2d animation",
#             "cut out animation",
#             "puppeteer animation",
#             "3d modeling",
#             "digital sculpting",
#             "texturing",
#             "uv mapping",
#             "rigging",
#             "character rigging",
#             "simulate rigid body",
#             "simulate soft body",
#             "shading",
#             "material design"
#         ],
#         "create": [
#             "create game assets",
#             "create visual assets",
#             "design characters",
#             "design environments",
#             "design props"
#         ]
#     },

#     # 19. Audio Editing
#     "audio_editing": {
#         "apply": [
#             "audio editing",
#             "sound editing",
#             "sound design",
#             "dubbing",
#             "voice over",
#             "record voice",
#             "record dialogue",
#             "sound effects",
#             "background music",
#             "music scoring",
#             "use daw",
#             "digital audio workstation",
#             "audacity",
#             "mixing audio",
#             "audio mixing",
#             "audio mastering"
#         ]
#     }
# }

CURRICULUM_COMPONENTS = {
    # 1. Berpikir Komputasional / Computational Thinking
    "computational_thinking": {
        "understand": [
            # understanding data structures, algorithms, models, OS, IPO
            "data structure concepts",
            "standard algorithms",
            "standard data structures",
            "algorithmic thinking",
            "problem decomposition",
            "abstraction",
            "pattern recognition",
            "control structures",
            "input-process-output",
            "input process output",
            "IPO model",
            "von neumann architecture",
            "von neumann model",
            "von neumann computer",
            "computer architecture",
            "operating system roles",
            "operating system concepts",
            "operating system fundamentals",
            "memory management",
            "process scheduling",
            "binary logic",
            "system logic",
            # CT in domain and real world
            "computational models",
            "role of computational thinking in problem solving",
            "computational thinking in specific domains",
            "computational thinking in workplace and society"
        ],
        "apply": [
            # applying computational processes (individually & in groups) to get quality data
            "computational process",
            "independent computation",
            "group computation",
            "use computational thinking to collect quality data",
            "apply computational thinking to domain-specific problems",
            "apply computational thinking to complex real-world problems",
            "apply computational thinking in workplace contexts",
            # pseudocode & representation
            "pseudocode",
            "write pseudocode",
            "design pseudocode",
            "pseudocode for simple programs",
            "convert problem descriptions to pseudocode",
            # algorithm execution & representation
            "algorithm execution",
            "flowcharting",
            "simulate algorithms",
            "trace algorithms",
            "debug algorithms",
            # tools and automation
            "command line",
            "terminal usage",
            "bash scripting",
            "shell scripting",
            "automating tasks",
            # implementing structures / algorithms
            "implement data structures",
            "implement algorithms",
            "apply standard algorithms"
        ],
        "analyze": [
            # analyzing problems and complexity; predicting
            "analyze computational problems",
            "analyze problem complexity",
            "predict outcomes using computational models",
            "evaluate algorithm choices",
            "compare algorithms for problem solving",
            "choose appropriate data structures",
            "analyze complex problems using computational thinking",
            "analyze real-world problems in workplace and society with CT"
        ],
        "create": [
            # designing CT-based solutions
            "program design",
            "solution design",
            "problem solving strategy",
            "system architecture design",
            "technical specification",
            "workflow design",
            "logic design",
            "design pseudocode-based solutions",
            "design computational solutions for domain-specific problems",
            "design computational solutions for complex workplace problems"
        ]
    },

    # 2. Literasi Digital / Digital Literacy
    "digital_literacy": {
        "understand": [
            # search, information ecosystem, fact-checking, lateral reading
            "search engine mechanisms",
            "search engine usage",
            "advanced search",
            "search operators",
            "information ecosystem",
            "fact-checking ecosystem",
            "fake news detection",
            "lateral reading",
            # digital tools for docs, sheets, presentations
            "digital technology devices",
            "document processing tools concepts",
            "spreadsheet tools concepts",
            "presentation tools concepts",
            # connectivity & basic security
            "cloud connectivity",
            "internet of things",
            "cloud storage concepts",
            "basic network security",
            "wifi security",
            "encryption basics",
            "basic network configuration concepts",
            # digital citizenship & culture
            "digital citizenship",
            "intellectual property rights",
            "ip rights in digital content",
            "informatics professions",
            "digitalization of Indonesian culture",
            "negative content in digital space",
            "filtering negative content"
        ],
        "apply": [
            # using tools for docs, sheets, presentations, collaboration
            "advanced search techniques",
            "use search engines",
            "document processing",
            "spreadsheet utilization",
            "use spreadsheets",
            "presentation software",
            "use presentation tools",
            "word processing",
            "microsoft office",
            "excel",
            "word",
            "powerpoint",
            "google workspace",
            "online collaboration",
            "collaboration tools",
            "slack",
            "trello",
            "jira",
            "zoom",
            "teams",
            # security & privacy practices
            "password management",
            "manage passwords",
            "two-factor authentication",
            "configure wifi",
            "configure privacy settings",
            "privacy configuration",
            "apply basic network security",
            "apply basic encryption",
            # media use for participation & collaboration
            "use digital media for participation",
            "use digital media for collaboration",
            # content production & dissemination (basic & advanced)
            "digital content creation",
            "multimedia production",
            "content dissemination",
            "blogging",
            "video editing",
            "social media management",
            "digital branding",
            "produce digital content",
            "create multimedia content",
            "create presentations",
            "create infographics",
            "create social media content",
            "digital storytelling",
            "apply digital content production in own field",
            "apply advanced digital content production for applications and AI"
        ],
        "evaluate": [
            # evaluating digital information and sources
            "fact checking",
            "lateral reading",
            "information evaluation",
            "evaluate online information",
            "detect misinformation",
            "assess credibility of sources",
            "evaluate digital sources",
            "verify information",
            "cross-check information",
            # privacy, security, ethics
            "data privacy",
            "gdpr compliance",
            "phishing awareness",
            "cyber hygiene",
            "information security basics",
            "digital footprint analysis",
            "evaluate digital behavior",
            "evaluate ethical use of digital content",
            "respect intellectual property in digital content"
        ],
        "create": [
            # intentional content creation for field & advanced apps/AI support
            "digital content creation",
            "multimedia production",
            "content dissemination",
            "blogging",
            "video editing",
            "content management systems",
            "cms",
            "social media management",
            "digital branding",
            "produce digital content",
            "create multimedia content",
            "create presentations",
            "create infographics",
            "create social media content",
            "digital storytelling",
            "create digital content for own field",
            "create advanced digital content to support application and AI development"
        ]
    },

    # 3. Algoritma dan Pemrograman / Algorithms & Programming
    "algorithms_programming": {
        "analyze": [
            "refractor",
            "debug",
            "algorithm comparison",
            "compare algorithms",
            "programming logic analysis",
            "analyze program logic",
            "code review",
            "complexity analysis",
            "big o notation",
            "benchmarking",
            "performance tuning",
            "analyze algorithm complexity",
            "time complexity",
            "space complexity",
            "profiling code"
        ],
        "apply": [
            # apply algorithms and programming to produce applications
            "algorithm implementation",
            "apply programming algorithms to build applications",
            "application programming",
            "coding",
            "refractor",
            "script",
            "scripting",
            "software implementation",
            "executable creation",
            "debugging basics",
            "write programs",
            "implement algorithms in code",
            "write functions",
            "write procedures",
            "use control structures",
            "write loops",
            "implement conditional statements",
            "implement selection structures",
            # applying OOP with advanced tools for AI utilization (bridging with OOP + AI)
            "apply object oriented programming for AI-related applications",
            "use advanced coding tools for AI utilization"
        ],
        "create": [
            "application generation",
            "produce applications",
            "build software",
            "deploy application",
            "software engineering",
            "release management",
            "design algorithms",
            "design programs",
            "develop applications",
            "develop software solutions",
            "build prototypes",
            "create AI-enabled applications"
        ]
    },

    # 4. Analisis Data / Data Analysis & DB Concepts
    "data_analysis": {
        "understand": [
            "database concepts",
            "basis data",
            "data processing concepts",
            "relational database",
            "rdbms",
            "data warehousing",
            "data mining concepts",
            "data encoding",
            "character encoding",
            "utf-8",
            "ascii",
            "database normalization concepts",
            "keys and constraints",
            "role of databases in workplace",
            "role of databases in society"
        ],
        "apply": [
            "data processing",
            "database management",
            "data cleaning",
            "etl",
            "extract transform load",
            "etl pipelines",
            "data pipelines",
            "excel formulas",
            "spreadsheets",
            "data visualization",
            "tableau",
            "power bi",
            "looker",
            "pandas",
            "numpy",
            "sql",
            "sql queries",
            "select statements",
            "join operations",
            "database queries",
            "data reporting",
            "generate reports",
            "apply databases to solve workplace problems",
            "apply databases to solve community problems"
        ]
    },

    # 5. Literasi & Etika AI / AI Literacy & Ethics
    "ai_literacy_ethics": {
        "understand": [
            # understanding AI tech and principles
            "artificial intelligence principles",
            "ai technology work",
            "machine learning concepts",

            "use ai",
            "ai applications",
            "ai use cases",
            # context-specific and societal understanding
            "social impact of ai",
            "responsible use of ai in own field",
            "responsible use of ai in workplace and society"
        ],
        "evaluate": [
            "ai ethics",
            "ethical ai",
            "ethical considerations",
            "responsible ai",
            "responsible data use",
            "data security in ai",
            "data protection",
            "data governance",
            "privacy by design",
            "bias detection",
            "model fairness",
            "explainable ai",
            "xai",
            "ai compliance",
            "data privacy",
            "evaluate ethical and social impact of ai in own field",
            "evaluate ethical and social impact of ai in workplace and society"
        ]
    },

    # 6. Pemanfaatan & Pengembangan AI / AI Utilization & Development
    "ai_development": {
        "apply": [
            # using generative AI and tools, prompt engineering as evaluation of understanding
            "prompt",
            "prompt engineering",
            "apply prompt engineering to evaluate project understanding",
            "generative ai utilization",
            "use generative ai",
            "large language models usage",
            "use llm tools",
            "chatgpt",
            "prompt",
            "midjourney",
            "copilot",
            "gemini",
            "ai tools integration",
            "use ai assistants",
            "ai-assisted coding",
            "ai-assisted design",
            "use ai tools in domain-specific projects",
            "neural networks",
            "deep learning",
            "nlp",
            "computer vision",
            "llm",
            "machine learning",
            "supervised learning",
            "unsupervised learning",
            "classification",
            "regression",
            "reinforcement learning"
        ],
        "create": [
            # planning and actualizing AI products/solutions
            "ai project planning",
            "product design with ai",
            "design ai solutions",
            "design ai workflows",
            "prototype development",
            "ai application development",
            "fine-tuning models",
            "rag",
            "retrieval augmented generation",
            "building chatbots",
            "ai solutions",
            "integrate ai into products",
            "build ai-powered features",
            "actualize ai products for domain-specific problems",
            "build ai solutions for workplace and community problems",
            "machine learning model",
            "machine learning solutions",
            "train machine learning models",
            "evaluate machine learning models",
            "train ai models",
            "evaluate ai models",
            "train ai"
        ]
    },

    # 7. Wawasan Dunia Kerja / Workforce Insights
    "workforce_insights": {
        "analyze": [
            "scrum",
            "agile",
            "kanban",
            "software development steps",
            "game development steps",
            "analyze software development process",
            "analyze game development process",
            "customer needs",
            "analyze customer needs",
            "cloud computing trends",
            "analyze technology trends",
            "job professions",
            "analyze job roles",
            "analyze career paths",
            "market research",
            "competitor analysis",
            "agile methodology",
            "scrum",
            "kanban",
            "waterfall",
            "devops culture",
            "industry standards",
            "analyze entrepreneurship opportunities in software",
            "analyze entrepreneurship opportunities in games",
            "analyze personal branding needs"
        ],
        "evaluate": [
            "career planning",
            "plan career",
            "evaluate career options",
            "entrepreneurship",
            "entrepreneurship in software",
            "entrepreneurship in games",
            "personal branding",
            "personal branding in tech",
            "project management",
            "product management",
            "freelancing",
            "consulting",
            "startup ecosystem",
            "portfolio building",
            "conclude importance of personal branding for career",
            "evaluate career and entrepreneurship opportunities in software and games"
        ]
    },

    # 8. Kecakapan Kerja Dasar & K3 / Basic Job Skills & OHS
    "basic_job_skills_ohs": {
        "apply": [
            "occupational health and safety",
            "osha",
            "hse",
            "workplace safety",
            "standard procedures",
            "standard operating procedures",
            "sop",
            "work culture",
            "work procedures",
            "safety procedures",
            "accident prevention",
            "tool management",
            "asset management",
            "inventory management",
            "user interface development",
            "use development tools",
            "manage development tools",
            "quality assurance",
            "qa",
            "version control",
            "git",
            "first aid",
            "cpr",
            "emergency response",
            "safety equipment",
            "ppe",
            "personal protective equipment",
            "electrical safety",
            "apply workplace culture in software and game development"
        ],
        "analyze": [
            "workflow optimization",
            "root cause analysis",
            "incident reporting",
            "risk assessment",
            "analyze development process",
            "analyze programming process",
            "analyze algorithms in development process for efficiency and correctness"
        ]
    },

    # 9. Teknologi Jaringan Komputer / Network Technology
    "network_technology": {
        "apply": [
            "network basics",
            "computer network",
            "network configuration",
            "connectivity",
            "cabling",
            "wireless setup",
            "lan",
            "wan",
            "tcp/ip",
            "dns",
            "dhcp",
            "vpn",
            "routing",
            "switching",
            "cisco",
            "mikrotik",
            "juniper",
            "network administration",
            "configure networks",
            "configure routers",
            "configure switches",
            "setup lan",
            "setup wifi",
            "network installation",
            "apply basic computer network concepts"
        ],
        "analyze": [
            "hardware analysis",
            "topology comparison",
            "network troubleshooting",
            "packet analysis",
            "wireshark",
            "network monitoring",
            "bandwidth analysis",
            "latency troubleshooting",
            "network diagnostics",
            "troubleshoot network issues",
            "diagnose network problems",
            "analyze network performance",
            "analyze network topology choices"
        ]
    },

    # 10. Pemrograman Terstruktur / Structured Programming
    "structured_programming": {
        "apply": [
            "structured programming principles",
            "logic programming",
            "procedural programming",
            "c programming",
            "pascal",
            "control structures",
            "loops",
            "functions",
            "arrays",
            "pointers",
            "write structured code",
            "use loops",
            "use conditionals",
            "simple console applications",
            "apply structured programming in simple software projects",
            "apply structured programming in simple game projects"
        ],
        "analyze": [
            "logic analysis",
            "debugging",
            "problem solving",
            "stack trace analysis",
            "memory management",
            "code optimization",
            "error handling",
            "analyze logic errors",
            "trace code",
            "debug procedural code",
            "analyze and solve logic problems for effective and efficient solutions"
        ],
        "create": [
            "effective solution creation",
            "efficient program",
            "algorithm design",
            "embedded systems programming",
            "utility scripts",
            "create simple software using structured programming",
            "create simple games using structured programming"
        ]
    },

    # 11. Pemrograman Berorientasi Objek / OOP
    "oop": {
        "apply": [
            "object oriented concepts",
            "oop",
            "encapsulation",
            "inheritance",
            "polymorphism",
            "java",
            "c#",
            ".net",
            "python",
            "c++",
            "classes",
            "objects",
            "instantiation",
            "object oriented programming",
            "oop principles",
            "apply encapsulation",
            "apply inheritance",
            "apply polymorphism",
            "apply oop in simple software projects",
            "apply oop in simple game projects"
        ],
        "create": [
            "oop solution",
            "class design",
            "software architecture",
            "class hierarchy",
            "api design",
            "backend logic",
            "modular code",
            "design class diagrams",
            "design class hierarchy",
            "design object model",
            "build solutions using encapsulation, inheritance, and polymorphism",
            "create simple software using oop",
            "create simple games using oop"
        ]
    },

    # 12. Basis Data / Database Systems
    "database_systems": {
        "create": [
            "database design",
            "database creation",
            "sql utilization",
            "structured query language",
            "mysql",
            "postgresql",
            "oracle",
            "sql server",
            "sqlite",
            "mongodb",
            "nosql",
            "database schema",
            "erd",
            "entity relationship diagram",
            "normalization",
            "stored procedures",
            "triggers",
            "crud",
            "design database schema",
            "design erd",
            "design entity relationship diagram",
            "design tables",
            "define primary key",
            "define foreign key",
            "create sql database",
            "create tables",
            "write ddl",
            "write dml",
            "use sql to manipulate data"
        ]
    },

    # 13. Pemrograman berbasis teks, grafis, dan multimedia / Multimedia Programming
    "multimedia_programming": {
        "apply": [
            "command execution",
            "library utilization",
            "gui programming",
            "graphical user interface",
            "swing",
            "javafx",
            "tkinter",
            "qt",
            "external libraries",
            "dependency management",
            "maven",
            "gradle",
            "pip",
            "npm",
            "event-driven programming",
            "desktop applications",
            "use graphics libraries",
            "implement structured programming",
            "implement advanced oop in gui and multimedia",
            "implement software modeling concepts"
        ],
        "create": [
            "software modeling",
            "multimedia application",
            "desktop application",
            "graphics rendering",
            "opengl",
            "directx",
            "game mechanics",
            "uml modeling",
            "design desktop applications",
            "design gui",
            "build multimedia apps",
            "build text-based and graphics-based applications"
        ]
    },

    # 14. Pemrograman Web / Web Development
    "web_development": {
        "apply": [
            "server-side programming",
            "web framework",
            "web documentation",
            "html",
            "css",
            "javascript",
            "typescript",
            "php",
            "laravel",
            "codeigniter",
            "bootstrap",
            "tailwind",
            "jquery",
            "frontend development",
            "backend development",
            "full stack developer",
            "http",
            "json",
            "restful api",
            "apply web programming for static and dynamic websites"
        ],
        "create": [
            "static web",
            "dynamic web",
            "web application",
            "react",
            "angular",
            "vue",
            "node.js",
            "express",
            "django",
            "flask",
            "asp.net",
            "ruby on rails",
            "rest api",
            "graphql",
            "full stack development",
            "frontend",
            "backend",
            "develop web applications",
            "build rest api",
            "implement authentication",
            "implement authorization",
            "deploy web application",
            "build contextual web applications"
        ]
    },

    # 15. Pemrograman Perangkat Bergerak / Mobile Development
    "mobile_development": {
        "apply": [
            "mobile programming",
            "ide utilization",
            "mobile framework",
            "mobile database",
            "api integration",
            "android studio",
            "xcode",
            "sdk",
            "firebase",
            "sqlite",
            "realm",
            "rest client",
            "android developer",
            "ios developer",
            "use mobile sdk",
            "call rest api from mobile app",
            "apply mobile programming for contextual needs"
        ],
        "create": [
            "mobile application",
            "android application",
            "ios application",
            "kotlin",
            "java for android",
            "swift",
            "objective-c",
            "flutter",
            "dart",
            "react native",
            "ionic",
            "cross-platform development",
            "app deployment",
            "play store",
            "app store",
            "develop mobile applications",
            "publish app to play store",
            "publish app to app store",
            "create contextual mobile applications"
        ]
    },

    # 16. Pemodelan Gim / Game Modeling
    "game_modeling": {
        "analyze": [
            "game concept",
            "game design document",
            "gdd",
            "game mechanic concept",
            "game system concept",
            "game technical concept",
            "game level concept",
            "game narrative concept",
            "game user research",
            "game user research concept",
            "analyze game balancing"
        ],
        "create": [
            "create game prototype",
            "game design prototype",
            "design game levels",
            "design game systems",
            "design game mechanics",
            "design game narrative"
        ]
    },

    # 17. Pemrograman Gim / Game Programming
    "game_programming": {
        "apply": [
            "game engine",
            "unity",
            "unreal engine",
            "godot",
            "gameplay programming",
            "implement gameplay",
            "implement ui ux in games",
            "integrate static assets",
            "integrate dynamic assets",
            "implement text-based and graphics-based programming in games",
            "implement additional functionality in game engine"
        ],
        "analyze": [
            "debug games",
            "debug gameplay",
            "optimize game performance",
            "profile game performance",
            "analyze game performance"
        ],
        "create": [
            "develop game",
            "build game",
            "implement game mechanics",
            "implement game features",
            "implement tools and plugins",
            "game testing",
            "game polishing",
            "create contextual games"
        ]
    },

    # 18. Komputer Grafis & Multimedia / Graphics & Multimedia
    "graphics_multimedia": {
        "apply": [
            "concept art",
            "art design document",
            "character design",
            "environment design",
            "prop design",
            "2d animation",
            "cut out animation",
            "puppeteer animation",
            "3d modeling",
            "digital sculpting",
            "texturing",
            "uv mapping",
            "rigging",
            "character rigging",
            "simulate rigid body",
            "simulate soft body",
            "shading",
            "material design",
            "apply visual asset development for games",
            "render",
            "animate",
            "rig",
            "shade",
            "texture",
            "light",
            "bake",
            "cull",
            "draw",
            "paint",
            "sculpt",
            "compose"
        ],
        "create": [
            "create game assets",
            "create visual assets",
            "design characters",
            "design environments",
            "design props",
            "create 2d animation for games",
            "create 3d models for games",
            "create visual effects for games"
        ]
    },

    # 19. Audio Editing
    "audio_editing": {
        "apply": [
            "audio editing",
            "sound editing",
            "sound design",
            "dubbing",
            "voice over",
            "record voice",
            "record dialogue",
            "sound effects",
            "background music",
            "music scoring",
            "use daw",
            "digital audio workstation",
            "audacity",
            "mixing audio",
            "audio mixing",
            "audio mastering",
            "plan audio assets for games",
            "develop audio assets for games"
        ]
    }
}
