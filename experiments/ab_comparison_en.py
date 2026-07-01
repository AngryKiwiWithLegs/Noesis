#!/usr/bin/env python3
"""
English A/B Experiment: With-memory (Noesis) vs Without-memory (direct)
=========================================================================
Same design as the Chinese UNIFIED_AB experiment, but in English,
with enlarged sample size: 10 profiles x 5 questions = 50 questions.

Purpose: Test whether Noesis works cross-lingually, and whether results
hold in English. Publishable finding for a bilingual paper.

Usage:
    GEMINI_API_KEY="yourkey" python3 ab_comparison_en.py
"""
import os, sys, json, time, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
PROXY_URL  = "http://127.0.0.1:8080/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-flash-lite-latest"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 10 user profiles, each with 5 build statements + 5 test questions ──────────
# Designed to be parallel to the Chinese version but distinct personas,
# covering identity / preference / position / event / fact.
PROFILES = [
    {"name": "prof_1",
     "build": ["My name is John, I am a backend engineer with 8 years of experience",
               "I prefer PostgreSQL over MySQL for its JSON support",
               "I decided to use sqlite-vec for vector search, lighter than FAISS",
               "I prefer Python for backend, cleaner than Java",
               "I think microservices are better than monoliths for fast iteration"],
     "questions": [
        {"q": "Recommend a database for me",          "expect": "postgres"},
        {"q": "What do I use for vector search?",     "expect": "sqlite"},
        {"q": "How do I feel about Python vs Java?",  "expect": "python"},
        {"q": "What is my view on microservices?",    "expect": "microservice"},
        {"q": "What is my job title?",                "expect": "backend"},
    ]},
    {"name": "prof_2",
     "build": ["My name is Sarah, I am a data scientist specializing in ML",
               "I prefer using pandas for data processing, better than Excel",
               "I decided to learn Rust for systems programming, safer than C++",
               "I like using Obsidian for notes, more flexible than Notion",
               "I tend to use Docker for dev environments, I dislike local installs"],
     "questions": [
        {"q": "What do I use to process data?",       "expect": "pandas"},
        {"q": "What programming language am I learning?","expect": "rust"},
        {"q": "What do I use for notes?",             "expect": "obsidian"},
        {"q": "How do I feel about local dependency installs?", "expect": "docker"},
        {"q": "What is my profession?",               "expect": "data"},
    ]},
    {"name": "prof_3",
     "build": ["My name is Mike, I am a frontend engineer expert in React",
               "I prefer TypeScript over JavaScript",
               "I decided to use Tailwind CSS for styling, faster than handwritten CSS",
               "I think GraphQL is better than REST for complex frontends",
               "I like using Vim keybindings, faster than a mouse"],
     "questions": [
        {"q": "What frontend framework am I expert in?", "expect": "react"},
        {"q": "What do I use for styling?",              "expect": "tailwind"},
        {"q": "What is my preference on TypeScript?",    "expect": "typescript"},
        {"q": "What is my view on GraphQL vs REST?",     "expect": "graphql"},
        {"q": "What kind of engineer am I?",             "expect": "frontend"},
    ]},
    {"name": "prof_4",
     "build": ["My name is Emma, I am a DevOps engineer focused on Kubernetes",
               "I prefer Grafana for monitoring, better than Prometheus UI alone",
               "I decided to use Terraform for infrastructure as code",
               "I think ArgoCD is better than Flux for GitOps",
               "I like using Helm charts for package management"],
     "questions": [
        {"q": "What is my monitoring tool of choice?",  "expect": "grafana"},
        {"q": "What do I use for infrastructure?",      "expect": "terraform"},
        {"q": "What is my view on ArgoCD vs Flux?",     "expect": "argocd"},
        {"q": "What do I use for package management?",  "expect": "helm"},
        {"q": "What kind of engineer am I?",            "expect": "devops"},
    ]},
    {"name": "prof_5",
     "build": ["My name is David, I am a mobile developer building iOS apps",
               "I prefer Swift over Objective-C for modern syntax",
               "I decided to use SwiftUI over UIKit for new projects",
               "I think Combine is better than RxSwift for reactive programming",
               "I like using Fastlane for CI/CD in mobile releases"],
     "questions": [
        {"q": "What language do I prefer for iOS?",     "expect": "swift"},
        {"q": "What UI framework do I use for new projects?", "expect": "swiftui"},
        {"q": "What is my view on Combine vs RxSwift?", "expect": "combine"},
        {"q": "What do I use for mobile CI/CD?",        "expect": "fastlane"},
        {"q": "What kind of developer am I?",           "expect": "mobile"},
    ]},
    {"name": "prof_6",
     "build": ["My name is Lisa, I am an ML engineer working on NLP models",
               "I prefer PyTorch over TensorFlow for research flexibility",
               "I decided to use Hugging Face Transformers for model deployment",
               "I think LoRA fine-tuning is better than full fine-tuning for cost",
               "I like using Weights & Biases for experiment tracking"],
     "questions": [
        {"q": "What deep learning framework do I prefer?", "expect": "pytorch"},
        {"q": "What do I use for model deployment?",       "expect": "hugging"},
        {"q": "What is my view on LoRA vs full fine-tuning?","expect": "lora"},
        {"q": "What do I use for experiment tracking?",    "expect": "weights"},
        {"q": "What kind of engineer am I?",               "expect": "ml"},
    ]},
    {"name": "prof_7",
     "build": ["My name is Alex, I am a game developer using Unity",
               "I prefer C# over C++ for game scripting productivity",
               "I decided to use Blender for 3D modeling, open-source and powerful",
               "I think ECS architecture is better than OOP for game entities",
               "I like using FMOD for game audio, more flexible than Wwise"],
     "questions": [
        {"q": "What game engine do I use?",               "expect": "unity"},
        {"q": "What language do I prefer for scripting?", "expect": "c#"},
        {"q": "What do I use for 3D modeling?",            "expect": "blender"},
        {"q": "What is my view on game architecture?",    "expect": "ecs"},
        {"q": "What kind of developer am I?",              "expect": "game"},
    ]},
    {"name": "prof_8",
     "build": ["My name is Rachel, I am a security engineer focused on cloud",
               "I prefer SentinelOne over CrowdStrike for endpoint detection",
               "I decided to use Vault for secrets management over AWS Secrets Manager",
               "I think zero-trust architecture is essential for modern security",
               "I like using Falco for runtime threat detection in Kubernetes"],
     "questions": [
        {"q": "What endpoint tool do I prefer?",          "expect": "sentinelone"},
        {"q": "What do I use for secrets management?",    "expect": "vault"},
        {"q": "What is my security philosophy?",          "expect": "zero-trust"},
        {"q": "What do I use for runtime threat detection?","expect": "falco"},
        {"q": "What kind of engineer am I?",              "expect": "security"},
    ]},
    {"name": "prof_9",
     "build": ["My name is Tom, I am a blockchain developer on Ethereum",
               "I prefer Solidity over Vyper for smart contract development",
               "I decided to use Hardhat over Truffle for development workflow",
               "I think layer-2 rollups are the future of Ethereum scaling",
               "I like using The Graph for indexing blockchain data"],
     "questions": [
        {"q": "What language do I use for smart contracts?", "expect": "solidity"},
        {"q": "What development tool do I prefer?",          "expect": "hardhat"},
        {"q": "What is my view on Ethereum scaling?",        "expect": "rollup"},
        {"q": "What do I use for data indexing?",            "expect": "graph"},
        {"q": "What kind of developer am I?",                "expect": "blockchain"},
    ]},
    {"name": "prof_10",
     "build": ["My name is Maria, I am a data engineer building ETL pipelines",
               "I prefer Apache Airflow over Luigi for workflow orchestration",
               "I decided to use dbt for data transformations, cleaner than SQL scripts",
               "I think Snowflake is better than BigQuery for data warehousing",
               "I like using Apache Kafka for real-time data streaming"],
     "questions": [
        {"q": "What orchestration tool do I prefer?",    "expect": "airflow"},
        {"q": "What do I use for transformations?",      "expect": "dbt"},
        {"q": "What is my view on Snowflake vs BigQuery?","expect": "snowflake"},
        {"q": "What do I use for streaming?",            "expect": "kafka"},
        {"q": "What kind of engineer am I?",             "expect": "data"},
    ]},
    {"name": "prof_11",
     "build": ["My name is Chris, I am an embedded systems engineer",
               "I prefer C over C++ for firmware, simpler toolchain",
               "I decided to use Zephyr RTOS over FreeRTOS",
               "I think PlatformIO is better than Arduino IDE for embedded",
               "I like using Segger J-Link for debugging microcontrollers"],
     "questions": [
        {"q": "What language do I prefer for firmware?",  "expect": "c"},
        {"q": "Which RTOS do I use?",                     "expect": "zephyr"},
        {"q": "What IDE do I prefer for embedded?",       "expect": "platformio"},
        {"q": "What debugger do I use?",                  "expect": "j-link"},
        {"q": "What kind of engineer am I?",              "expect": "embedded"},
    ]},
    {"name": "prof_12",
     "build": ["My name is Diana, I am an SRE focused on reliability",
               "I prefer Prometheus over InfluxDB for metrics collection",
               "I decided to use PagerDuty over Opsgenie for on-call",
               "I think Chaos Mesh is better than Litmus for chaos engineering",
               "I like using Linkerd over Istio for service mesh simplicity"],
     "questions": [
        {"q": "What metrics tool do I prefer?",           "expect": "prometheus"},
        {"q": "What on-call tool do I use?",              "expect": "pagerduty"},
        {"q": "What chaos engineering tool do I prefer?", "expect": "chaos"},
        {"q": "Which service mesh do I prefer?",          "expect": "linkerd"},
        {"q": "What is my SRE role?",                     "expect": "sre"},
    ]},
    {"name": "prof_13",
     "build": ["My name is Frank, I am a QA automation engineer",
               "I prefer Playwright over Cypress for E2E testing",
               "I decided to use TestRail over Zephyr for test management",
               "I think Pytest is better than Unittest for Python testing",
               "I like using Allure for test reporting dashboards"],
     "questions": [
        {"q": "What E2E testing tool do I prefer?",      "expect": "playwright"},
        {"q": "What test management tool do I use?",     "expect": "testrail"},
        {"q": "What Python testing framework do I prefer?", "expect": "pytest"},
        {"q": "What reporting tool do I use?",           "expect": "allure"},
        {"q": "What is my QA role?",                     "expect": "qa"},
    ]},
    {"name": "prof_14",
     "build": ["My name is Grace, I am a technical writer",
               "I prefer Markdown over reStructuredText for docs",
               "I decided to use Docusaurus over MkDocs for documentation sites",
               "I think GitBook is better than Confluence for developer docs",
               "I like using Mermaid for diagrams in documentation"],
     "questions": [
        {"q": "What markup language do I prefer?",       "expect": "markdown"},
        {"q": "What docs framework do I use?",           "expect": "docusaurus"},
        {"q": "What do I think of GitBook vs Confluence?", "expect": "gitbook"},
        {"q": "What diagram tool do I use?",             "expect": "mermaid"},
        {"q": "What is my profession?",                  "expect": "writer"},
    ]},
    {"name": "prof_15",
     "build": ["My name is Henry, I am an AR/VR developer",
               "I prefer Unity over Unreal for AR development",
               "I decided to use AR Foundation over ARCore directly",
               "I think WebXR is the future of immersive web",
               "I like using Blender for 3D asset creation"],
     "questions": [
        {"q": "What engine do I prefer for AR?",         "expect": "unity"},
        {"q": "What AR framework do I use?",             "expect": "ar foundation"},
        {"q": "What is my view on immersive web?",       "expect": "webxr"},
        {"q": "What 3D tool do I use?",                  "expect": "blender"},
        {"q": "What kind of developer am I?",            "expect": "ar"},
    ]},
    {"name": "prof_16",
     "build": ["My name is Ivy, I am a cloud architect",
               "I prefer Terraform over CloudFormation for IaC",
               "I decided to use Pulumi for multi-cloud programming",
               "I think AWS CDK is better than Serverless Framework",
               "I like using Terragrunt for Terraform orchestration"],
     "questions": [
        {"q": "What IaC tool do I prefer?",              "expect": "terraform"},
        {"q": "What multi-cloud tool do I use?",         "expect": "pulumi"},
        {"q": "What is my view on AWS CDK?",             "expect": "cdk"},
        {"q": "What Terraform orchestration tool do I use?", "expect": "terragrunt"},
        {"q": "What is my role?",                        "expect": "architect"},
    ]},
    {"name": "prof_17",
     "build": ["My name is Jack, I am a database administrator",
               "I prefer PostgreSQL over Oracle for enterprise databases",
               "I decided to use pgAdmin over DBeaver for administration",
               "I think Patroni is better than repmgr for HA PostgreSQL",
               "I like using PgBouncer for connection pooling"],
     "questions": [
        {"q": "What enterprise database do I prefer?",   "expect": "postgresql"},
        {"q": "What admin tool do I use?",               "expect": "pgadmin"},
        {"q": "What HA solution do I prefer?",           "expect": "patroni"},
        {"q": "What connection pooler do I use?",        "expect": "pgbouncer"},
        {"q": "What is my DBA role?",                    "expect": "database"},
    ]},
    {"name": "prof_18",
     "build": ["My name is Kate, I am a UX researcher",
               "I prefer Figma over Sketch for design work",
               "I decided to use Maze for prototype usability testing",
               "I think Dovetail is better than Notion for research synthesis",
               "I like using Optimal Workshop for card sorting"],
     "questions": [
        {"q": "What design tool do I prefer?",           "expect": "figma"},
        {"q": "What usability testing tool do I use?",   "expect": "maze"},
        {"q": "What research synthesis tool do I prefer?", "expect": "dovetail"},
        {"q": "What card sorting tool do I use?",        "expect": "optimal"},
        {"q": "What is my UX role?",                     "expect": "ux"},
    ]},
    {"name": "prof_19",
     "build": ["My name is Leo, I am a network engineer",
               "I prefer Cisco IOS-XE over Junos for network OS",
               "I decided to use Ansible over Puppet for network automation",
               "I think WireGuard is better than OpenVPN for VPN",
               "I like using Nornir over Netmiko for network scripting"],
     "questions": [
        {"q": "What network OS do I prefer?",            "expect": "cisco"},
        {"q": "What automation tool do I use?",          "expect": "ansible"},
        {"q": "What VPN solution do I prefer?",          "expect": "wireguard"},
        {"q": "What scripting framework do I use?",      "expect": "nornir"},
        {"q": "What kind of engineer am I?",             "expect": "network"},
    ]},
    {"name": "prof_20",
     "build": ["My name is Mona, I am a data analyst",
               "I prefer Tableau over Power BI for dashboards",
               "I decided to use Looker over Mode for analytics",
               "I think SQL is more important than Python for analysis",
               "I like using dbt for metric definitions"],
     "questions": [
        {"q": "What dashboard tool do I prefer?",        "expect": "tableau"},
        {"q": "What analytics platform do I use?",       "expect": "looker"},
        {"q": "What is my view on SQL vs Python?",       "expect": "sql"},
        {"q": "What metric tool do I use?",              "expect": "dbt"},
        {"q": "What is my analyst role?",                "expect": "analyst"},
    ]},
    {"name": "prof_21",
     "build": ["My name is Nathan, I am a backend developer using Elixir",
               "I prefer Phoenix over Rails for web frameworks",
               "I decided to use LiveView over React for real-time UI",
               "I think OTP is the best concurrency model",
               "I like using Ecto over ActiveRecord for database queries"],
     "questions": [
        {"q": "What web framework do I prefer?",         "expect": "phoenix"},
        {"q": "What real-time UI approach do I use?",    "expect": "liveview"},
        {"q": "What is my concurrency philosophy?",      "expect": "otp"},
        {"q": "What database query tool do I use?",      "expect": "ecto"},
        {"q": "What language do I use?",                 "expect": "elixir"},
    ]},
    {"name": "prof_22",
     "build": ["My name is Olivia, I am an iOS developer",
               "I prefer UIKit over SwiftUI for complex apps",
               "I decided to use SnapKit over Auto Layout for constraints",
               "I think Combine is better than RxSwift for reactive",
               "I like using Fastlane for beta distribution"],
     "questions": [
        {"q": "What UI framework do I prefer for complex apps?", "expect": "uikit"},
        {"q": "What constraints tool do I use?",          "expect": "snapkit"},
        {"q": "What reactive framework do I prefer?",     "expect": "combine"},
        {"q": "What beta distribution tool do I use?",    "expect": "fastlane"},
        {"q": "What platform do I develop for?",          "expect": "ios"},
    ]},
    {"name": "prof_23",
     "build": ["My name is Peter, I am an Android developer",
               "I prefer Kotlin over Java for Android",
               "I decided to use Jetpack Compose over XML layouts",
               "I think Hilt is better than Dagger for DI",
               "I like using Room over Realm for local storage"],
     "questions": [
        {"q": "What language do I prefer for Android?",   "expect": "kotlin"},
        {"q": "What UI toolkit do I use?",                "expect": "compose"},
        {"q": "What DI framework do I prefer?",           "expect": "hilt"},
        {"q": "What local storage do I use?",             "expect": "room"},
        {"q": "What platform do I develop for?",          "expect": "android"},
    ]},
    {"name": "prof_24",
     "build": ["My name is Quinn, I am a Rust developer",
               "I prefer Tokio over async-std for async runtime",
               "I decided to use Axum over Actix for web servers",
               "I think Serde is the best serialization library",
               "I like using Cargo workspaces for monorepos"],
     "questions": [
        {"q": "What async runtime do I prefer?",         "expect": "tokio"},
        {"q": "What web framework do I use?",             "expect": "axum"},
        {"q": "What serialization library do I like?",    "expect": "serde"},
        {"q": "What monorepo approach do I use?",         "expect": "cargo"},
        {"q": "What language do I use?",                  "expect": "rust"},
    ]},
    {"name": "prof_25",
     "build": ["My name is Rita, I am a Go developer",
               "I prefer Gin over Echo for HTTP routing",
               "I decided to use GORM over sqlx for ORM",
               "I think context package is essential for Go concurrency",
               "I like using Cobra for CLI applications"],
     "questions": [
        {"q": "What HTTP router do I prefer?",           "expect": "gin"},
        {"q": "What ORM do I use?",                      "expect": "gorm"},
        {"q": "What concurrency primitive do I value?",  "expect": "context"},
        {"q": "What CLI framework do I use?",            "expect": "cobra"},
        {"q": "What language do I use?",                 "expect": "go"},
    ]},
    {"name": "prof_26",
     "build": ["My name as Sam, I am a Ruby developer",
               "I prefer Rails over Sinatra for web apps",
               "I decided to use Sidekiq over Resque for background jobs",
               "I think RSpec is better than Minitest for testing",
               "I like using Sorbet for type checking Ruby"],
     "questions": [
        {"q": "What web framework do I prefer?",         "expect": "rails"},
        {"q": "What background job system do I use?",    "expect": "sidekiq"},
        {"q": "What testing framework do I prefer?",     "expect": "rspec"},
        {"q": "What type checker do I use?",             "expect": "sorbet"},
        {"q": "What language do I use?",                 "expect": "ruby"},
    ]},
    {"name": "prof_27",
     "build": ["My name is Tina, I am a Python backend developer",
               "I prefer FastAPI over Flask for APIs",
               "I decided to use SQLAlchemy over Django ORM",
               "I think Celery is better than RQ for task queues",
               "I like using Pydantic for data validation"],
     "questions": [
        {"q": "What API framework do I prefer?",         "expect": "fastapi"},
        {"q": "What ORM do I use?",                      "expect": "sqlalchemy"},
        {"q": "What task queue do I prefer?",            "expect": "celery"},
        {"q": "What validation library do I use?",       "expect": "pydantic"},
        {"q": "What language do I use?",                 "expect": "python"},
    ]},
    {"name": "prof_28",
     "build": ["My name is Uma, I am a JavaScript frontend developer",
               "I prefer React over Vue for UI",
               "I decided to use Zustand over Redux for state",
               "I think TanStack Query is better than SWR for data fetching",
               "I like using Vite over Webpack for bundling"],
     "questions": [
        {"q": "What UI library do I prefer?",            "expect": "react"},
        {"q": "What state manager do I use?",            "expect": "zustand"},
        {"q": "What data fetching library do I prefer?", "expect": "tanstack"},
        {"q": "What bundler do I use?",                  "expect": "vite"},
        {"q": "What language do I use?",                 "expect": "javascript"},
    ]},
    {"name": "prof_29",
     "build": ["My name is Victor, I am a Scala developer",
               "I prefer Cats Effect over ZIO for functional effects",
               "I decided to use Akka over Pekko for actor systems",
               "I think SBT is better than Mill for builds",
               "I like using Tapir over Http4s for API definitions"],
     "questions": [
        {"q": "What effect system do I prefer?",         "expect": "cats"},
        {"q": "What actor framework do I use?",          "expect": "akka"},
        {"q": "What build tool do I prefer?",            "expect": "sbt"},
        {"q": "What API library do I use?",              "expect": "tapir"},
        {"q": "What language do I use?",                 "expect": "scala"},
    ]},
    {"name": "prof_30",
     "build": ["My name is Wendy, I am a PHP developer",
               "I prefer Laravel over Symfony for web apps",
               "I decided to use Filament over Nova for admin panels",
               "I think Pest is better than PHPUnit for testing",
               "I like using Octane over Horizon for performance"],
     "questions": [
        {"q": "What PHP framework do I prefer?",         "expect": "laravel"},
        {"q": "What admin panel do I use?",              "expect": "filament"},
        {"q": "What testing framework do I prefer?",     "expect": "pest"},
        {"q": "What performance tool do I use?",         "expect": "octane"},
        {"q": "What language do I use?",                 "expect": "php"},
    ]},
]

# ── Communication functions with retry ──────────────────────────────────────────
def ask_proxy(user_id, question):
    """A-group: via Noesis proxy (with memory injection)."""
    for attempt in range(4):
        try:
            r = httpx.post(PROXY_URL, headers={
                "Authorization": f"Bearer {GEMINI_KEY}",
                "X-User-ID": user_id, "Content-Type": "application/json",
            }, json={"model": MODEL, "messages":[{"role":"user","content":question}],
                    "max_tokens": 150, "stream": False}, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(12 * (attempt + 1)); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"

def ask_direct(question):
    """B-group: direct to Gemini (no memory)."""
    for attempt in range(4):
        try:
            r = httpx.post(GEMINI_URL, headers={
                "Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json",
            }, json={"model": MODEL, "messages":[{"role":"user","content":question}],
                    "max_tokens": 150, "stream": False}, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(12 * (attempt + 1)); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"

def build_memory(user_id, statements):
    """Build memories for a profile via proxy."""
    for stmt in statements:
        ask_proxy(user_id, stmt)
        time.sleep(1.5)

def clear_user(user_id):
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    db.execute("DELETE FROM items WHERE user_id=?", (user_id,))
    db.commit(); db.close()

def count_memory(user_id):
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (user_id,)).fetchone()[0]
    db.close(); return n

# ── Main ────────────────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("  English A/B: Noesis (memory) vs Direct (no memory) — 50 questions")
    print("=" * 72)
    print(f"  Profiles: 10, Questions each: 5, Total: 50")
    print(f"  Model: {MODEL}")
    print("=" * 72)

    summary = {"with_hit": 0, "without_hit": 0, "total": 0, "errors": 0}
    all_results = []

    for pi, profile in enumerate(PROFILES, 1):
        uid = profile["name"]
        print(f"\n[{pi}/{len(PROFILES)}] Profile: {uid}")
        clear_user(uid)
        print(f"  Building memory ({len(profile['build'])} stmts)...", end=" ", flush=True)
        build_memory(uid, profile["build"])
        time.sleep(3)
        n = count_memory(uid)
        print(f"done, {n} memories")

        for q_info in profile["questions"]:
            q, expect = q_info["q"], q_info["expect"]

            ok_a, ans_a = ask_proxy(uid, q)
            hit_a = expect.lower() in ans_a.lower() if ok_a else False
            time.sleep(2)

            ok_b, ans_b = ask_direct(q)
            hit_b = expect.lower() in ans_b.lower() if ok_b else False
            time.sleep(2)

            summary["total"] += 1
            if hit_a: summary["with_hit"] += 1
            if hit_b: summary["without_hit"] += 1
            if not ok_a or not ok_b: summary["errors"] += 1

            all_results.append({
                "profile": uid, "question": q, "expect": expect,
                "with_mem_hit": hit_a, "without_mem_hit": hit_b,
                "with_mem_ans": ans_a[:100], "without_mem_ans": ans_b[:100],
            })

            ma = "✅" if hit_a else "❌"
            mb = "✅" if hit_b else "❌"
            diff = "🟢" if (hit_a and not hit_b) else ("🟡" if hit_a==hit_b else "🔴")
            print(f"    [{diff}] {q[:36]:38s} mem{ma} direct{mb} ('{expect}')")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FINAL SUMMARY")
    print(f"{'='*72}\n")
    wh = summary["with_hit"]; woh = summary["without_hit"]; tot = summary["total"]
    err = summary["errors"]
    print(f"  With memory (Noesis):    {wh}/{tot} ({100*wh/tot:.1f}%)")
    print(f"  Without memory (direct): {woh}/{tot} ({100*woh/tot:.1f}%)")
    print(f"  Improvement:             +{wh-woh} ({100*(wh-woh)/tot:.1f}pp)")
    print(f"  Errors (429/network):    {err}")
    print()

    # Chi-square test for significance
    from math import comb
    print("  Significance (McNemar's test approximation):")
    # Discordant pairs: with-hit/no-miss (b) vs with-miss/no-hit (c)
    b = sum(1 for r in all_results if r["with_mem_hit"] and not r["without_mem_hit"])
    c = sum(1 for r in all_results if not r["with_mem_hit"] and r["without_mem_hit"])
    print(f"    Discordant pairs: mem+direct- = {b}, mem-direct+ = {c}")
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
        print(f"    Chi-square (McNemar): {chi2:.2f}  (>3.84 = significant at p<0.05)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"experiment": "ab_comparison_en", "language": "en", "model": MODEL,
              "timestamp": ts, "num_questions": tot, "summary": summary,
              "discordant": {"b": b, "c": c}, "details": all_results}
    path = RESULTS_DIR / f"ab_comparison_en_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 {path}")

if __name__ == "__main__":
    if not GEMINI_KEY:
        print("Error: set GEMINI_API_KEY"); sys.exit(1)
    run()
