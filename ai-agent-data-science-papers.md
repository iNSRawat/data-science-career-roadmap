# 🤖 Autonomous AI Agents for Data Science & Scientific Discovery

> A curated collection of 6 must-read white papers exploring how autonomous AI agents are transforming data science, ML engineering, and scientific research.

---

## Table of Contents

1. [DeepAnalyze: Autonomous Data Science via Agentic LLMs](#1-deepanalyze-autonomous-data-science-via-agentic-llms)
2. [AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench](#2-ai-research-agents-for-machine-learning-search-exploration-and-generalization-in-mle-bench)
3. [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](#3-the-ai-scientist-towards-fully-automated-open-ended-scientific-discovery)
4. [Jupiter: Enhancing LLM Data Analysis via Notebook and Inference-Time Value-Guided Search](#4-jupiter-enhancing-llm-data-analysis-via-notebook-and-inference-time-value-guided-search)
5. [DS-STAR: A State-of-the-Art Versatile Data Science Agent](#5-ds-star-a-state-of-the-art-versatile-data-science-agent)
6. [Kosmos: An AI Scientist for Autonomous Discovery](#6-kosmos-an-ai-scientist-for-autonomous-discovery)

---

## 1. DeepAnalyze: Autonomous Data Science via Agentic LLMs

**Summary:** Presents DeepAnalyze-8B, an agentic LLM trained to perform the full data science pipeline — from raw data processing to generating analytical reports — using a curriculum-based training process that mimics how human analysts learn.

**Key Highlights:**
- Automates data preparation, analysis, modeling, visualization, and report generation
- Uses curriculum-based agentic training paradigm (simple → complex tasks)
- Introduces data-grounded trajectory synthesis for high-quality training traces
- Outperforms prior agents built on stronger proprietary models despite only 8B parameters

**Paper:** [https://arxiv.org/abs/2510.16872](https://arxiv.org/abs/2510.16872)

---

## 2. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

**Summary:** Formalizes AI research agents as search policies that navigate the space of ML solutions and evaluates them on MLE-bench — a benchmark of real Kaggle competition tasks — using strategies like Monte Carlo Tree Search and evolutionary algorithms.

**Key Highlights:**
- Agents iteratively generate and evaluate ML solutions using MCTS and evolutionary algorithms
- Introduces AIRA-dojo, a scalable framework for running AI research agents
- Achieves state-of-the-art medal rates on MLE-bench Lite
- Shows interplay between operator design and search policy is critical to performance

**Paper:** [https://arxiv.org/abs/2507.02554](https://arxiv.org/abs/2507.02554)

---

## 3. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

**Summary:** Introduces an AI system that can autonomously run the entire ML research process — from generating ideas to running experiments and writing full research papers — treating research as an open-ended, iterative process.

**Key Highlights:**
- End-to-end automation of ML research: idea generation → experiment design → implementation → paper writing
- Multiple coordinated agents maintain coherent research trajectories over extended periods
- Significantly reduces human effort needed per research idea
- Demonstrates open-ended scientific exploration beyond fixed benchmarks

**Paper:** [https://arxiv.org/abs/2408.06292](https://arxiv.org/abs/2408.06292)

---

## 4. Jupiter: Enhancing LLM Data Analysis via Notebook and Inference-Time Value-Guided Search

**Summary:** Treats notebook-based data analysis as a sequential decision-making problem. Introduces NbQA (a large Jupyter notebook task dataset) and uses Monte Carlo Tree Search with a value model to guide LLMs through complex multi-step analysis.

**Key Highlights:**
- Builds NbQA: a large dataset of real Jupyter notebook tasks with multi-step solutions
- Applies MCTS to generate diverse solution trajectories for training a value model
- At inference time, combines value scores and search statistics for high-quality plans
- Enables open-source LLMs to match commercial agent systems on notebook analysis tasks

**Paper:** [https://arxiv.org/abs/2509.09245](https://arxiv.org/abs/2509.09245)

---

## 5. DS-STAR: A State-of-the-Art Versatile Data Science Agent

**Summary:** A Google research agent designed for real-world data science workflows. It integrates heterogeneous data formats (CSV, JSON, text), decomposes questions into sub-tasks, and generates evidence-based analytical reports with LLM-based verification.

**Key Highlights:**
- Seamlessly processes CSV, JSON, text and other heterogeneous data formats
- Sequential planning + LLM-based verifier for iterative refinement and correctness
- Achieves SOTA on DABStep, DABStep-Research, KramaBench, and DA-Code benchmarks
- Human evaluators preferred its reports over strong baselines in 88%+ of cases

**Paper (arXiv):** [https://arxiv.org/abs/2509.21825](https://arxiv.org/abs/2509.21825)

**Blog Post:** [https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/)

---

## 6. Kosmos: An AI Scientist for Autonomous Discovery

**Summary:** An AI scientist built around a structured world model and multi-agent architecture, capable of running long autonomous research cycles. It coordinates literature search and data analysis agents to make scientific discoveries over 12+ hour sessions.

**Key Highlights:**
- Takes an open-ended research objective + dataset and autonomously runs full research cycles
- Coordinates specialized agents (literature, data analysis) through a central world model
- Produces traceable scientific reports enabling verification and reuse of findings
- Shows approximately linear scaling of discovery output with runtime

**Paper:** [https://arxiv.org/abs/2511.02824](https://arxiv.org/abs/2511.02824)

---

## How These Papers Connect to the Data Science Roadmap

These 6 papers sit directly at the frontier of the **Agentic AI / AutoML** trends shaping the future of the profession:

| Paper | Relevance to Data Scientists |
|-------|------------------------------|
| DeepAnalyze | Automates the full DS pipeline — from EDA to reports |
| MLE-bench Agents | Shows AI competing in real Kaggle tasks using search algorithms |
| The AI Scientist | End-to-end ML research automation |
| Jupiter | Automates Jupyter notebook-based analysis workflows |
| DS-STAR | Real-world, multi-format data science at SOTA level |
| Kosmos | Long-horizon autonomous scientific discovery |

> Understanding these systems will help you position yourself as a data scientist who can **orchestrate and collaborate with AI agents** — not just use traditional tools.

---

## Related Sections in This Roadmap

- [Future Trends (2025-2030)](./README.md#-future-trends-2025-2030)
- [Phase 4: Specialization](./README.md#-phase-4-specialization-months-10-12)

---

*Last updated: March 2026*
