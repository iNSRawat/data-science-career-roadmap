# 🤖 Autonomous AI Agents for Data Science & Scientific Discovery

> A curated collection of 11 must-read white papers and research exploring how autonomous AI agents are transforming data science, ML engineering, and scientific research.

---

## Table of Contents

1. [DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](#1-deepanalyze-agentic-large-language-models-for-autonomous-data-science)
2. [AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench](#2-ai-research-agents-for-machine-learning-search-exploration-and-generalization-in-mle-bench)
3. [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](#3-the-ai-scientist-towards-fully-automated-open-ended-scientific-discovery)
4. [Jupiter: Enhancing LLM Data Analysis via Notebook and Inference-Time Value-Guided Search](#4-jupiter-enhancing-llm-data-analysis-via-notebook-and-inference-time-value-guided-search)
5. [DS-STAR: A State-of-the-Art Versatile Data Science Agent](#5-ds-star-a-state-of-the-art-versatile-data-science-agent)
6. [Kosmos: An AI Scientist for Autonomous Discovery](#6-kosmos-an-ai-scientist-for-autonomous-discovery)
7. [AIDE: AI-Driven Exploration in the Space of Code](#7-aide-ai-driven-exploration-in-the-space-of-code)
8. [AutoKaggle: Multi-Agent Framework for Autonomous Data Science](#8-autokaggle-multi-agent-framework-for-autonomous-data-science)
9. [MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation](#9-mlagentbench-evaluating-language-agents-on-machine-learning-experimentation)
10. [Large Language Model-based Data Science Agent: A Survey](#10-large-language-model-based-data-science-agent-a-survey)
11. [DSBench & DataSciBench: Benchmarking Data Science Agents](#11-dsbench--datascibench-benchmarking-data-science-agents)

---

## 1. DeepAnalyze: Agentic Large Language Models for Autonomous Data Science

**Summary:** Presents DeepAnalyze-8B, an agentic LLM trained to perform the full data science pipeline — from raw data processing to generating analytical reports — using a curriculum-based training process that mimics how human analysts learn.

**Key Highlights:**
- Automates data preparation, analysis, modeling, visualization, and report generation
- Uses curriculum-based agentic training paradigm (simple → complex tasks)
- Introduces data-grounded trajectory synthesis for high-quality training traces
- Outperforms prior agents built on stronger proprietary models despite only 8B parameters

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2510.16872](https://arxiv.org/abs/2510.16872)
- 📄 **PDF:** [https://arxiv.org/pdf/2510.16872](https://arxiv.org/pdf/2510.16872)
- 🐞 **GitHub:** [https://github.com/ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze)

---

## 2. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

**Summary:** Formalizes AI research agents as search policies that navigate the space of ML solutions and evaluates them on MLE-bench — a benchmark of real Kaggle competition tasks — using strategies like Monte Carlo Tree Search and evolutionary algorithms.

**Key Highlights:**
- Agents iteratively generate and evaluate ML solutions using MCTS and evolutionary algorithms
- Introduces AIRA-dojo, a scalable framework for running AI research agents
- Achieves state-of-the-art medal rates on MLE-bench Lite
- Shows interplay between operator design and search policy is critical to performance

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2507.02554](https://arxiv.org/abs/2507.02554)
- 📄 **PDF:** [https://arxiv.org/pdf/2507.02554](https://arxiv.org/pdf/2507.02554)
- 🐞 **GitHub (AIRA-dojo):** [https://github.com/facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo)

---

## 3. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

**Summary:** Introduces an AI system by Sakana AI that can autonomously run the entire ML research process — from generating ideas to running experiments and writing full research papers — treating research as an open-ended, iterative process.

**Key Highlights:**
- End-to-end automation of ML research: idea generation → experiment design → implementation → paper writing
- Multiple coordinated agents maintain coherent research trajectories over extended periods
- Significantly reduces human effort needed per research idea (~$15 per paper)
- Demonstrates open-ended scientific exploration beyond fixed benchmarks

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2408.06292](https://arxiv.org/abs/2408.06292)
- 📄 **PDF:** [https://arxiv.org/pdf/2408.06292](https://arxiv.org/pdf/2408.06292)
- 🐞 **GitHub (SakanaAI):** [https://github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
- 🌐 **Project Page:** [https://sakana.ai/ai-scientist/](https://sakana.ai/ai-scientist/)

---

## 4. Jupiter: Enhancing LLM Data Analysis via Notebook and Inference-Time Value-Guided Search

**Summary:** Treats notebook-based data analysis as a sequential decision-making problem. Introduces NbQA (a large Jupyter notebook task dataset) and uses Monte Carlo Tree Search with a value model to guide LLMs through complex multi-step analysis.

**Key Highlights:**
- Builds NbQA: a large dataset of real Jupyter notebook tasks with multi-step solutions
- Applies MCTS to generate diverse solution trajectories for training a value model
- At inference time, combines value scores and search statistics for high-quality plans
- Enables open-source LLMs to match commercial agent systems on notebook analysis tasks

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2509.09245](https://arxiv.org/abs/2509.09245)
- 📄 **PDF:** [https://arxiv.org/pdf/2509.09245](https://arxiv.org/pdf/2509.09245)
- 🐞 **GitHub (Microsoft):** [https://github.com/microsoft/Jupiter](https://github.com/microsoft/Jupiter)

---

## 5. DS-STAR: A State-of-the-Art Versatile Data Science Agent

**Summary:** A Google research agent designed for real-world data science workflows. It integrates heterogeneous data formats (CSV, JSON, text), decomposes questions into sub-tasks, and generates evidence-based analytical reports with LLM-based verification.

**Key Highlights:**
- Seamlessly processes CSV, JSON, text and other heterogeneous data formats
- Sequential planning + LLM-based verifier for iterative refinement and correctness
- Achieves SOTA on DABStep, DABStep-Research, KramaBench, and DA-Code benchmarks
- Human evaluators preferred its reports over strong baselines in 88%+ of cases

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2509.21825](https://arxiv.org/abs/2509.21825)
- 📄 **PDF:** [https://arxiv.org/pdf/2509.21825](https://arxiv.org/pdf/2509.21825)
- 📖 **Google Research Blog:** [https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/)

---

## 6. Kosmos: An AI Scientist for Autonomous Discovery

**Summary:** An AI scientist by FutureHouse / Edison Scientific built around a structured world model and multi-agent architecture, capable of running long autonomous research cycles that coordinate literature search and data analysis to make scientific discoveries.

**Key Highlights:**
- Takes an open-ended research objective + dataset and autonomously runs full research cycles (up to 12 hours)
- Coordinates specialized agents (literature, data analysis) through a central world model
- Produces traceable scientific reports enabling verification and reuse of findings
- Shows approximately linear scaling of discovery output with runtime
- 79.4% of conclusions verified accurate by independent scientists

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2511.02824](https://arxiv.org/abs/2511.02824)
- 📄 **PDF:** [https://arxiv.org/pdf/2511.02824](https://arxiv.org/pdf/2511.02824)
- 🌐 **Announcement (Edison Scientific):** [https://edisonscientific.com/articles/announcing-kosmos](https://edisonscientific.com/articles/announcing-kosmos)

---

## 7. AIDE: AI-Driven Exploration in the Space of Code

**Summary:** AIDE frames machine learning engineering as a code optimization problem and formulates trial-and-error as a tree search over the space of potential solutions. By strategically reusing and refining promising solutions, it achieves state-of-the-art results across multiple ML engineering benchmarks including OpenAI's MLE-bench.

**Key Highlights:**
- Frames ML engineering as a code optimization problem solved via tree search
- Systematically drafts, debugs, and refines solutions over extended time periods (up to 24h)
- Achieves 4x more Kaggle medals than runner-up agents on OpenAI's MLE-bench (75 tasks)
- Top-performing agent framework when paired with state-of-the-art LLMs
- Generalizes beyond tabular ML to neural architecture search and AI R&D tasks
- Fully open-source; available as `aideml` package on PyPI

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2502.13138](https://arxiv.org/abs/2502.13138)
- 📄 **PDF:** [https://arxiv.org/pdf/2502.13138](https://arxiv.org/pdf/2502.13138)
- 🐞 **GitHub (WecoAI):** [https://github.com/WecoAI/aideml](https://github.com/WecoAI/aideml)
- 📦 **PyPI:** [https://pypi.org/project/aideml/](https://pypi.org/project/aideml/)

---

## 8. AutoKaggle: Multi-Agent Framework for Autonomous Data Science

**Summary:** AutoKaggle proposes a collaborative multi-agent system for solving Kaggle data science competitions end-to-end. It uses a phase-based workflow with five specialized agents and iterative debugging with unit testing to ensure robust, correct code generation throughout the pipeline.

**Key Highlights:**
- Six-phase workflow: Background Understanding, EDA, Data Cleaning, Feature Engineering, Model Building, Report
- Five specialized agents: Reader, Planner, Developer, Reviewer, Summarizer
- Iterative code debugging and unit testing baked into each phase
- Achieves 0.85 validation submission rate and 0.82 comprehensive score on Kaggle competitions
- Universal ML tools library reduces burden of low-level programming tasks on agents
- Accepted at ICLR 2025

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2410.20424](https://arxiv.org/abs/2410.20424)
- 📄 **PDF:** [https://arxiv.org/pdf/2410.20424](https://arxiv.org/pdf/2410.20424)
- 🐞 **GitHub:** [https://github.com/multimodal-art-projection/AutoKaggle](https://github.com/multimodal-art-projection/AutoKaggle)
- 🌐 **Project Page:** [https://m-a-p.ai/AutoKaggle.github.io/](https://m-a-p.ai/AutoKaggle.github.io/)

---

## 9. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation

**Summary:** MLAgentBench introduces the first benchmark for evaluating LLM-based agents on machine learning experimentation tasks — covering a range from well-established datasets like CIFAR-10 to recent Kaggle challenges — to assess whether agents can autonomously conduct ML research end-to-end.

**Key Highlights:**
- 13 diverse ML tasks spanning text, image, time series, graphs, and tabular data
- General framework for specifying tasks with clear goals and automatic evaluation via performance metrics
- Demonstrates feasibility of LM-based agents for ML experimentation (37.5% average success rate)
- Success rates vary widely: 100% on established datasets, 0% on recent post-training Kaggle tasks
- Lays the groundwork for benchmarking autonomous ML research agents
- Open-source benchmark used by many subsequent works (AIDE, AIRA-dojo, etc.)

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2310.03302](https://arxiv.org/abs/2310.03302)
- 📄 **PDF:** [https://arxiv.org/pdf/2310.03302](https://arxiv.org/pdf/2310.03302)
- 🐞 **GitHub (Stanford CRFM):** [https://github.com/snap-stanford/MLAgentBench](https://github.com/snap-stanford/MLAgentBench)

---

## 10. Large Language Model-based Data Science Agent: A Survey

**Summary:** A comprehensive survey of LLM-based agents designed for data science tasks, analyzing insights from recent studies from two complementary perspectives: agent design (roles, execution, knowledge, reflection) and data science application (preprocessing, modeling, evaluation, visualization).

**Key Highlights:**
- Reviews 40+ LLM-based data science agent systems from 2023-2025
- Dual-perspective framework: agent design principles + data science workflow stages
- Covers single-agent systems, collaborative multi-agent structures, and dynamic agent generation
- Analyzes core components: planning, memory, tool use, reflection mechanisms
- Identifies open research challenges and future directions
- Excellent entry point to understand the entire landscape of DS agents

**Links:**
- 📜 **arXiv Paper:** [https://arxiv.org/abs/2508.02744](https://arxiv.org/abs/2508.02744)
- 📄 **PDF:** [https://arxiv.org/pdf/2508.02744](https://arxiv.org/pdf/2508.02744)

---

## 11. DSBench & DataSciBench: Benchmarking Data Science Agents

**Summary:** Two complementary benchmarks released in 2024-2025 for evaluating data science agents on realistic tasks: **DSBench** (466 data analysis tasks + 74 modeling tasks from Kaggle and Eloquence) and **DataSciBench** (comprehensive evaluation of LLM capabilities on data science with uncertain ground truth and complex prompts).

**Key Highlights:**
- **DSBench:** 466 data analysis tasks + 74 modeling tasks sourced from real competitions
- **DataSciBench:** Constructed from natural, challenging prompts with semi-automated evaluation pipeline
- Both assess multi-step reasoning, long-context handling, multi-table structures
- State-of-the-art agents solve only 34% of tasks, showing significant room for improvement
- Evaluation frameworks measure both coarse-grained (task completion) and fine-grained (intermediate steps) performance
- Identify critical gaps in current LLM/agent capabilities for data science

**Links:**
- 📜 **DSBench arXiv:** [https://arxiv.org/abs/2409.07703](https://arxiv.org/abs/2409.07703)
- 📜 **DataSciBench arXiv:** [https://arxiv.org/abs/2502.13897](https://arxiv.org/abs/2502.13897)
- 📄 **DSBench PDF:** [https://arxiv.org/pdf/2409.07703](https://arxiv.org/pdf/2409.07703)
- 📄 **DataSciBench PDF:** [https://arxiv.org/pdf/2502.13897](https://arxiv.org/pdf/2502.13897)

---

## Quick Reference: All Links

| # | Paper | arXiv | PDF | GitHub / Links |
|---|-------|-------|-----|----------------|
| 1 | DeepAnalyze | [2510.16872](https://arxiv.org/abs/2510.16872) | [PDF](https://arxiv.org/pdf/2510.16872) | [ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) |
| 2 | MLE-bench Agents | [2507.02554](https://arxiv.org/abs/2507.02554) | [PDF](https://arxiv.org/pdf/2507.02554) | [facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo) |
| 3 | The AI Scientist | [2408.06292](https://arxiv.org/abs/2408.06292) | [PDF](https://arxiv.org/pdf/2408.06292) | [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) |
| 4 | Jupiter | [2509.09245](https://arxiv.org/abs/2509.09245) | [PDF](https://arxiv.org/pdf/2509.09245) | [microsoft/Jupiter](https://github.com/microsoft/Jupiter) |
| 5 | DS-STAR | [2509.21825](https://arxiv.org/abs/2509.21825) | [PDF](https://arxiv.org/pdf/2509.21825) | [Google Blog](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/) |
| 6 | Kosmos | [2511.02824](https://arxiv.org/abs/2511.02824) | [PDF](https://arxiv.org/pdf/2511.02824) | [Edison Scientific](https://edisonscientific.com/articles/announcing-kosmos) |
| 7 | AIDE | [2502.13138](https://arxiv.org/abs/2502.13138) | [PDF](https://arxiv.org/pdf/2502.13138) | [WecoAI/aideml](https://github.com/WecoAI/aideml) |
| 8 | AutoKaggle | [2410.20424](https://arxiv.org/abs/2410.20424) | [PDF](https://arxiv.org/pdf/2410.20424) | [multimodal-art-projection](https://github.com/multimodal-art-projection/AutoKaggle) |
| 9 | MLAgentBench | [2310.03302](https://arxiv.org/abs/2310.03302) | [PDF](https://arxiv.org/pdf/2310.03302) | [snap-stanford/MLAgentBench](https://github.com/snap-stanford/MLAgentBench) |
| 10 | LLM DS Agent Survey | [2508.02744](https://arxiv.org/abs/2508.02744) | [PDF](https://arxiv.org/pdf/2508.02744) | N/A |
| 11 | DSBench & DataSciBench | [DSB](https://arxiv.org/abs/2409.07703) / [DSCIB](https://arxiv.org/abs/2502.13897) | [DSB](https://arxiv.org/pdf/2409.07703) / [DSCIB](https://arxiv.org/pdf/2502.13897) | N/A |

---

## How These Papers Connect to the Data Science Roadmap

These 11 papers sit directly at the frontier of the **Agentic AI / AutoML** trends shaping the future of the profession:

| Paper | Relevance to Data Scientists |
|-------|------------------------------|
| DeepAnalyze | Automates the full DS pipeline — from EDA to reports |
| MLE-bench Agents | Shows AI competing in real Kaggle tasks using search algorithms |
| The AI Scientist | End-to-end ML research automation |
| Jupiter | Automates Jupyter notebook-based analysis workflows |
| DS-STAR | Real-world, multi-format data science at SOTA level |
| Kosmos | Long-horizon autonomous scientific discovery |
| AIDE | Tree-search-based ML engineering agent; 4x MLE-bench medals |
| AutoKaggle | Multi-agent collaboration with phase-based workflow for Kaggle |
| MLAgentBench | First benchmark for ML experimentation agents (foundation work) |
| LLM DS Agent Survey | Comprehensive review of 40+ agent systems (2023-2025) |
| DSBench/DataSciBench | Realistic benchmarks revealing current limitations of agents |

> Understanding these systems will help you position yourself as a data scientist who can **orchestrate and collaborate with AI agents** — not just use traditional tools.

---

## Related Sections in This Roadmap

- [Future Trends (2025-2030)](./README.md#-future-trends-2025-2030)
- [Phase 4: Specialization](./README.md#-phase-4-specialization-months-10-12)

---

*Last updated: March 2026*
