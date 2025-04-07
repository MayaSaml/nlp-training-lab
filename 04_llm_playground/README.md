# LLM Playground: Practical Workflows & Demos

Topics covered:
- Tokens and Embeddings
- Text Classification
- Text Clustering
- Topic Modeling
- Text Generation
- Creating Text Embedding Models
- Semantic search with sentence embeddings
- Prompt engineering variations
- Fine-tuning small LLMs on custom tasks
- Document Q&A with retrieval-augmented generation (RAG)
- Systematic comparison of open-source LLMs for reasoning tasks

Frameworks:
- OpenAI API, HuggingFace Transformers, LangChain, FAISS, PyTorch

Educational Intent:
This notebook series is part of a broader effort to **share applied LLM knowledge** in a way that’s practical and actionable. These labs are intended for other engineers and data scientists building modern NLP and LLM-based systems.

**Note**: This section is inspired by the book *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst — a phenomenal resource for applied large language models. I’ve extended several labs with my own experiments and explanations.


## Global LLM Landscape (2024–2025)

Here’s a high-level summary of the top Large Language Model developers and ecosystems worldwide — focused on real-world impact, innovation, and openness.

---

Summary Table of Key LLM Players

| Region | Company / Lab | Flagship Models | Openness | Focus / Notes |
|----------|------------------|--------------------|-------------|------------------|
| 🇺🇸 USA | **OpenAI** | GPT-4 / GPT-4 Turbo | ❌ Closed | Market leader, high accuracy, API-first. Powers ChatGPT, Bing, Copilot |
| 🇺🇸 USA | **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | ❌ Closed | Strong alignment + reasoning, safety-focused, growing fast |
| 🇺🇸 USA | **Google DeepMind** | Gemini 1.5 | ❌ Closed | Multimodal, native to search, integrated into Google tools |
| 🇺🇸 USA | **Meta AI** | LLaMA 2 / LLaMA 3 (soon) | ✅ Open | Foundation for many open models, strong multilingual benchmarks |
| 🇫🇷 EU (France) | **Mistral AI** | Mistral 7B, Mixtral 8x7B | ✅ Open | Top open-source models, efficient + high performing, very active |
| 🇩🇪 EU (Germany) | **Aleph Alpha** | Luminous | ❌ Closed | Enterprise + explainable AI focus, compliant with EU regulations |
| 🇨🇳 China | **Zhipu AI** | ChatGLM3 | ✅ Open | Chinese-English, top-tier academic + community use |
| 🇨🇳 China | **DeepSeek** | DeepSeek-Coder, DeepSeek-VL | ✅ Open | Rapid rise, open-weight models for code, vision-language, and instruction |

---

Highlights

- **OpenAI / Anthropic / Google**: Still dominate cutting-edge closed models used in products.
- **Meta & Mistral**: Power the **open-source ecosystem** — widely adopted in fine-tuning, RAG, chatbots, etc.
- **DeepSeek & Zhipu AI**: Leading China's charge into open-source LLMs.
- **Mistral (France)**: Arguably the **#1 open LLM lab in the world** right now in terms of performance and code release quality.

---

