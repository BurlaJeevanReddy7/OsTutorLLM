# OSTutorLLM: A Retrieval-Augmented Tutor for Operating Systems (BCSE303L)

**Team Members:**
- Shaik Mohammad Javad (24BCE2178)
- Burla Jeevan Reddy (24BCE2166)

## System Overview
OSTutorLLM is a curriculum-aware Operating Systems tutoring system built using Retrieval-Augmented Generation (RAG). It ingests lecture slides, lab manuals, and syllabus guidelines to provide structured explanations, multi-step numerical problem solving, and pedagogical guidance without hallucinating missing problem parameters.

## Architecture
- **Vector Retrieval**: Custom embedding vector index storing chunked course modules.
- **Backend Service**: FastAPI with endpoint `/api/chat` handling context retrieval and prompt construction.
- **Frontend Interface**: Responsive HTML5/CSS3 chat interface with marked.js for Markdown and KaTeX for mathematical notation rendering.
- **Guardrails**: Missing-value detection for scheduling, page replacement, and deadlock avoidance algorithms.

## Quickstart Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt