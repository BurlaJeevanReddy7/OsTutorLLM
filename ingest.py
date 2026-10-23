import os
import glob
import math
import json
import re
from collections import Counter

DATA_DIR = "./data"
INDEX_FILE = "./chroma_db_lite.json"

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def extract_pptx_text(filepath):
    prs = Presentation(filepath)
    text_runs = []
    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text_runs.append(f"[Slide {slide_idx + 1}] {paragraph.text}")
    return "\n".join(text_runs)

def load_documents():
    docs = []
    for txt_file in glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.md")):
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                docs.append({"content": f.read(), "source": txt_file, "module": "General"})
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")

    if HAS_PPTX:
        for pptx_file in glob.glob(os.path.join(DATA_DIR, "*.pptx")):
            try:
                docs.append({"content": extract_pptx_text(pptx_file), "source": pptx_file, "module": "Lecture Slides"})
            except Exception as e:
                print(f"Error loading {pptx_file}: {e}")

    return docs

def split_text(text, max_chars=1200, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def build_vector_store():
    print("Loading documents from ./data...")
    raw_docs = load_documents()

    if not raw_docs:
        print("No files found in ./data. Seeding default BCSE303L core OS knowledge...")
        raw_docs = [
            {
                "content": "CPU Scheduling Metrics: Turnaround Time = Completion Time - Arrival Time. Waiting Time = Turnaround Time - Burst Time. Response Time = First CPU response - Arrival Time. FCFS, SJF, and Round Robin algorithms allocate CPU time based on priority or quantum.",
                "source": "BCSE303L_Module2_Scheduling",
                "module": "Module 2"
            },
            {
                "content": "Banker's Algorithm: Deadlock avoidance algorithm. Need Matrix = Max Matrix - Allocation Matrix. A system is in a safe state if there exists a safe sequence where available resources satisfy Need: Work = Work + Allocation.",
                "source": "BCSE303L_Module3_Deadlocks",
                "module": "Module 3"
            },
            {
                "content": "Page Replacement Algorithms: FIFO replaces the oldest page in memory. LRU replaces the page not referenced for the longest duration. Optimal replaces the page that will not be used for the longest period in future. Belady's Anomaly affects FIFO.",
                "source": "BCSE303L_Module4_Memory",
                "module": "Module 4"
            },
            {
                "content": "Process Synchronization: Critical Section Problem requires Mutual Exclusion, Progress, and Bounded Waiting. Peterson's Solution uses flag boolean array and a turn integer variable to resolve two-process synchronization.",
                "source": "BCSE303L_Module1_Processes",
                "module": "Module 1"
            }
        ]

    chunked_docs = []
    for doc in raw_docs:
        splits = split_text(doc["content"])
        for chunk in splits:
            if chunk.strip():
                chunked_docs.append({
                    "content": chunk.strip(),
                    "source": doc["source"],
                    "module": doc.get("module", "Core OS")
                })

    print(f"Total chunks indexed: {len(chunked_docs)}")
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, indent=2)

    print("Vector database built successfully!")

if __name__ == "__main__":
    build_vector_store()