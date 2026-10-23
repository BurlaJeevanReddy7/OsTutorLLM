import json
import re
import urllib.request
from typing import List, Dict, Any

INDEX_FILE = "./chroma_db_lite.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"  # ~1.1GB RAM, vastly superior reasoning

# ==========================================
# 1. RETRIEVAL AGENT
# ==========================================
class RetrievalAgent:
    def __init__(self, index_path: str = INDEX_FILE):
        self.index_path = index_path
        self.documents = self.load_index()

    def load_index(self) -> List[Dict[str, str]]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def retrieve(self, query: str, top_k: int = 2, module_filter: str = None) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\w+", query.lower()))
        stop_words = {
            "can", "u", "you", "teach", "me", "what", "whats", "is", "a", "an",
            "the", "in", "of", "and", "or", "to", "for", "find", "explain",
            "tell", "about", "how", "why", "hello", "hi", "hey"
        }
        key_words = query_words - stop_words
        if not key_words or not self.documents:
            return []

        scored = []
        for doc in self.documents:
            if module_filter and module_filter.lower() not in doc.get("module", "").lower():
                continue
            text = doc["content"].lower()
            score = sum(text.count(w) for w in key_words)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


# ==========================================
# 2. DETERMINISTIC SOLVER AGENT
# ==========================================
class MathEngineAgent:
    @staticmethod
    def solve_fcfs(burst_times: List[int]) -> str:
        n = len(burst_times)
        ct, tat, wt = [0]*n, [0]*n, [0]*n
        current_time = 0
        gantt_chart = "0"

        for i in range(n):
            current_time += burst_times[i]
            ct[i] = current_time
            tat[i] = ct[i]
            wt[i] = tat[i] - burst_times[i]
            gantt_chart += f" | P{i+1} | {current_time}"

        avg_tat = sum(tat) / n
        avg_wt = sum(wt) / n

        res = "### FCFS Scheduling Calculation ($AT = 0$)\n\n"
        res += "**Visual Gantt Chart:**\n```text\n" + gantt_chart + "\n```\n\n"
        res += "| Process | Burst Time (BT) | Completion Time (CT) | Turnaround Time (TAT) | Waiting Time (WT) |\n"
        res += "|:---:|:---:|:---:|:---:|:---:|\n"
        for i in range(n):
            res += f"| $P_{i+1}$ | {burst_times[i]} | {ct[i]} | {tat[i]} | {wt[i]} |\n"

        res += f"\n- **Average Turnaround Time (TAT)**: $\\frac{{{sum(tat)}}}{{{n}}} = {avg_tat:.2f}$\n"
        res += f"- **Average Waiting Time (WT)**: $\\frac{{{sum(wt)}}}{{{n}}} = {avg_wt:.2f}$\n"
        return res

    @staticmethod
    def solve_bankers_sample() -> str:
        alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
        max_m = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
        avail = [3, 3, 2]
        n, m = len(alloc), len(avail)

        need = [[max_m[i][j] - alloc[i][j] for j in range(m)] for i in range(n)]
        finish = [False] * n
        safe_seq = []
        work = list(avail)
        log = []

        while len(safe_seq) < n:
            found = False
            for i in range(n):
                if not finish[i]:
                    if all(need[i][j] <= work[j] for j in range(m)):
                        work = [work[j] + alloc[i][j] for j in range(m)]
                        finish[i] = True
                        safe_seq.append(f"P{i}")
                        log.append(f"- **Step {len(safe_seq)}**: $P_{i}$ allocated. Work = `{work}`")
                        found = True
                        break
            if not found:
                break

        res = "### Banker's Algorithm: Safe State Verification\n\n"
        res += "**Calculated Need Matrix:** $(\\text{Need} = \\text{Max} - \\text{Allocation})$\n\n"
        res += "| Process | Allocation | Max | Need |\n|:---:|:---:|:---:|:---:|\n"
        for i in range(n):
            res += f"| $P_{i}$ | [{' '.join(map(str, alloc[i]))}] | [{' '.join(map(str, max_m[i]))}] | **[{' '.join(map(str, need[i]))}]** |\n"

        res += f"\n**Initial Available:** `[{' '.join(map(str, avail))}]`\n\n"
        res += "**Steps:**\n" + "\n".join(log) + "\n\n"
        if len(safe_seq) == n:
            res += f"**Result**: Safe State. Sequence: $\\langle " + ", ".join(safe_seq) + " \\rangle$"
        else:
            res += "**Result**: Unsafe State (Deadlock risk)."
        return res


# ==========================================
# 3. PEDAGOGICAL TUTOR (Direct Persona Grounding)
# ==========================================
class PedagogicalTutorAgent:
    @staticmethod
    def converse(query: str, history: List[Dict[str, str]], retrieved_context: List[Dict[str, Any]]) -> str:
        q_clean = query.strip().lower()

        # Handle simple greetings directly
        if q_clean in ["hello", "hi", "hey", "hello!"]:
            return "Hey! Ready to dive into Operating Systems. What topic are you working on right now—processes, scheduling, memory, or deadlocks?"

        if q_clean in ["can u help me", "can you help me", "can u teach me", "help me learn"]:
            return "Of course! Let's get started. We can work through concepts like system calls (`fork`, `exec`), scheduling algorithms (FCFS, Round Robin), virtual memory, or practice exam questions. What would you like to cover first?"

        context_str = ""
        if retrieved_context:
            context_str = "\nRelevant Course Material:\n" + "\n".join([f"- {c['content'][:400]}" for c in retrieved_context])

        system_instruction = (
            "You are an authentic, direct, and knowledgeable Operating Systems tutor for university students. "
            "Rules:\n"
            "1. Respond directly to the user's prompt. Do NOT generate scripted multi-turn dialogues (never output 'Tutor:', 'Student:', or 'User:').\n"
            "2. Focus exclusively on Operating Systems computer science concepts.\n"
            "3. If explaining a concept, use practical intuition and code examples where appropriate.\n"
            "4. If calculating turnaround time, waiting time, or paging, do the exact math step-by-step.\n"
            "5. Use standard Markdown formatting."
        )

        formatted_dialogue = ""
        for h in history[-2:]:
            role = "Student" if h.get("role") in ["Student", "user"] else "Tutor"
            formatted_dialogue += f"{role}: {h['content']}\n"

        prompt = f"{system_instruction}\n\n{formatted_dialogue}Student: {query}\n{context_str}\nTutor:"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 400,
                "stop": ["Student:", "Tutor:", "\nStudent", "\nTutor"]
            }
        }

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError:
            return "Ollama is not running. Please start it using: `ollama run qwen2.5:1.5b`"
        except Exception as e:
            return f"Inference fault: {str(e)}"


# ==========================================
# 4. MASTER ORCHESTRATOR
# ==========================================
class OSTutorEngine:
    def __init__(self, index_path: str = INDEX_FILE):
        self.retriever = RetrievalAgent(index_path)
        self.math_agent = MathEngineAgent()
        self.tutor_agent = PedagogicalTutorAgent()

    @property
    def documents(self):
        return self.retriever.documents

    @documents.setter
    def documents(self, val):
        self.retriever.documents = val

    def retrieve(self, query: str, top_k: int = 2, module_filter: str = None):
        return self.retriever.retrieve(query, top_k, module_filter)

    def generate_response(self, user_query: str, history: List[Dict[str, str]] = None, module_filter: str = None) -> Dict[str, Any]:
        if history is None:
            history = []

        lower_q = user_query.lower()
        nums = [int(x) for x in re.findall(r"\b\d+\b", user_query)]

        if "banker" in lower_q and any(term in lower_q for term in ["solve", "run", "calculate", "safe sequence"]):
            answer = self.math_agent.solve_bankers_sample()
            return {"answer": answer, "retrieved_context": [], "agent_used": "MathEngineAgent (Banker's Algorithm)"}

        if ("fcfs" in lower_q or "burst" in lower_q) and "calculate" in lower_q and len(nums) >= 2:
            answer = self.math_agent.solve_fcfs(nums)
            return {"answer": answer, "retrieved_context": [], "agent_used": "MathEngineAgent (CPU Scheduler)"}

        retrieved_chunks = self.retriever.retrieve(user_query, top_k=2, module_filter=module_filter)
        answer = self.tutor_agent.converse(user_query, history, retrieved_chunks)

        return {
            "answer": answer,
            "retrieved_context": retrieved_chunks,
            "agent_used": f"OSTutor Core ({OLLAMA_MODEL})"
        }