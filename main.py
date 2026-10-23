import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from tutor_backend import OSTutorEngine
from ingest import extract_pptx_text, split_text, INDEX_FILE, HAS_PPTX

app = FastAPI(title="OSTutorLLM")
engine = OSTutorEngine()
UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []
    module_filter: Optional[str] = None


@app.get("/api/status")
async def get_system_status():
    doc_count = len(engine.documents)
    unique_sources = len(set(d.get("source", "") for d in engine.documents if d.get("source")))
    return {
        "rag_chunks": doc_count,
        "sources": unique_sources,
        "model": "qwen2.5:1.5b",
        "engine": "Chroma-Lite",
        "solver": "Deterministic"
    }


@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history] if req.history else []
    return engine.generate_response(
        user_query=req.query,
        history=history_dicts,
        module_filter=req.module_filter
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = ""
        fn = file.filename.lower()
        if fn.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif fn.endswith(".pptx") and HAS_PPTX:
            text = extract_pptx_text(file_path)
        elif fn.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        text += f"\n[Page {idx+1}] " + t
            except Exception:
                pass

        if not text.strip():
            return JSONResponse({"status": "error", "message": "No selectable text found."}, status_code=400)

        new_chunks = split_text(text)
        current = engine.documents
        for c in new_chunks:
            if c.strip():
                current.append({"content": c.strip(), "source": file.filename, "module": "Uploaded Notes"})

        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

        engine.documents = current
        return {"status": "success", "filename": file.filename, "chunks_added": len(new_chunks)}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OSTUTOR</title>
        
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

        <style>
            :root {
                --bg: #0e0e11;
                --surface: rgba(26, 26, 30, 0.72);
                --border: rgba(255, 255, 255, 0.1);
                --text: #f1f3f9;
                --text-sub: #9aa0a6;
                --accent-blue: #70a5ff;
                --accent-green: #34d399;
                --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
                --mono: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }
            body { 
                background: var(--bg); 
                color: var(--text); 
                font-family: var(--font); 
                height: 100vh; 
                display: flex; 
                flex-direction: column; 
                overflow: hidden; 
            }

            ::-webkit-scrollbar { width: 5px; }
            ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 10px; }

            header {
                height: 52px;
                background: rgba(14, 14, 17, 0.88);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 28px;
                position: fixed;
                top: 0; left: 0; right: 0;
                z-index: 50;
            }
            .brand-group {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .brand { 
                font-size: 0.95rem; 
                font-weight: 600; 
                letter-spacing: -0.01em;
                background: linear-gradient(90deg, #70a5ff, #c38fff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .telemetry-row {
                display: flex;
                align-items: center;
                gap: 8px;
                overflow-x: auto;
            }
            .top-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: var(--mono);
                font-size: 0.72rem;
                color: var(--text-sub);
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border);
                padding: 4px 10px;
                border-radius: 9999px;
                white-space: nowrap;
            }
            .top-badge strong {
                color: var(--text);
                font-weight: 600;
            }
            .top-badge.online {
                color: var(--accent-green);
                border-color: rgba(52, 211, 153, 0.25);
                background: rgba(52, 211, 153, 0.08);
            }
            .pulse-dot {
                width: 6px;
                height: 6px;
                background: var(--accent-green);
                border-radius: 50%;
                box-shadow: 0 0 6px var(--accent-green);
            }

            main {
                flex: 1;
                overflow-y: auto;
                padding: 85px 23% 160px 23%;
                display: flex;
                flex-direction: column;
                gap: 28px;
            }
            @media (max-width: 1100px) {
                main { padding: 75px 6% 145px 6%; }
            }

            .hero {
                text-align: center;
                padding: 30px 0 10px 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 12px;
            }
            .hero-greeting {
                font-size: 2.8rem;
                font-weight: 600;
                letter-spacing: -0.03em;
                background: linear-gradient(135deg, #70a5ff 0%, #c38fff 50%, #ff8f70 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .hero-sub {
                font-size: 1.15rem;
                color: var(--text-sub);
                max-width: 520px;
                line-height: 1.5;
            }
            .chips {
                display: flex;
                flex-wrap: wrap;
                gap: 9px;
                justify-content: center;
                margin-top: 14px;
            }
            .chip-btn {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border);
                color: var(--text);
                padding: 8px 16px;
                border-radius: 9999px;
                font-size: 0.84rem;
                cursor: pointer;
                transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            }
            .chip-btn:hover {
                background: rgba(255, 255, 255, 0.12);
                border-color: rgba(112, 165, 255, 0.4);
                transform: translateY(-1px);
            }

            .turn {
                display: flex;
                flex-direction: column;
                gap: 12px;
                animation: turnFade 0.25s ease-out;
            }
            @keyframes turnFade {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .user-bubble {
                align-self: flex-end;
                background: rgba(40, 42, 54, 0.8);
                border: 1px solid var(--border);
                padding: 12px 20px;
                border-radius: 20px 20px 4px 20px;
                font-size: 1rem;
                max-width: 82%;
                line-height: 1.5;
            }

            .ai-bubble {
                background: var(--surface);
                backdrop-filter: blur(28px);
                -webkit-backdrop-filter: blur(28px);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 24px 28px;
                font-size: 1.02rem;
                line-height: 1.72;
                color: var(--text);
            }
            .ai-bubble p { margin-bottom: 12px; }
            .ai-bubble p:last-child { margin-bottom: 0; }
            .ai-bubble h1, .ai-bubble h2, .ai-bubble h3 {
                margin: 18px 0 8px 0;
                font-weight: 600;
                color: #fff;
            }
            .ai-bubble pre {
                background: #06070a;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 18px;
                overflow-x: auto;
                font-family: var(--mono);
                font-size: 0.88rem;
                margin: 14px 0;
            }
            .ai-bubble code {
                font-family: var(--mono);
                background: rgba(112, 165, 255, 0.12);
                color: var(--accent-blue);
                padding: 2px 6px;
                border-radius: 5px;
                font-size: 0.88em;
            }
            .ai-bubble pre code { background: transparent; padding: 0; color: #fff; }

            .ai-bubble table {
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                margin: 16px 0;
                font-size: 0.9rem;
                border: 1px solid var(--border);
                border-radius: 10px;
                overflow: hidden;
            }
            .ai-bubble th, .ai-bubble td {
                padding: 10px 14px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }
            .ai-bubble th {
                background: rgba(255, 255, 255, 0.04);
                font-weight: 600;
                color: #fff;
            }
            .ai-bubble tr:last-child td { border-bottom: none; }

            .source-tag {
                display: inline-block;
                margin-top: 14px;
                font-family: var(--mono);
                font-size: 0.72rem;
                color: var(--text-sub);
                background: rgba(255, 255, 255, 0.04);
                padding: 3px 10px;
                border-radius: 9999px;
                border: 1px solid var(--border);
            }

            .gemini-loader {
                display: flex;
                flex-direction: column;
                gap: 12px;
                padding: 22px 26px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                backdrop-filter: blur(28px);
                max-width: 580px;
            }
            .gemini-aurora-bar {
                height: 14px;
                border-radius: 8px;
                background: linear-gradient(
                    90deg, 
                    rgba(112, 165, 255, 0.15) 0%, 
                    rgba(195, 143, 255, 0.35) 25%, 
                    rgba(255, 143, 112, 0.35) 50%, 
                    rgba(112, 230, 255, 0.35) 75%, 
                    rgba(112, 165, 255, 0.15) 100%
                );
                background-size: 300% 100%;
                animation: auroraShimmer 2.2s infinite ease-in-out;
            }
            .gemini-aurora-bar:nth-child(1) { width: 85%; }
            .gemini-aurora-bar:nth-child(2) { width: 100%; }
            .gemini-aurora-bar:nth-child(3) { width: 60%; }

            @keyframes auroraShimmer {
                0% { background-position: 100% 0; }
                100% { background-position: -100% 0; }
            }

            .typing-cursor {
                display: inline-block;
                width: 2px;
                height: 1em;
                background: var(--accent-blue);
                margin-left: 2px;
                vertical-align: text-bottom;
                animation: blink 0.8s infinite;
            }
            @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

            .input-dock {
                position: fixed;
                bottom: 0; left: 0; right: 0;
                padding: 0 23% 28px 23%;
                background: linear-gradient(180deg, transparent 0%, rgba(14,14,17,0.85) 50%, #0e0e11 100%);
                z-index: 50;
            }
            @media (max-width: 1100px) {
                .input-dock { padding: 0 6% 20px 6%; }
            }

            .pill-bar {
                background: rgba(28, 28, 32, 0.88);
                backdrop-filter: blur(28px);
                border: 1px solid var(--border);
                border-radius: 9999px;
                padding: 6px 8px 6px 14px;
                display: flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
                transition: border-color 0.2s;
            }
            .pill-bar:focus-within {
                border-color: rgba(112, 165, 255, 0.5);
                box-shadow: 0 0 0 3px rgba(112, 165, 255, 0.2), 0 16px 36px rgba(0, 0, 0, 0.5);
            }

            .btn-paperclip {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid var(--border);
                color: var(--text);
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s;
                flex-shrink: 0;
            }
            .btn-paperclip:hover { background: rgba(255, 255, 255, 0.14); }
            .btn-paperclip svg { width: 14px; height: 14px; stroke: currentColor; fill: none; }

            input[type="text"] {
                flex: 1;
                background: transparent;
                border: none;
                outline: none;
                color: #fff;
                font-family: inherit;
                font-size: 0.98rem;
                padding: 6px 4px;
            }
            input[type="text"]::placeholder { color: var(--text-sub); }

            .btn-send {
                width: 34px;
                height: 34px;
                border-radius: 50%;
                background: #fff;
                color: #000;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s;
                flex-shrink: 0;
            }
            .btn-send:hover { transform: scale(1.06); }
            .btn-send svg { width: 15px; height: 15px; fill: #000; }
        </style>
    </head>
    <body>
        <header>
            <div class="brand-group">
                <div class="brand">OSTUTOR</div>
                <div class="top-badge online"><span class="pulse-dot"></span> Ollama Active</div>
            </div>

            <div class="telemetry-row">
                <div class="top-badge">Model: <strong>qwen2.5:1.5b</strong></div>
                <div class="top-badge">Engine: <strong>Chroma-Lite RAG</strong></div>
                <div class="top-badge" id="ragStats">Index: <strong id="chunkCount">--</strong> chunks</div>
                <div class="top-badge">Math: <strong>Deterministic</strong></div>
            </div>
        </header>

        <main id="chatBox">
            <div class="hero">
                <h1 class="hero-greeting">Hello, Student</h1>
                <p class="hero-sub">How can I help you master Operating Systems today?</p>
                
                <div class="chips">
                    <button class="chip-btn" onclick="quickSend('explain fork() to me like I am 10')">💡 Explain fork() simply</button>
                    <button class="chip-btn" onclick="quickSend('calculate turnaround time and waiting time if CT=18, AT=4, BT=6')">📊 Solve TAT & WT</button>
                    <button class="chip-btn" onclick="quickSend('what are the 4 conditions for a deadlock?')">🔒 Deadlock Conditions</button>
                    <button class="chip-btn" onclick="quickSend('how does virtual memory paging work?')">🧠 Virtual Memory</button>
                </div>
            </div>
        </main>

        <div class="input-dock">
            <div class="pill-bar">
                <input type="file" id="fileInput" style="display:none;" onchange="uploadFile()" />
                <button class="btn-paperclip" onclick="document.getElementById('fileInput').click()" title="Attach PDF/PPTX/TXT">
                    <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                    </svg>
                </button>

                <input type="text" id="userInput" placeholder="Ask anything about operating systems..." onkeypress="handleKey(event)" autocomplete="off" />

                <button class="btn-send" onclick="sendMessage()" title="Send">
                    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
        </div>

        <script>
            let history = [];

            async function updateTelemetry() {
                try {
                    const res = await fetch("/api/status");
                    const data = await res.json();
                    document.getElementById("chunkCount").innerText = `${data.rag_chunks} (${data.sources} docs)`;
                } catch(e) {
                    document.getElementById("chunkCount").innerText = "0 chunks";
                }
            }
            updateTelemetry();

            function renderKaTeX(elem) {
                if (window.renderMathInElement) {
                    renderMathInElement(elem, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false}
                        ],
                        throwOnError: false
                    });
                }
            }

            function quickSend(text) {
                document.getElementById("userInput").value = text;
                sendMessage();
            }

            async function uploadFile() {
                const input = document.getElementById("fileInput");
                if (!input.files.length) return;
                const file = input.files[0];
                const fd = new FormData();
                fd.append("file", file);

                const chat = document.getElementById("chatBox");
                const notice = document.createElement("div");
                notice.className = "turn";
                notice.innerHTML = `<div class="ai-bubble" style="padding:14px 20px;">Analyzing and vectorizing <code>${file.name}</code>...</div>`;
                chat.appendChild(notice);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch("/api/upload", { method: "POST", body: fd });
                    const data = await res.json();
                    if (res.ok) {
                        notice.innerHTML = `<div class="ai-bubble" style="padding:14px 20px; border-color: rgba(112, 165, 255, 0.4);">
                            <strong>File Indexed:</strong> <code>${data.filename}</code> is now in active memory (${data.chunks_added} chunks added).
                        </div>`;
                        updateTelemetry();
                    } else {
                        notice.innerHTML = `<div class="ai-bubble" style="color:#ff453a;">${data.message || 'Upload failed'}</div>`;
                    }
                } catch(e) {
                    notice.innerHTML = `<div class="ai-bubble" style="color:#ff453a;">Upload failed.</div>`;
                }
                chat.scrollTop = chat.scrollHeight;
            }

            function revealTokensProgressively(fullText, container, onComplete) {
                const tokens = fullText.split(/(?<=\\s+)/);
                let currentText = "";
                let index = 0;

                const cursor = document.createElement("span");
                cursor.className = "typing-cursor";

                function step() {
                   if (index < tokens.length) {
                        // Reveal 1 word at a time
                        const chunk = tokens.slice(index, index + 1).join("");
                        index += 1;
                        currentText += chunk;
                        
                        container.innerHTML = marked.parse(currentText);
                        container.appendChild(cursor);

                        const chat = document.getElementById("chatBox");
                        chat.scrollTop = chat.scrollHeight;

                        // Increased delay (40ms) for slower pacing
                        setTimeout(step, 40);
                    } else {
                        cursor.remove();
                        container.innerHTML = marked.parse(fullText);
                        renderKaTeX(container);
                        if (onComplete) onComplete();
                    }
                }
                step();
            }

            async function sendMessage() {
                const input = document.getElementById("userInput");
                const text = input.value.trim();
                if (!text) return;

                const chat = document.getElementById("chatBox");

                const turn = document.createElement("div");
                turn.className = "turn";

                const userMsg = document.createElement("div");
                userMsg.className = "user-bubble";
                userMsg.innerText = text;
                turn.appendChild(userMsg);

                input.value = "";
                chat.appendChild(turn);
                chat.scrollTop = chat.scrollHeight;

                history.push({role: "Student", content: text});

                const loader = document.createElement("div");
                loader.className = "gemini-loader";
                loader.innerHTML = `
                    <div class="gemini-aurora-bar"></div>
                    <div class="gemini-aurora-bar"></div>
                    <div class="gemini-aurora-bar"></div>
                `;
                turn.appendChild(loader);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch("/api/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({query: text, history: history})
                    });
                    const data = await res.json();
                    loader.remove();

                    const aiMsg = document.createElement("div");
                    aiMsg.className = "ai-bubble";
                    turn.appendChild(aiMsg);

                    revealTokensProgressively(data.answer, aiMsg, () => {
                        if (data.retrieved_context && data.retrieved_context.length > 0) {
                            const src = data.retrieved_context.map(c => c.source).join(", ");
                            const srcTag = document.createElement("div");
                            srcTag.className = "source-tag";
                            srcTag.innerText = `📄 Context: ${src}`;
                            aiMsg.appendChild(srcTag);
                        }
                    });

                    history.push({role: "Tutor", content: data.answer});
                } catch(e) {
                    loader.remove();
                    const err = document.createElement("div");
                    err.className = "ai-bubble";
                    err.style.color = "#ff453a";
                    err.innerText = "Connection lost. Please ensure both Ollama and main.py are active.";
                    turn.appendChild(err);
                }
                chat.scrollTop = chat.scrollHeight;
            }

            function handleKey(e) {
                if (e.key === "Enter") sendMessage();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)