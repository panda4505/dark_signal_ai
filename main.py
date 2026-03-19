from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from openpyxl import load_workbook
import fitz
import os, tempfile

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Temp directory for uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai_app_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Upload endpoint: just saves the file ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "path": path}


# --- Ask endpoint ---

class PromptRequest(BaseModel):
    prompt: str
    file_path: str | None = None
    file_name: str | None = None


TEXT_FILE_EXTENSIONS = {"csv", "txt", "json", "md"}
SPREADSHEET_EXTENSIONS = {"xlsx", "xls"}
MAX_EXTRACTED_CHARS = 50000


def limit_text(text: str) -> str:
    return text[:MAX_EXTRACTED_CHARS]


def extract_pdf_text(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    return limit_text(text)


def extract_plain_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return limit_text(f.read())


def extract_spreadsheet_text(file_path: str) -> str:
    workbook = load_workbook(file_path, data_only=True)
    sections = []

    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            values = ["" if value is None else str(value) for value in row]
            rows.append(",".join(values).rstrip(","))
        sheet_text = "\n".join(rows).strip()
        sections.append(f"[Sheet: {sheet.title}]\n{sheet_text}".strip())

    workbook.close()
    return limit_text("\n\n".join(section for section in sections if section))


def extract_file_text(file_path: str, file_name: str | None) -> str:
    name = file_name or file_path
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf":
        return extract_pdf_text(file_path)
    if ext in TEXT_FILE_EXTENSIONS:
        return extract_plain_text(file_path)
    if ext in SPREADSHEET_EXTENSIONS:
        return extract_spreadsheet_text(file_path)
    return "[Unsupported file type]"


@app.post("/ask")
def ask_ai(data: PromptRequest):
    try:
        extracted_text = ""
        if data.file_path:
            extracted_text = extract_file_text(data.file_path, data.file_name)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question using the provided file text "
                        "when available. If the file text says '[Unsupported file type]', explain that the file "
                        "type is not supported."
                    ),
                },
                {"role": "user", "content": (
                    f"User question:\n{data.prompt}\n\n"
                    f"Filename: {data.file_name or '[No file uploaded]'}\n\n"
                    f"Extracted file text:\n---\n{extracted_text or '[No file uploaded]'}\n---"
                )}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


# --- Frontend ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>My First AI App</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {
                --bg: #05030a;
                --bg-soft: rgba(20, 14, 34, 0.82);
                --panel: rgba(16, 12, 29, 0.86);
                --panel-strong: rgba(24, 17, 43, 0.94);
                --border: rgba(178, 120, 255, 0.24);
                --border-strong: rgba(206, 148, 255, 0.42);
                --text: #f4ebff;
                --muted: #b9accd;
                --accent: #b145ff;
                --accent-strong: #ff4fd8;
                --accent-soft: rgba(177, 69, 255, 0.16);
                --success: #7ef7c7;
                --shadow: 0 24px 70px rgba(0, 0, 0, 0.55);
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                min-height: 100vh;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at top left, rgba(177, 69, 255, 0.28), transparent 30%),
                    radial-gradient(circle at top right, rgba(255, 79, 216, 0.16), transparent 24%),
                    linear-gradient(145deg, #05030a 0%, #0b0815 38%, #130d24 100%);
                overflow-x: hidden;
            }

            body::before {
                content: "";
                position: fixed;
                inset: 0;
                background-image:
                    linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
                background-size: 42px 42px;
                mask-image: radial-gradient(circle at center, black 35%, transparent 90%);
                pointer-events: none;
                opacity: 0.35;
            }

            .page-shell {
                width: min(980px, calc(100% - 32px));
                margin: 0 auto;
                padding: 48px 0 56px;
                position: relative;
            }

            .panel {
                position: relative;
                background: linear-gradient(180deg, rgba(28, 20, 48, 0.92), rgba(11, 8, 21, 0.92));
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(16px);
            }

            .hero {
                padding: 32px;
                overflow: hidden;
                animation: fade-up 0.7s ease-out both;
            }

            .hero::after {
                content: "";
                position: absolute;
                inset: auto -60px -90px auto;
                width: 260px;
                height: 260px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(177, 69, 255, 0.34), transparent 68%);
                filter: blur(12px);
            }

            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 8px 14px;
                border-radius: 999px;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 11px;
                color: #f7dcff;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            .eyebrow::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--accent-strong), var(--accent));
                box-shadow: 0 0 16px rgba(255, 79, 216, 0.7);
            }

            h1 {
                margin-top: 18px;
                font-size: clamp(2.2rem, 5vw, 4.4rem);
                line-height: 0.94;
                letter-spacing: -0.05em;
                max-width: 10ch;
            }

            .hero-copy {
                max-width: 620px;
                margin-top: 18px;
                font-size: 16px;
                line-height: 1.7;
                color: var(--muted);
            }

            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 26px;
            }

            .meta-pill, .signal {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                border-radius: 999px;
                padding: 10px 14px;
                font-size: 13px;
                color: #f4eaff;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            .meta-pill::before, .signal::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--success);
                box-shadow: 0 0 12px rgba(126, 247, 199, 0.65);
            }

            .meta-pill.idle::before, .signal.idle::before {
                background: rgba(255, 255, 255, 0.35);
                box-shadow: none;
            }

            .meta-pill.busy::before, .signal.busy::before {
                background: #ffd166;
                box-shadow: 0 0 12px rgba(255, 209, 102, 0.65);
            }

            .composer,
            .output-panel {
                margin-top: 22px;
                padding: 26px;
                animation: fade-up 0.85s ease-out both;
            }

            .output-panel {
                animation-delay: 0.08s;
            }

            .section-label {
                display: block;
                margin-bottom: 12px;
                color: #ded0f4;
                font-size: 13px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            textarea {
                width: 100%;
                min-height: 150px;
                padding: 18px 20px;
                font: inherit;
                font-size: 16px;
                line-height: 1.6;
                color: var(--text);
                resize: vertical;
                border-radius: 20px;
                border: 1px solid rgba(195, 150, 255, 0.22);
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)),
                    var(--bg-soft);
                outline: none;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
            }

            textarea:focus {
                border-color: var(--border-strong);
                box-shadow: 0 0 0 4px rgba(177, 69, 255, 0.14);
                transform: translateY(-1px);
            }

            textarea::placeholder {
                color: #8d7da8;
            }

            .drop-zone {
                margin-top: 16px;
                border-radius: 22px;
                border: 1px dashed rgba(216, 175, 255, 0.36);
                background:
                    linear-gradient(135deg, rgba(177, 69, 255, 0.12), rgba(255, 79, 216, 0.06)),
                    rgba(255, 255, 255, 0.02);
                padding: 24px;
                cursor: pointer;
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            }

            .drop-zone:hover,
            .drop-zone.drag-over {
                transform: translateY(-2px);
                border-color: rgba(239, 198, 255, 0.75);
                box-shadow: 0 16px 30px rgba(116, 32, 179, 0.25);
                background:
                    linear-gradient(135deg, rgba(177, 69, 255, 0.18), rgba(255, 79, 216, 0.1)),
                    rgba(255, 255, 255, 0.03);
            }

            .drop-zone.busy {
                border-style: solid;
            }

            .drop-top {
                display: flex;
                justify-content: space-between;
                gap: 16px;
                align-items: center;
                margin-bottom: 12px;
            }

            .drop-title {
                font-size: 18px;
                font-weight: 700;
                letter-spacing: -0.02em;
            }

            .drop-copy {
                color: var(--muted);
                font-size: 14px;
                line-height: 1.6;
            }

            .drop-badge {
                flex-shrink: 0;
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.08);
                font-size: 12px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #f0dcff;
            }

            .filename {
                display: inline-block;
                margin-top: 8px;
                font-weight: 700;
                color: #ffffff;
                text-shadow: 0 0 18px rgba(177, 69, 255, 0.5);
                word-break: break-word;
            }

            .actions {
                display: flex;
                justify-content: flex-end;
                margin-top: 18px;
            }

            button {
                padding: 14px 24px;
                min-width: 180px;
                border: 0;
                border-radius: 16px;
                font: inherit;
                font-weight: 700;
                letter-spacing: 0.02em;
                color: #fff;
                cursor: pointer;
                background: linear-gradient(135deg, var(--accent), var(--accent-strong));
                box-shadow:
                    0 12px 28px rgba(177, 69, 255, 0.35),
                    inset 0 1px 0 rgba(255, 255, 255, 0.18);
                transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
            }

            button:hover:not(:disabled) {
                transform: translateY(-1px);
                box-shadow:
                    0 16px 32px rgba(177, 69, 255, 0.42),
                    inset 0 1px 0 rgba(255, 255, 255, 0.18);
            }

            button:disabled {
                opacity: 0.58;
                cursor: not-allowed;
                transform: none;
            }

            .output-header {
                display: flex;
                justify-content: space-between;
                gap: 16px;
                align-items: center;
                margin-bottom: 18px;
            }

            .output-title {
                font-size: 24px;
                letter-spacing: -0.03em;
            }

            pre {
                min-height: 220px;
                white-space: pre-wrap;
                word-wrap: break-word;
                padding: 20px;
                border-radius: 18px;
                border: 1px solid rgba(201, 159, 255, 0.14);
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
                    var(--panel-strong);
                color: #efe6ff;
                font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
                line-height: 1.7;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }

            @keyframes fade-up {
                from {
                    opacity: 0;
                    transform: translateY(18px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 700px) {
                .page-shell {
                    width: min(100% - 20px, 980px);
                    padding: 20px 0 28px;
                }

                .hero,
                .composer,
                .output-panel {
                    padding: 20px;
                    border-radius: 20px;
                }

                .drop-top,
                .output-header {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .actions {
                    justify-content: stretch;
                }

                button {
                    width: 100%;
                }

                h1 {
                    max-width: none;
                }
            }
        </style>
    </head>
    <body>
        <div class="page-shell">
            <section class="panel hero">
                <div class="eyebrow">Secure Analysis Console</div>
                <h1>Dark Signal AI</h1>
                <p class="hero-copy">
                    Drop in a file, write a prompt, and check out the answer inside a cleaner cyber-inspired workspace.
                </p>
                <div class="hero-meta">
                    <span class="meta-pill">FastAPI + AI Workflow</span>
                    <span id="file-status" class="meta-pill idle">No file attached</span>
                </div>
            </section>

            <section class="panel composer">
                <label class="section-label" for="prompt">Prompt Input</label>
                <textarea id="prompt" rows="5" placeholder="Ask a question about a file, request a summary, or describe the analysis you want."></textarea>

                <div id="drop-zone" class="drop-zone" onclick="document.getElementById('file-input').click()"></div>
                <input type="file" id="file-input" hidden>

                <div class="actions">
                    <button id="send-btn" onclick="sendPrompt()">Run Analysis</button>
                </div>
            </section>

            <section class="panel output-panel">
                <div class="output-header">
                    <div>
                        <div class="section-label">Response Feed</div>
                        <div class="output-title">Model Output</div>
                    </div>
                    <div id="response-state" class="signal idle">Standing by</div>
                </div>
                <pre id="output">Your answer will appear here.</pre>
            </section>
        </div>

        <script>
            let uploadedFile = null;

            const dropZone = document.getElementById("drop-zone");
            const fileStatus = document.getElementById("file-status");
            const responseState = document.getElementById("response-state");
            const output = document.getElementById("output");

            function escapeHtml(text) {
                return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
            }

            function renderDropZone() {
                if (uploadedFile) {
                    dropZone.innerHTML = `
                        <div class="drop-top">
                            <div class="drop-title">File locked and ready</div>
                            <div class="drop-badge">Attached</div>
                        </div>
                        <div class="drop-copy">
                            Click to replace the current file or drag a new one into the zone.
                            <div class="filename">${escapeHtml(uploadedFile.name)}</div>
                        </div>
                    `;
                    return;
                }

                dropZone.innerHTML = `
                    <div class="drop-top">
                        <div class="drop-title">Drop a file into the grid</div>
                        <div class="drop-badge">Any format</div>
                    </div>
                    <div class="drop-copy">
                        Drag and drop or click to attach a document, spreadsheet, PDF, or code file for analysis.
                    </div>
                `;
            }

            function setFileStatus(text, stateClass) {
                fileStatus.textContent = text;
                fileStatus.className = "meta-pill " + stateClass;
            }

            function setResponseState(text, stateClass) {
                responseState.textContent = text;
                responseState.className = "signal " + stateClass;
            }

            renderDropZone();

            dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
            dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
            dropZone.addEventListener("drop", e => {
                e.preventDefault(); dropZone.classList.remove("drag-over");
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            });

            document.getElementById("file-input").addEventListener("change", e => {
                if (e.target.files.length) handleFile(e.target.files[0]);
            });

            async function handleFile(file) {
                dropZone.classList.add("busy");
                dropZone.innerHTML = `
                    <div class="drop-top">
                        <div class="drop-title">Uploading file</div>
                        <div class="drop-badge">Syncing</div>
                    </div>
                    <div class="drop-copy">Preparing ${escapeHtml(file.name)} for analysis...</div>
                `;
                setFileStatus("Uploading file", "busy");
                const form = new FormData();
                form.append("file", file);
                const res = await fetch("/upload", { method: "POST", body: form });
                const data = await res.json();

                if (data.error) {
                    uploadedFile = null;
                    renderDropZone();
                    output.textContent = "Upload failed: " + data.error;
                    setFileStatus("Upload failed", "idle");
                } else {
                    uploadedFile = { path: data.path, name: data.filename };
                    renderDropZone();
                    setFileStatus("Attached: " + data.filename, "ready");
                }
                dropZone.classList.remove("busy");
            }

            async function sendPrompt() {
                const btn = document.getElementById("send-btn");
                const prompt = document.getElementById("prompt").value;
                if (!prompt.trim()) return;

                btn.disabled = true;
                btn.textContent = "Processing...";
                output.textContent = "Running analysis...";
                setResponseState("Analyzing", "busy");

                const body = { prompt };
                if (uploadedFile) {
                    body.file_path = uploadedFile.path;
                    body.file_name = uploadedFile.name;
                }

                const res = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                output.textContent = data.answer || data.error;
                setResponseState("Response ready", "ready");

                btn.disabled = false;
                btn.textContent = "Run Analysis";
            }
        </script>
    </body>
    </html>
    """
