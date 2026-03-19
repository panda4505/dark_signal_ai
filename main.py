from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from openpyxl import load_workbook
import base64
import fitz
import mimetypes
import os, tempfile
import zipfile
import xml.etree.ElementTree as ET

load_dotenv()

app = FastAPI()

TEXT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", TEXT_MODEL)

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


TEXT_FILE_EXTENSIONS = {
    "c", "cfg", "conf", "cpp", "css", "csv", "eml", "go", "html", "htm", "ini",
    "java", "js", "json", "log", "md", "py", "rb", "rs", "rtf", "sh", "sql",
    "tex", "toml", "ts", "tsx", "txt", "xml", "yaml", "yml",
}
SPREADSHEET_EXTENSIONS = {"xlsx", "xlsm", "xltx", "xltm"}
WORD_EXTENSIONS = {"docx", "docm"}
PRESENTATION_EXTENSIONS = {"pptx", "pptm"}
OPENDOCUMENT_EXTENSIONS = {"odt", "ods", "odp"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
LEGACY_OFFICE_EXTENSIONS = {"doc", "xls", "ppt"}
MAX_EXTRACTED_CHARS = 50000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def limit_text(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_EXTRACTED_CHARS:
        return text

    suffix = "\n\n[Truncated]"
    cutoff = max(0, MAX_EXTRACTED_CHARS - len(suffix))
    return text[:cutoff].rstrip() + suffix


def get_extension(file_path: str, file_name: str | None) -> str:
    name = file_name or file_path
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def with_fallback_text(text: str, fallback: str) -> str:
    return limit_text(text) or fallback


def natural_sort_key(text: str):
    parts = []
    chunk = ""
    chunk_is_digit = None

    for char in text:
        is_digit = char.isdigit()
        if chunk_is_digit is None or is_digit == chunk_is_digit:
            chunk += char
        else:
            parts.append(int(chunk) if chunk_is_digit else chunk.lower())
            chunk = char
        chunk_is_digit = is_digit

    if chunk:
        parts.append(int(chunk) if chunk_is_digit else chunk.lower())

    return parts


def collect_xml_text(element: ET.Element) -> str:
    parts = []

    for node in element.iter():
        if node.tag.endswith(("}tab", "}br", "}cr", "}line-break")):
            parts.append(" ")
        if node.text:
            parts.append(node.text)

    return " ".join("".join(parts).split())


def extract_pdf_text(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    return with_fallback_text(text, "[No extractable text found in PDF]")


def extract_plain_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return with_fallback_text(f.read(), "[No extractable text found in file]")


def extract_spreadsheet_text(file_path: str) -> str:
    workbook = load_workbook(file_path, data_only=True)
    try:
        sections = []

        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(value for value in values):
                    rows.append(",".join(values).rstrip(","))
            sheet_text = "\n".join(rows).strip()
            if sheet_text:
                sections.append(f"[Sheet: {sheet.title}]\n{sheet_text}")

        return with_fallback_text(
            "\n\n".join(sections),
            "[No extractable text found in spreadsheet]",
        )
    finally:
        workbook.close()


def extract_docx_text(file_path: str) -> str:
    doc_members = []

    with zipfile.ZipFile(file_path) as archive:
        for name in archive.namelist():
            if name == "word/document.xml":
                doc_members.append(name)
            elif name.startswith("word/header") and name.endswith(".xml"):
                doc_members.append(name)
            elif name.startswith("word/footer") and name.endswith(".xml"):
                doc_members.append(name)
            elif name in {"word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"}:
                doc_members.append(name)

        sections = []
        for name in sorted(doc_members, key=natural_sort_key):
            root = ET.fromstring(archive.read(name))
            paragraphs = []
            for element in root.iter():
                if element.tag.endswith("}p"):
                    text = collect_xml_text(element)
                    if text:
                        paragraphs.append(text)

            if paragraphs:
                label = "Document" if name == "word/document.xml" else name.rsplit("/", 1)[-1].replace(".xml", "")
                sections.append(f"[{label}]\n" + "\n".join(paragraphs))

    return with_fallback_text("\n\n".join(sections), "[No extractable text found in document]")


def extract_presentation_text(file_path: str) -> str:
    with zipfile.ZipFile(file_path) as archive:
        slide_members = sorted(
            [
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ],
            key=natural_sort_key,
        )

        sections = []
        for index, name in enumerate(slide_members, start=1):
            root = ET.fromstring(archive.read(name))
            texts = []
            for element in root.iter():
                if element.tag.endswith("}t") and element.text:
                    chunk = " ".join(element.text.split())
                    if chunk:
                        texts.append(chunk)

            if texts:
                sections.append(f"[Slide {index}]\n" + "\n".join(texts))

    return with_fallback_text("\n\n".join(sections), "[No extractable text found in presentation]")


def extract_opendocument_text(file_path: str) -> str:
    with zipfile.ZipFile(file_path) as archive:
        if "content.xml" not in archive.namelist():
            return "[No extractable text found in document]"

        root = ET.fromstring(archive.read("content.xml"))
        blocks = []

        for element in root.iter():
            if element.tag.endswith(("}p", "}h", "}list-item", "}table-cell")):
                text = collect_xml_text(element)
                if text:
                    blocks.append(text)

    return with_fallback_text("\n".join(blocks), "[No extractable text found in document]")


def build_image_data_url(file_path: str, file_name: str | None) -> str:
    with open(file_path, "rb") as f:
        data = f.read()

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image files larger than 8 MB are not supported yet.")

    mime_type = mimetypes.guess_type(file_name or file_path)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_file_context(file_path: str, file_name: str | None) -> dict:
    ext = get_extension(file_path, file_name)

    if ext == "pdf":
        return {"kind": "text", "content": extract_pdf_text(file_path)}
    if ext in TEXT_FILE_EXTENSIONS:
        return {"kind": "text", "content": extract_plain_text(file_path)}
    if ext in SPREADSHEET_EXTENSIONS:
        return {"kind": "text", "content": extract_spreadsheet_text(file_path)}
    if ext in WORD_EXTENSIONS:
        return {"kind": "text", "content": extract_docx_text(file_path)}
    if ext in PRESENTATION_EXTENSIONS:
        return {"kind": "text", "content": extract_presentation_text(file_path)}
    if ext in OPENDOCUMENT_EXTENSIONS:
        return {"kind": "text", "content": extract_opendocument_text(file_path)}
    if ext in IMAGE_EXTENSIONS:
        return {"kind": "image", "content": build_image_data_url(file_path, file_name)}
    if ext in LEGACY_OFFICE_EXTENSIONS:
        return {
            "kind": "text",
            "content": (
                f"[Unsupported legacy Office file type: .{ext}. Please re-save the file as a modern format "
                "such as .docx, .xlsx, or .pptx.]"
            ),
        }
    return {"kind": "text", "content": "[Unsupported file type]"}


def build_request_payload(data: PromptRequest):
    if not data.file_path:
        return "none", TEXT_MODEL, [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data.prompt},
        ]

    file_context = build_file_context(data.file_path, data.file_name)

    if file_context["kind"] == "image":
        return "image", VISION_MODEL, [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Analyze the uploaded image and answer the user's question. "
                    "Be honest if the image is blurry or does not contain enough information."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Filename: {data.file_name or '[Uploaded image]'}\n"
                            f"User question: {data.prompt}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": file_context["content"]},
                    },
                ],
            },
        ]

    return "text", TEXT_MODEL, [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using the provided file content when it "
                "is available. If the extracted content says the file is unsupported or no text could be found, "
                "explain that clearly and suggest a better format when appropriate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{data.prompt}\n\n"
                f"Filename: {data.file_name or '[No file uploaded]'}\n\n"
                f"Extracted file content:\n---\n{file_context['content']}\n---"
            ),
        },
    ]


@app.post("/ask")
def ask_ai(data: PromptRequest):
    file_kind = "none"
    try:
        file_kind, model_name, messages = build_request_payload(data)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=16384,
        )
        reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""
        content = response.choices[0].message.content or ""
        if reasoning:
            return {"answer": content, "reasoning": reasoning}
        return {"answer": content}
    except Exception as e:
        if file_kind == "image":
            return {
                "error": (
                    "Image analysis failed with the configured DeepSeek model. "
                    "If your current model is text-only, set DEEPSEEK_VISION_MODEL to a vision-capable model. "
                    f"Details: {e}"
                )
            }
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
            @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap");

            @property --angle {
                syntax: "<angle>";
                inherits: false;
                initial-value: 0deg;
            }

            :root {
                --page-bg: #0d0a14;
                --panel-bg: #1a1625;
                --panel-border: rgba(255, 255, 255, 0.08);
                --panel-hover-bg: #201b2d;
                --panel-hover-border: rgba(0, 243, 255, 0.20);
                --violet: rgb(139, 92, 246);
                --violet-border: rgba(139, 92, 246, 0.22);
                --violet-soft: rgba(139, 92, 246, 0.08);
                --violet-focus: rgba(139, 92, 246, 0.40);
                --fuchsia: rgb(232, 121, 249);
                --cyan: #00f3ff;
                --emerald: rgb(16, 185, 129);
                --text-primary: #ffffff;
                --text-secondary: rgba(196, 187, 220, 0.82);
                --text-muted: rgba(161, 151, 188, 0.65);
                --placeholder: rgb(100, 116, 139);
                --field-bg: rgba(8, 8, 15, 0.38);
                --field-border: rgba(255, 255, 255, 0.08);
                --shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                min-height: 100vh;
                font-family: "Inter", system-ui, sans-serif;
                color: var(--text-primary);
                background: var(--page-bg);
                overflow-x: hidden;
            }

            body::before {
                content: "";
                position: fixed;
                inset: 0;
                background: radial-gradient(circle at top center, rgba(139, 92, 246, 0.14), transparent 58%);
                pointer-events: none;
                z-index: 0;
            }

            .page-shell {
                width: min(1280px, calc(100% - 64px));
                margin: 0 auto;
                padding: 32px 0;
                position: relative;
                z-index: 1;
                display: grid;
                gap: 24px;
            }

            .background-orbs {
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 0;
            }

            .orb {
                position: absolute;
                border-radius: 999px;
                filter: blur(48px);
                opacity: 0.12;
            }

            .orb-violet {
                width: 420px;
                height: 420px;
                top: -120px;
                left: -100px;
                background: rgba(139, 92, 246, 0.95);
            }

            .orb-fuchsia {
                width: 360px;
                height: 360px;
                top: 140px;
                right: 80px;
                background: rgba(232, 121, 249, 0.95);
            }

            .orb-mix {
                width: 460px;
                height: 460px;
                right: 18%;
                bottom: -140px;
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.95), rgba(232, 121, 249, 0.9));
            }

            .panel {
                position: relative;
                background: var(--panel-bg);
                border: 1px solid var(--panel-border);
                border-radius: 12px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                transition: background 0.25s ease, border-color 0.25s ease, transform 0.25s ease;
                animation: card-enter 0.45s ease both;
                overflow: hidden;
                isolation: isolate;
            }

            .panel:hover {
                background: var(--panel-hover-bg);
                border-color: var(--panel-hover-border);
            }

            .panel::before {
                content: "";
                position: absolute;
                inset: -1px;
                border-radius: inherit;
                padding: 1px;
                background: rgba(139, 92, 246, 0.22);
                -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                -webkit-mask-composite: xor;
                mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                mask-composite: exclude;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s ease;
            }

            .hero { animation-delay: 0s; }
            .composer { animation-delay: 0.08s; }
            .output-panel { animation-delay: 0.16s; }

            .composer:focus-within::before,
            .output-panel:has(#response-state.busy)::before,
            .output-panel:has(#response-state.ready)::before {
                opacity: 1;
                background: conic-gradient(from var(--angle), rgba(0, 243, 255, 0.92), rgba(232, 121, 249, 0.82), rgba(0, 243, 255, 0.92));
                animation: rotate-border 6s linear infinite;
            }

            .top-bar {
                padding: 12px 24px;
                max-height: 56px;
            }

            .topbar-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                min-height: 32px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
                min-width: 0;
            }

            .brand-icon {
                width: 24px;
                height: 24px;
                color: var(--fuchsia);
                flex-shrink: 0;
            }

            .app-title {
                font-family: "Space Mono", monospace;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: var(--text-primary);
                white-space: nowrap;
            }

            .topbar-pills {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 8px;
                min-width: 0;
                flex: 1;
            }

            .composer,
            .output-panel {
                padding: 24px;
            }

            .composer {
            }

            .content-grid {
                display: grid;
                gap: 24px;
                align-items: start;
            }

            @media (min-width: 900px) {
                .content-grid {
                    grid-template-columns: 45fr 55fr;
                }
            }

            .meta-pill, .signal {
                display: flex;
                align-items: center;
                gap: 8px;
                border-radius: 999px;
                padding: 4px 10px;
                font-family: "Space Mono", monospace;
                font-size: 10px;
                letter-spacing: 0.24em;
                text-transform: uppercase;
                color: var(--text-primary);
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                min-width: 0;
            }

            .meta-pill::before, .signal::before {
                content: "";
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: var(--cyan);
                box-shadow: 0 0 14px rgba(0, 243, 255, 0.55);
                flex-shrink: 0;
            }

            .meta-pill.idle::before, .signal.idle::before {
                background: var(--cyan);
                box-shadow: 0 0 14px rgba(0, 243, 255, 0.55);
            }

            .meta-pill.busy::before, .signal.busy::before {
                background: var(--cyan);
                box-shadow: 0 0 14px rgba(0, 243, 255, 0.55);
            }

            .meta-pill.ready,
            .signal.ready {
                background: rgba(0, 243, 255, 0.06);
                border-color: rgba(0, 243, 255, 0.18);
            }

            .meta-pill.ready::before,
            .signal.ready::before {
                background: var(--cyan);
                box-shadow: 0 0 14px rgba(0, 243, 255, 0.55);
            }

            .section-label {
                display: block;
                margin-bottom: 16px;
                color: var(--text-muted);
                font-size: 0.85rem;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                font-family: "Space Mono", monospace;
                opacity: 0.65;
            }

            textarea {
                width: 100%;
                min-height: 120px;
                padding: 16px;
                font-family: "Inter", system-ui, sans-serif;
                font-size: 15px;
                line-height: 1.6;
                color: var(--text-primary);
                resize: vertical;
                border-radius: 12px;
                border: 1px solid var(--field-border);
                border-left: 2px solid rgba(139, 92, 246, 0.40);
                background: var(--field-bg);
                outline: none;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            }

            textarea:focus {
                border-color: rgba(0, 243, 255, 0.28);
                background: rgba(11, 16, 21, 0.9);
                box-shadow: 0 0 0 1px rgba(0, 243, 255, 0.10), 0 0 20px rgba(0, 243, 255, 0.10);
            }

            textarea::placeholder {
                color: var(--placeholder);
            }

            .drop-zone {
                margin-top: 16px;
                border-radius: 12px;
                border: 1px dashed rgba(139, 92, 246, 0.35);
                background: rgba(139, 92, 246, 0.06);
                padding: 12px 16px;
                cursor: pointer;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            }

            .drop-zone:hover,
            .drop-zone.drag-over {
                border-color: rgba(0, 243, 255, 0.28);
                background: rgba(9, 18, 23, 0.92);
                box-shadow: 0 0 24px rgba(0, 243, 255, 0.08);
            }

            .drop-zone.busy {
                border-color: rgba(0, 243, 255, 0.22);
                background: rgba(14, 21, 28, 0.96);
            }

            .drop-inline {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: center;
                min-height: 24px;
            }

            .drop-main {
                display: flex;
                align-items: center;
                gap: 12px;
                min-width: 0;
                flex: 1;
            }

            .drop-icon {
                width: 18px;
                height: 18px;
                color: var(--fuchsia);
                flex-shrink: 0;
            }

            .drop-message {
                color: var(--text-secondary);
                font-size: 14px;
                line-height: 1.6;
                font-family: "Inter", system-ui, sans-serif;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .drop-badge {
                flex-shrink: 0;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(139, 92, 246, 0.10);
                border: 1px solid rgba(139, 92, 246, 0.35);
                font-size: 10px;
                letter-spacing: 0.24em;
                text-transform: uppercase;
                color: var(--text-primary);
                font-family: "Space Mono", monospace;
            }

            .filename {
                display: inline;
                margin-left: 8px;
                font-weight: 600;
                color: var(--text-primary);
                font-family: "Inter", system-ui, sans-serif;
            }

            .actions {
                display: block;
                margin-top: 16px;
                padding-top: 24px;
                width: 100%;
            }

            button {
                width: 100%;
                padding: 14px 16px;
                border: 1px solid rgba(0, 243, 255, 0.22);
                border-radius: 12px;
                font-family: "Space Mono", monospace;
                font-weight: 600;
                letter-spacing: 0.02em;
                color: rgba(224, 251, 255, 0.98);
                cursor: pointer;
                background: #2a243b;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), inset 0 0 24px rgba(0, 243, 255, 0.05), 0 12px 24px rgba(0, 0, 0, 0.25);
                transition: background 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, transform 0.18s ease, opacity 0.18s ease;
            }

            button:hover:not(:disabled) {
                background: #312944;
                border-color: rgba(0, 243, 255, 0.42);
                color: #ffffff;
                box-shadow: inset 0 0 24px rgba(0, 243, 255, 0.08), 0 0 28px rgba(0, 243, 255, 0.12);
                transform: translateY(-1px);
            }

            .button-processing {
                animation: pulse-glow 1.8s ease-in-out infinite;
                background: rgba(0, 243, 255, 0.08);
                border-color: rgba(0, 243, 255, 0.30);
                color: var(--cyan);
                cursor: wait;
                pointer-events: none;
            }

            .secondary-button {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                color: var(--text-primary);
                box-shadow: none;
            }

            .secondary-button:hover:not(:disabled) {
                background: rgba(255, 255, 255, 0.06);
                border-color: rgba(139, 92, 246, 0.30);
                box-shadow: 0 0 24px rgba(139, 92, 246, 0.12);
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
                align-items: baseline;
                margin-bottom: 16px;
            }

            .output-title {
                font-size: 22px;
                letter-spacing: -0.02em;
                font-weight: 600;
                color: var(--text-primary);
            }

            pre {
                min-height: 400px;
                white-space: pre-wrap;
                word-wrap: break-word;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid var(--field-border);
                background: rgba(0, 0, 0, 0.35);
                color: var(--text-secondary);
                font-family: "Space Mono", monospace;
                font-size: 14px;
                line-height: 1.7;
                position: relative;
                overflow: hidden;
                flex: 1;
                text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.9);
            }

            pre::before {
                content: "";
                position: absolute;
                inset: 0;
                background: repeating-linear-gradient(
                    to bottom,
                    transparent 0,
                    transparent 2px,
                    rgba(255, 255, 255, 1) 2px,
                    rgba(255, 255, 255, 1) 4px
                );
                opacity: 0.03;
                pointer-events: none;
            }

            pre::after {
                content: "";
                position: relative;
                z-index: 1;
                margin-left: 6px;
            }

            .output-panel:has(#response-state.idle) pre::after,
            .output-panel:has(#response-state.ready) pre::after {
                content: "█";
                color: var(--cyan);
                animation: blink 1s steps(1, end) infinite;
            }

            .output-panel {
                display: flex;
                flex-direction: column;
            }

            .typing-dots::after {
                content: "";
                animation: dots 1.2s steps(1) infinite;
            }

            pre.typing-dots::after {
                content: none;
                animation: none;
            }

            .output-ready {
                animation: fade-in-text 0.4s ease both;
            }

            @keyframes card-enter {
                from {
                    opacity: 0;
                    transform: translateY(16px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes rotate-border {
                from { --angle: 0deg; }
                to { --angle: 360deg; }
            }

            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }

            @keyframes pulse-glow {
                0%, 100% { box-shadow: inset 0 0 24px rgba(0, 243, 255, 0.05), 0 0 20px rgba(0, 243, 255, 0.08); }
                50% { box-shadow: inset 0 0 24px rgba(0, 243, 255, 0.12), 0 0 36px rgba(0, 243, 255, 0.18); }
            }

            @keyframes dots {
                0% { content: ""; }
                25% { content: "."; }
                50% { content: ".."; }
                75% { content: "..."; }
            }

            @keyframes fade-in-text {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 899px) {
                .top-bar {
                    max-height: none;
                }

                .topbar-row,
                .output-header,
                .drop-inline {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .topbar-pills {
                    width: 100%;
                    justify-content: flex-start;
                    flex-wrap: wrap;
                }

                pre {
                    min-height: 320px;
                }
            }

            @media (max-width: 700px) {
                .page-shell {
                    width: min(100% - 32px, 1280px);
                    padding: 24px 0;
                }

                .composer,
                .output-panel {
                    padding: 24px;
                }

                .top-bar {
                    padding: 12px 16px;
                }
            }
        </style>
    </head>
    <body>
        <div class="background-orbs" aria-hidden="true">
            <div class="orb orb-violet"></div>
            <div class="orb orb-fuchsia"></div>
            <div class="orb orb-mix"></div>
        </div>
        <div class="page-shell">
            <section class="panel hero top-bar">
                <div class="topbar-row">
                    <div class="brand">
                        <svg class="brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3z"></path>
                            <path d="M9.5 12.5l1.7 1.7 3.3-4"></path>
                        </svg>
                        <div class="app-title">Dark Signal AI</div>
                    </div>
                    <div class="topbar-pills">
                        <span id="file-status" class="meta-pill idle">No file attached</span>
                        <span class="meta-pill">FastAPI + AI Workflow</span>
                    </div>
                </div>
            </section>

            <div class="content-grid">
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
                        <div class="drop-inline">
                            <div class="drop-main">
                                <svg class="drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                    <path d="M12 16V4"></path>
                                    <path d="M8 8l4-4 4 4"></path>
                                    <path d="M20 15v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3"></path>
                                </svg>
                                <div class="drop-message">File attached<span class="filename">${escapeHtml(uploadedFile.name)}</span></div>
                            </div>
                            <div class="drop-badge">Attached</div>
                        </div>
                    `;
                    return;
                }

                dropZone.innerHTML = `
                    <div class="drop-inline">
                        <div class="drop-main">
                            <svg class="drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M12 16V4"></path>
                                <path d="M8 8l4-4 4 4"></path>
                                <path d="M20 15v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3"></path>
                            </svg>
                            <div class="drop-message">Drop file or click to attach</div>
                        </div>
                        <div class="drop-badge">Any format</div>
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

            output.addEventListener("animationend", () => output.classList.remove("output-ready"));

            async function handleFile(file) {
                dropZone.classList.add("busy");
                dropZone.innerHTML = `
                    <div class="drop-inline">
                        <div class="drop-main">
                            <svg class="drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M12 16V4"></path>
                                <path d="M8 8l4-4 4 4"></path>
                                <path d="M20 15v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3"></path>
                            </svg>
                            <div class="drop-message">Uploading<span class="filename">${escapeHtml(file.name)}</span></div>
                        </div>
                        <div class="drop-badge">Syncing</div>
                    </div>
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
                btn.classList.add("button-processing");
                btn.innerHTML = '<span class="typing-dots">Analyzing</span>';
                output.classList.remove("output-ready");
                output.classList.add("typing-dots");
                output.textContent = "";
                output.innerHTML = '<span class="typing-dots">Processing your request</span>';
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
                if (data.reasoning) {
                    output.textContent = "💭 Thinking:\n" + data.reasoning + "\n\n---\n\n" + (data.answer || data.error);
                } else {
                    output.textContent = data.answer || data.error;
                }
                output.classList.remove("typing-dots");
                btn.classList.remove("button-processing");
                output.classList.add("output-ready");
                setResponseState("Response ready", "ready");

                btn.textContent = "Run Analysis";
                btn.disabled = false;
            }
        </script>
    </body>
    </html>
    """
