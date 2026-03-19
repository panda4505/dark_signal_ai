from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from openpyxl import load_workbook
import base64
import fitz
import json
import logging
import mimetypes
import os, tempfile
import re
import time
from uuid import uuid4
import zipfile
import xml.etree.ElementTree as ET

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

TEXT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", TEXT_MODEL)
API_TIMEOUT_SECONDS = 900

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=API_TIMEOUT_SECONDS,
)

# Temp directory for uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai_app_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_METADATA_SUFFIX = ".meta.json"
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_TTL_SECONDS = 24 * 60 * 60
CLEANUP_INTERVAL_SECONDS = 300
LAST_CLEANUP_AT = 0.0
FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_PROMPT_CHARS = 10000


class UploadTooLargeError(ValueError):
    pass


def sanitize_uploaded_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return name or "upload"


def get_upload_metadata_path(file_id: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{file_id}{UPLOAD_METADATA_SUFFIX}")


def safe_remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove file during cleanup: %s", path, exc_info=True)


def remove_upload_artifacts(file_id: str, metadata: dict | None = None) -> None:
    safe_remove_file(get_upload_metadata_path(file_id))

    stored_name = metadata.get("stored_name") if metadata else None
    if stored_name:
        safe_remove_file(os.path.join(UPLOAD_DIR, stored_name))


def cleanup_corrupt_upload_artifacts(file_id: str) -> None:
    safe_remove_file(get_upload_metadata_path(file_id))

    try:
        entries = list(os.scandir(UPLOAD_DIR))
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Could not scan upload directory while cleaning corrupt metadata.", exc_info=True)
        return

    file_prefix = f"{file_id}."

    for entry in entries:
        if not entry.is_file():
            continue
        if entry.name != file_id and not entry.name.startswith(file_prefix):
            continue
        safe_remove_file(entry.path)


def load_upload_metadata(file_id: str) -> dict | None:
    metadata_path = get_upload_metadata_path(file_id)
    if not os.path.isfile(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable upload metadata for file_id=%s. Cleaning related artifacts.", file_id)
        cleanup_corrupt_upload_artifacts(file_id)
        return None

    if not isinstance(metadata, dict):
        logger.warning("Invalid upload metadata structure for file_id=%s. Cleaning related artifacts.", file_id)
        cleanup_corrupt_upload_artifacts(file_id)
        return None

    return metadata


def cleanup_expired_uploads() -> None:
    now = time.time()
    cutoff = now - UPLOAD_TTL_SECONDS

    try:
        entries = list(os.scandir(UPLOAD_DIR))
    except FileNotFoundError:
        return

    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(UPLOAD_METADATA_SUFFIX):
            continue

        file_id = entry.name[:-len(UPLOAD_METADATA_SUFFIX)]
        if not FILE_ID_PATTERN.fullmatch(file_id):
            continue

        metadata = load_upload_metadata(file_id)
        created_at = 0.0
        if metadata:
            try:
                created_at = float(metadata.get("created_at", 0))
            except (TypeError, ValueError):
                created_at = 0.0

        if created_at and created_at >= cutoff:
            continue

        remove_upload_artifacts(file_id, metadata)

    for entry in entries:
        if not entry.is_file() or entry.name.endswith(UPLOAD_METADATA_SUFFIX):
            continue

        try:
            is_expired = entry.stat().st_mtime < cutoff
        except FileNotFoundError:
            continue

        if is_expired:
            safe_remove_file(entry.path)


def maybe_cleanup_expired_uploads(force: bool = False) -> None:
    global LAST_CLEANUP_AT

    now = time.time()
    if not force and LAST_CLEANUP_AT and (now - LAST_CLEANUP_AT) < CLEANUP_INTERVAL_SECONDS:
        return

    cleanup_expired_uploads()
    LAST_CLEANUP_AT = now


@app.on_event("startup")
def startup_cleanup() -> None:
    maybe_cleanup_expired_uploads(force=True)


# --- Upload endpoint: just saves the file ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    maybe_cleanup_expired_uploads()

    original_name = sanitize_uploaded_filename(file.filename)
    _, ext = os.path.splitext(original_name)
    file_id = uuid4().hex
    stored_name = f"{file_id}{ext.lower()}"
    path = os.path.join(UPLOAD_DIR, stored_name)
    metadata = {
        "file_id": file_id,
        "stored_name": stored_name,
        "original_name": original_name,
        "created_at": time.time(),
    }
    bytes_written = 0

    try:
        with open(path, "wb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise UploadTooLargeError(
                        f"Files larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB are not supported."
                    )

                f.write(chunk)

        with open(get_upload_metadata_path(file_id), "w", encoding="utf-8") as f:
            json.dump(metadata, f)
    except UploadTooLargeError as e:
        remove_upload_artifacts(file_id, metadata)
        return JSONResponse(status_code=413, content={"error": str(e)})
    except Exception:
        logger.exception("Unexpected error while saving uploaded file.")
        remove_upload_artifacts(file_id, metadata)
        return JSONResponse(
            status_code=500,
            content={"error": "Could not save the uploaded file."},
        )
    finally:
        await file.close()

    return {"file_id": file_id, "filename": original_name}


# --- Ask endpoint ---

class PromptRequest(BaseModel):
    prompt: str
    file_id: str | None = None


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

    if ext in LEGACY_OFFICE_EXTENSIONS:
        return {
            "kind": "text",
            "content": (
                f"[Unsupported legacy Office file type: .{ext}. Please re-save the file as a modern format "
                "such as .docx, .xlsx, or .pptx.]"
            ),
        }

    try:
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
        return {"kind": "text", "content": "[Unsupported file type]"}
    except Exception:
        logger.exception("Could not extract uploaded file content.")
        return {
            "kind": "text",
            "content": "[Could not extract readable content from the uploaded file.]",
        }


def resolve_uploaded_file(file_id: str | None):
    maybe_cleanup_expired_uploads()

    if not file_id:
        return None

    if not FILE_ID_PATTERN.fullmatch(file_id):
        raise ValueError("Invalid uploaded file reference. Please upload the file again.")

    metadata = load_upload_metadata(file_id)
    if not metadata:
        raise ValueError("Uploaded file was not found. Please upload it again.")

    try:
        created_at = float(metadata.get("created_at", 0))
    except (TypeError, ValueError):
        created_at = 0.0

    if created_at and (time.time() - created_at) > UPLOAD_TTL_SECONDS:
        remove_upload_artifacts(file_id, metadata)
        raise ValueError("Uploaded file has expired. Please upload it again.")

    stored_name = metadata.get("stored_name")
    if not stored_name:
        cleanup_corrupt_upload_artifacts(file_id)
        raise ValueError("Uploaded file metadata is invalid. Please upload it again.")

    file_path = os.path.join(UPLOAD_DIR, stored_name)
    if not os.path.isfile(file_path):
        remove_upload_artifacts(file_id, metadata)
        raise ValueError("Uploaded file is no longer available. Please upload it again.")

    return file_path, metadata.get("original_name") or "upload"


def build_request_payload(data: PromptRequest):
    if len(data.prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt is too long. Please keep it under {MAX_PROMPT_CHARS} characters.")

    uploaded_file = resolve_uploaded_file(data.file_id)

    if not uploaded_file:
        return "none", TEXT_MODEL, [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data.prompt},
        ]

    file_path, file_name = uploaded_file
    file_context = build_file_context(file_path, file_name)

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
                            f"Filename: {file_name or '[Uploaded image]'}\n"
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
                f"Filename: {file_name or '[No file uploaded]'}\n\n"
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
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception:
        if file_kind == "image":
            logger.exception("Image analysis failed.")
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Image analysis failed with the configured DeepSeek model. "
                        "If your current model is text-only, set DEEPSEEK_VISION_MODEL to a vision-capable model."
                    )
                },
            )
        logger.exception("Unexpected error while processing ask request.")
        return JSONResponse(
            status_code=500,
            content={"error": "Something went wrong while processing your request."},
        )


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
                overflow: visible;
            }

            .panel:hover {
                background: var(--panel-hover-bg);
                border-color: var(--panel-hover-border);
            }

            .hero { animation-delay: 0s; }
            .composer { animation-delay: 0.08s; }
            .output-panel { animation-delay: 0.16s; }

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

            #send-btn {
                position: relative;
                z-index: 10;
            }

            #drop-zone {
                position: relative;
                z-index: 10;
            }

            #prompt {
                position: relative;
                z-index: 10;
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

            .output-controls {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .output-title {
                font-size: 22px;
                letter-spacing: -0.02em;
                font-weight: 600;
                color: var(--text-primary);
            }

            .copy-button {
                width: auto;
                padding: 8px 12px;
                border-radius: 10px;
                font-size: 11px;
                line-height: 1;
                box-shadow: none;
                flex-shrink: 0;
            }

            .reasoning-toggle {
                width: auto;
                align-self: flex-start;
                margin-top: 16px;
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 11px;
                line-height: 1;
                box-shadow: none;
            }

            .reasoning-panel {
                margin-top: 12px;
                padding: 16px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(9, 12, 18, 0.72);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
                animation: fade-in-text 0.25s ease both;
            }

            .reasoning-label {
                margin-bottom: 10px;
                color: var(--text-muted);
                font-size: 11px;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                font-family: "Space Mono", monospace;
            }

            .reasoning-output {
                white-space: pre-wrap;
                word-break: break-word;
                color: var(--text-secondary);
                font-family: "Space Mono", monospace;
                font-size: 13px;
                line-height: 1.7;
                text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.9);
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

            .robot-loader {
                position: relative;
                z-index: 1;
                width: 100%;
                min-height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 16px;
                text-align: center;
                white-space: normal;
                color: var(--text-secondary);
            }

            .loader-ring {
                width: 44px;
                height: 44px;
                border-radius: 999px;
                border: 3px solid rgba(0, 243, 255, 0.14);
                border-top-color: var(--cyan);
                border-right-color: rgba(232, 121, 249, 0.82);
                box-shadow: 0 0 24px rgba(0, 243, 255, 0.12);
                animation: robot-spin 0.95s linear infinite;
                flex-shrink: 0;
            }

            .loader-text {
                font-family: "Space Mono", monospace;
                font-size: 13px;
                letter-spacing: 0.04em;
                color: var(--cyan);
                text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.9);
            }

            pre.output-loading {
                display: flex;
                align-items: center;
                justify-content: center;
                white-space: normal;
            }

            .typing-dots::after {
                content: "";
                animation: dots 1.2s steps(1) infinite;
            }

            pre.typing-dots::after {
                content: none;
                animation: none;
            }

            pre.output-loading::after {
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

            @keyframes robot-spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
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

                .output-controls {
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

                    <div id="drop-zone" class="drop-zone"></div>
                    <input type="file" id="file-input" hidden>

                    <div class="actions">
                        <button type="button" id="send-btn">Run Analysis</button>
                    </div>
                </section>

                <section class="panel output-panel">
                    <div class="output-header">
                        <div>
                            <div class="section-label">Response Feed</div>
                            <div class="output-title">Model Output</div>
                        </div>
                        <div class="output-controls">
                            <div id="response-state" class="signal idle">Standing by</div>
                            <button type="button" id="copy-output-btn" class="secondary-button copy-button">Copy</button>
                        </div>
                    </div>
                    <pre id="output">Your answer will appear here.</pre>
                    <button type="button" id="reasoning-toggle" class="secondary-button reasoning-toggle" aria-controls="reasoning-panel" aria-expanded="false" hidden>Show reasoning</button>
                    <div id="reasoning-panel" class="reasoning-panel" hidden>
                        <div class="reasoning-label">Model Reasoning</div>
                        <div id="reasoning-output" class="reasoning-output"></div>
                    </div>
                </section>
            </div>
        </div>

        <script>
document.addEventListener("DOMContentLoaded", () => {
    let uploadedFile = null;
    let isUploading = false;
    let isAnalyzing = false;

    const UPLOAD_TIMEOUT_MS = 60000;

    const dropZone = document.getElementById("drop-zone");
    const fileStatus = document.getElementById("file-status");
    const responseState = document.getElementById("response-state");
    const output = document.getElementById("output");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("send-btn");
    const copyButton = document.getElementById("copy-output-btn");
    const fileInput = document.getElementById("file-input");
    const reasoningToggle = document.getElementById("reasoning-toggle");
    const reasoningPanel = document.getElementById("reasoning-panel");
    const reasoningOutput = document.getElementById("reasoning-output");

    function escapeHtml(text) {
        return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }

    function updateSendButton() {
        sendButton.classList.toggle("button-processing", isAnalyzing);

        if (isAnalyzing) {
            sendButton.disabled = true;
            sendButton.textContent = "Analyzing...";
            return;
        }

        if (isUploading) {
            sendButton.disabled = true;
            sendButton.textContent = "Uploading file...";
            return;
        }

        sendButton.disabled = false;
        sendButton.textContent = "Run Analysis";
    }

    function updateCopyButton() {
        copyButton.disabled = isAnalyzing;
    }

    function canStartUpload() {
        return !isUploading && !isAnalyzing;
    }

    function renderDropZone() {
        if (isUploading) {
            dropZone.innerHTML = '<div class="drop-inline"><div class="drop-main"><div class="drop-message">Uploading file...</div></div><div class="drop-badge">Please wait</div></div>';
            return;
        }
        if (uploadedFile) {
            dropZone.innerHTML = '<div class="drop-inline"><div class="drop-main"><div class="drop-message">File attached <span class="filename">' + escapeHtml(uploadedFile.name) + '</span></div></div><div class="drop-badge">Attached</div></div>';
            return;
        }
        dropZone.innerHTML = '<div class="drop-inline"><div class="drop-main"><div class="drop-message">Drop file or click to attach</div></div><div class="drop-badge">Any format</div></div>';
    }

    function setFileStatus(text, stateClass) {
        fileStatus.textContent = text;
        fileStatus.className = "meta-pill " + stateClass;
    }

    function setResponseState(text, stateClass) {
        responseState.textContent = text;
        responseState.className = "signal " + stateClass;
    }

    async function readJsonResponse(res) {
        const contentType = (res.headers.get("content-type") || "").toLowerCase();

        if (contentType.includes("application/json")) {
            let data;
            try {
                data = await res.json();
            } catch (error) {
                throw new Error("Server returned invalid JSON.");
            }
            if (!res.ok) {
                throw new Error(data.error || data.detail || `HTTP ${res.status}`);
            }
            return data;
        }

        const text = (await res.text()).trim();
        if (!res.ok) {
            if (text && !text.startsWith("<!DOCTYPE") && !text.startsWith("<html")) {
                throw new Error(text);
            }
            throw new Error(`HTTP ${res.status}`);
        }

        throw new Error("Server returned an unexpected response.");
    }

    async function fetchWithTimeout(url, options, timeoutMs) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } catch (error) {
            if (error instanceof DOMException && error.name === "AbortError") {
                throw new Error("Request timed out. Please try again.");
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function hasReasoning(reasoning) {
        return typeof reasoning === "string" && reasoning.trim().length > 0;
    }

    function resetReasoningUI() {
        reasoningOutput.textContent = "";
        reasoningPanel.hidden = true;
        reasoningToggle.hidden = true;
        reasoningToggle.textContent = "Show reasoning";
        reasoningToggle.setAttribute("aria-expanded", "false");
    }

    function setReasoning(reasoning) {
        if (!hasReasoning(reasoning)) {
            resetReasoningUI();
            return;
        }

        reasoningOutput.textContent = reasoning;
        reasoningPanel.hidden = true;
        reasoningToggle.hidden = false;
        reasoningToggle.textContent = "Show reasoning";
        reasoningToggle.setAttribute("aria-expanded", "false");
    }

    let copyResetTimer = null;

    function setCopyButtonLabel(text) {
        copyButton.textContent = text;
        if (copyResetTimer) {
            clearTimeout(copyResetTimer);
        }
        if (text !== "Copy") {
            copyResetTimer = setTimeout(() => {
                copyButton.textContent = "Copy";
                copyResetTimer = null;
            }, 1200);
        }
    }

    async function handleFile(file) {
        if (!canStartUpload()) {
            return;
        }
        isUploading = true;
        uploadedFile = null;
        updateSendButton();
        renderDropZone();
        dropZone.classList.add("busy");
        setFileStatus("Uploading file", "busy");
        try {
            const form = new FormData();
            form.append("file", file);
            const res = await fetchWithTimeout("/upload", { method: "POST", body: form }, UPLOAD_TIMEOUT_MS);
            const data = await readJsonResponse(res);
            uploadedFile = { id: data.file_id, name: data.filename };
            setFileStatus("Attached: " + data.filename, "ready");
        } catch (err) {
            uploadedFile = null;
            output.textContent = err instanceof Error ? err.message : "Upload failed.";
            resetReasoningUI();
            setFileStatus("Upload failed", "idle");
        } finally {
            isUploading = false;
            updateSendButton();
            renderDropZone();
            dropZone.classList.remove("busy");
        }
    }

    async function sendPrompt() {
        const prompt = promptInput.value;
        if (isUploading) {
            output.textContent = "Please wait for the file upload to finish.";
            setResponseState("Upload in progress", "busy");
            return;
        }
        if (!prompt.trim()) {
            resetReasoningUI();
            output.textContent = "Please enter a prompt first.";
            setResponseState("No prompt entered", "idle");
            return;
        }
        resetReasoningUI();
        isAnalyzing = true;
        updateSendButton();
        updateCopyButton();
        output.textContent = "";
        output.classList.add("output-loading");
        output.innerHTML = '<span class="robot-loader"><span class="loader-ring"></span><span class="loader-text">The Robot is thinking...</span></span>';
        setResponseState("Analyzing", "busy");
        try {
            const body = { prompt };
            if (uploadedFile) {
                body.file_id = uploadedFile.id;
            }
            const res = await fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(body)
            });
            const data = await readJsonResponse(res);
            output.classList.remove("output-loading");
            output.textContent = data.answer || data.error || "No response.";
            if (hasReasoning(data.reasoning)) {
                setReasoning(data.reasoning);
            }
            setResponseState("Response ready", "ready");
        } catch (error) {
            output.classList.remove("output-loading");
            output.textContent = error instanceof Error ? error.message : "Request failed. Please try again.";
            resetReasoningUI();
            setResponseState("Request failed", "idle");
        } finally {
            isAnalyzing = false;
            updateSendButton();
            updateCopyButton();
        }
    }

    renderDropZone();
    updateSendButton();
    updateCopyButton();

    dropZone.addEventListener("click", () => {
        if (canStartUpload()) {
            fileInput.click();
        }
    });
    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        if (canStartUpload()) {
            dropZone.classList.add("drag-over");
        }
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (canStartUpload() && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", e => {
        const selectedFile = e.target.files && e.target.files[0];
        fileInput.value = "";
        if (canStartUpload() && selectedFile) handleFile(selectedFile);
    });
    sendButton.addEventListener("click", sendPrompt);
    reasoningToggle.addEventListener("click", () => {
        const shouldShow = reasoningPanel.hidden;
        reasoningPanel.hidden = !shouldShow;
        reasoningToggle.textContent = shouldShow ? "Hide reasoning" : "Show reasoning";
        reasoningToggle.setAttribute("aria-expanded", shouldShow ? "true" : "false");
    });
    copyButton.addEventListener("click", async () => {
        if (output.classList.contains("output-loading")) {
            setCopyButtonLabel("Wait");
            return;
        }
        const text = output.textContent.trim();
        if (!text || text === "Your answer will appear here.") {
            setCopyButtonLabel("Empty");
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            setCopyButtonLabel("Copied");
        } catch (error) {
            setCopyButtonLabel("Failed");
        }
    });
});
</script>
    </body>
    </html>
    """
