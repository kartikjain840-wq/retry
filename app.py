import streamlit as st
import json
import time
import sqlite3
import pdfplumber
import docx
from pptx import Presentation
import pandas as pd
from openai import OpenAI, RateLimitError

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Document Intelligence Dashboard", layout="wide")

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# ---------------- DATABASE ----------------
conn = sqlite3.connect("documents.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    objective TEXT,
    tools TEXT,
    results TEXT,
    industry TEXT,
    region TEXT,
    client_type TEXT
)
""")
conn.commit()

# ---------------- SAFE JSON HANDLING ----------------
def safe_json_list(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return [str(value)]

# ---------------- BULLET NORMALIZATION ----------------
# Rule:
# - split on commas
# - each bullet on its own line
def normalize_bullets(value):
    raw_items = safe_json_list(value)
    bullets = []

    for item in raw_items:
        parts = [p.strip() for p in str(item).split(",") if p.strip()]
        bullets.extend(parts)

    return bullets

def bullets_to_text(value):
    bullets = normalize_bullets(value)
    if not bullets:
        return ""
    return "\n".join([f"• {b}" for b in bullets])

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file, filetype):
    text = ""
    if filetype == "pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif filetype == "docx":
        doc = docx.Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif filetype == "pptx":
        prs = Presentation(file)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    return text[:10000]

# ---------------- AI EXTRACTION ----------------
def ai_extract(text):
    prompt = f"""
You are a management consulting analyst.

Return STRICT JSON.
Objective, Tools, Results must be bullet lists.

{{
  "objective": ["point 1", "point 2"],
  "tools": ["tool 1", "tool 2"],
  "results": ["result 1", "result 2"],
  "industry": "",
  "region": "",
  "client_type": ""
}}

TEXT:
{text}
"""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        except RateLimitError:
            time.sleep(5)
        except Exception:
            break

    return {
        "objective": ["AI extraction failed"],
        "tools": ["AI extraction failed"],
        "results": ["AI extraction failed"],
        "industry": "Unknown",
        "region": "Unknown",
        "client_type": "Unknown"
    }

# ---------------- UI ----------------
st.title("📊 AI Document Intelligence Dashboard")

# 🔍 SEARCH BAR
search_query = st.text_input("🔍 Search across case studies")

tab1, tab2 = st.tabs(["📂 Upload & Process", "📈 Dashboard"])

# ---------------- TAB 1 ----------------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process Documents"):
        for file in uploaded_files:
            filename = file.name
            filetype = filename.split(".")[-1].lower()

            text = extract_text(file, filetype)
            ai_data = ai_extract(text)

            cursor.execute("""
                INSERT INTO documents
                (filename, objective, tools, results, industry, region, client_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                json.dumps(ai_data["objective"]),
                json.dumps(ai_data["tools"]),
                json.dumps(ai_data["results"]),
                ai_data["industry"],
                ai_data["region"],
                ai_data["client_type"]
            ))
            conn.commit()

        st.success("✅ Documents processed")

# ---------------- TAB 2 ----------------
with tab2:
    # Remove duplicates (latest only)
    df = pd.read_sql("""
        SELECT *
        FROM documents
        GROUP BY filename
        ORDER BY id DESC
    """, conn)

    if df.empty:
        st.info("No documents processed yet.")
    else:
        if search_query:
            df = df[df.apply(
                lambda r: search_query.lower() in " ".join(r.astype(str)).lower(),
                axis=1
            )]

        df["Objectives"] = df["objective"].apply(bullets_to_text)
        df["Results"] = df["results"].apply(bullets_to_text)

        # Auto-wrap table cells
        st.markdown(
            """
            <style>
            .stDataFrame td {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                max-width: 600px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            df[[
                "filename",
                "Objectives",
                "Results",
                "industry",
                "region",
                "client_type"
            ]],
            use_container_width=True,
            hide_index=True
        )

        selected = st.selectbox("Select a document", df["filename"].unique())
        row = df[df["filename"] == selected].iloc[0]

        st.markdown("### 🎯 Objective")
        for b in normalize_bullets(row["objective"]):
            st.markdown(f"• {b}")

        st.markdown("### 🛠 Tools Used")
        for t in safe_json_list(row["tools"]):
            st.markdown(f"- {t}")

        st.markdown("### 📈 Results")
        for b in normalize_bullets(row["results"]):
            st.markdown(f"• {b}")

        st.markdown(f"### 🏭 Industry\n{row['industry']}")
        st.markdown(f"### 🌍 Region\n{row['region']}")
        st.markdown(f"### 👥 Client Type\n{row['client_type']}")
