import streamlit as st
import os
import pdfplumber
import docx
from pptx import Presentation
import sqlite3
import pandas as pd
import json
from openai import OpenAI

# ---------------- CONFIG ---------------- #
st.set_page_config("AI Document Intelligence Dashboard", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DB_PATH = "doc_data.db"
DOC_FOLDER = "documents"

os.makedirs(DOC_FOLDER, exist_ok=True)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    client_type TEXT,
    full_text TEXT
)
""")
conn.commit()

# ---------------- TEXT EXTRACTORS ---------------- #
def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_pptx(path):
    prs = Presentation(path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_pdf(file_path)
    if file_path.endswith(".docx"):
        return extract_docx(file_path)
    if file_path.endswith(".pptx"):
        return extract_pptx(file_path)
    return ""

# ---------------- AI EXTRACTION ---------------- #
def ai_extract(text):
    prompt = f"""
    You are a management consulting analyst.

    From the document text below, extract:
    1. Objective
    2. Tools / Frameworks Used
    3. Results / Impact
    4. Industry
    5. Region / Geography
    6. Client Type

    Return STRICT JSON with keys:
    objective, tools, results, industry, region, client_type

    TEXT:
    {text[:12000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# ---------------- UI ---------------- #
st.title("📊 AI Document Intelligence Dashboard")

tab1, tab2 = st.tabs(["📂 Upload & Process", "📈 Dashboard"])

# ---------------- UPLOAD TAB ---------------- #
with tab1:
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / PPT / DOCX",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            path = os.path.join(DOC_FOLDER, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            text = extract_text(path)

            ai_data = ai_extract(text)

            cursor.execute("""
            INSERT INTO documents
            (filename, objective, tools, results, industry, region, client_type, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file.name,
                ai_data["objective"],
                ai_data["tools"],
                ai_data["results"],
                ai_data["industry"],
                ai_data["region"],
                ai_data["client_type"],
                text
            ))
            conn.commit()

        st.success("Documents processed successfully!")

# ---------------- DASHBOARD TAB ---------------- #
with tab2:
    df = pd.read_sql("SELECT * FROM documents", conn)

    if df.empty:
        st.info("No documents processed yet.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            region_filter = st.multiselect("Filter by Region", df["region"].unique())

        with col2:
            industry_filter = st.multiselect("Filter by Industry", df["industry"].unique())

        with col3:
            tool_filter = st.multiselect("Filter by Tools", df["tools"].unique())

        if region_filter:
            df = df[df["region"].isin(region_filter)]
        if industry_filter:
            df = df[df["industry"].isin(industry_filter)]
        if tool_filter:
            df = df[df["tools"].isin(tool_filter)]

        st.dataframe(
            df[["filename", "objective", "tools", "results", "industry", "region", "client_type"]],
            use_container_width=True
        )

        st.subheader("📄 Document Details")

        selected = st.selectbox("Select document", df["filename"])

        doc = df[df["filename"] == selected].iloc[0]

        st.markdown(f"### 🎯 Objective\n{doc['objective']}")
        st.markdown(f"### 🛠 Tools Used\n{doc['tools']}")
        st.markdown(f"### 📈 Results\n{doc['results']}")
        st.markdown(f"### 🌍 Region\n{doc['region']}")
        st.markdown(f"### 🏭 Industry\n{doc['industry']}")
        st.markdown(f"### 👥 Client Type\n{doc['client_type']}")
