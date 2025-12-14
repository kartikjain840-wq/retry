import streamlit as st
import os
import json
import time
import sqlite3
import pdfplumber
import docx
from pptx import Presentation
import pandas as pd
from openai import OpenAI
from openai import RateLimitError

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Document Intelligence Dashboard",
    layout="wide"
)

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# ---------------- DATABASE (LOCAL, SAFE) ----------------
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

    return text[:10000]  # token safety

# ---------------- AI EXTRACTION (RATE SAFE) ----------------
def ai_extract(text):
    prompt = f"""
You are a management consulting analyst.

Extract the following and return STRICT JSON only:
{{
  "objective": "",
  "tools": "",
  "results": "",
  "industry": "",
  "region": "",
  "client_type": ""
}}

TEXT:
{text}
"""

    for _ in range(3):  # retry protection
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return json.loads(response.choices[0].message.content)

        except RateLimitError:
            time.sleep(5)

    return {
        "objective": "Rate limit / quota issue",
        "tools": "Rate limit / quota issue",
        "results": "Rate limit / quota issue",
        "industry": "Rate limit / quota issue",
        "region": "Rate limit / quota issue",
        "client_type": "Rate limit / quota issue"
    }

# ---------------- UI ----------------
st.title("📊 AI Document Intelligence Dashboard")

tab1, tab2 = st.tabs(["📂 Upload & Process", "📈 Dashboard"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Upload Consulting Case Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process Documents"):
        for file in uploaded_files:
            filetype = file.name.split(".")[-1].lower()
            text = extract_text(file, filetype)

            ai_data = ai_extract(text)

            cursor.execute("""
                INSERT INTO documents
                (filename, objective, tools, results, industry, region, client_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file.name,
                ai_data["objective"],
                ai_data["tools"],
                ai_data["results"],
                ai_data["industry"],
                ai_data["region"],
                ai_data["client_type"]
            ))
            conn.commit()

        st.success("✅ Documents processed successfully!")

# ---------------- TAB 2 ----------------
with tab2:
    df = pd.read_sql("SELECT * FROM documents", conn)

    if df.empty:
        st.info("No documents processed yet.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            region_filter = st.multiselect("Region", df["region"].unique())
        with col2:
            industry_filter = st.multiselect("Industry", df["industry"].unique())
        with col3:
            tool_filter = st.multiselect("Tools", df["tools"].unique())

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

        selected = st.selectbox("Select a document", df["filename"])

        row = df[df["filename"] == selected].iloc[0]

        st.markdown(f"### 🎯 Objective\n{row['objective']}")
        st.markdown(f"### 🛠 Tools\n{row['tools']}")
        st.markdown(f"### 📈 Results\n{row['results']}")
        st.markdown(f"### 🏭 Industry\n{row['industry']}")
        st.markdown(f"### 🌍 Region\n{row['region']}")
        st.markdown(f"### 👥 Client Type\n{row['client_type']}")
