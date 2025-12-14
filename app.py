import streamlit as st
import os
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
    filename TEXT UNIQUE,
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

    return text[:10000]

# ---------------- AI EXTRACTION ----------------
def ai_extract(text):
    prompt = f"""
You are a management consulting analyst.

Extract information and return STRICT JSON.
Write Objective, Tools, and Results as BULLET POINTS.

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

    return None

# ---------------- UI ----------------
st.title("📊 AI Document Intelligence Dashboard")

# 🔍 SEARCH BAR (TOP)
search_query = st.text_input("🔍 Search across case studies (file name, objective, tools, results)")

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

            # 🚫 Prevent duplicate insertion
            cursor.execute("SELECT 1 FROM documents WHERE filename=?", (filename,))
            if cursor.fetchone():
                continue

            filetype = filename.split(".")[-1].lower()
            text = extract_text(file, filetype)
            ai_data = ai_extract(text)

            if not ai_data:
                continue

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

        st.success("✅ Documents processed (duplicates skipped).")

# ---------------- TAB 2 ----------------
with tab2:
    df = pd.read_sql("SELECT * FROM documents", conn)

    if df.empty:
        st.info("No documents processed yet.")
    else:
        # 🔍 SEARCH FILTER
        if search_query:
            df = df[
                df.apply(
                    lambda row: search_query.lower() in " ".join(row.astype(str)).lower(),
                    axis=1
                )
            ]

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
            df[["filename", "industry", "region", "client_type"]],
            use_container_width=True
        )

        selected = st.selectbox("Select a document", df["filename"].unique())
        row = df[df["filename"] == selected].iloc[0]

        # 🔹 BULLET POINT DISPLAY
        st.markdown("### 🎯 Objective")
        for o in json.loads(row["objective"]):
            st.markdown(f"- {o}")

        st.markdown("### 🛠 Tools Used")
        for t in json.loads(row["tools"]):
            st.markdown(f"- {t}")

        st.markdown("### 📈 Results")
        for r in json.loads(row["results"]):
            st.markdown(f"- {r}")

        st.markdown(f"### 🏭 Industry\n{row['industry']}")
        st.markdown(f"### 🌍 Region\n{row['region']}")
        st.markdown(f"### 👥 Client Type\n{row['client_type']}")
