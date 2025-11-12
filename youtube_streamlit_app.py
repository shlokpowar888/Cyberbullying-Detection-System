# youtube_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import base64
import json
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
import matplotlib.pyplot as plt

# optional seaborn; fallback to matplotlib-only if not present
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

# transformers & torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


st.set_page_config(page_title="Cyberbullying Detector Pro", layout="wide", page_icon="🛡️")
st.title("🛡️ Cyberbullying Detector Pro — YouTube / CSV / Manual")

# ---------------------------
# ---- Configurable params ---
# ---------------------------
ALERT_THRESHOLD_PERCENT = st.sidebar.slider("Alert threshold (% toxic to trigger alert)", 5, 90, 30)
TOP_USERS_N = st.sidebar.slider("Top N toxic users to show", 1, 20, 5)
MAX_YT_COMMENTS = st.sidebar.slider("Max YouTube comments to fetch (per video)", 10, 500, 200)

# ---------------------------
# ---- Helper utilities -----
# ---------------------------
@st.cache_resource
def load_label_map(path="./trained_model/label_map.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        inv_map = {v: k for k, v in label_map.items()}
        return label_map, inv_map
    except Exception as e:
        st.error("Error loading label_map.json from ./trained_model/. Make sure it exists.")
        raise e

@st.cache_resource
def load_model_and_tokenizer(model_dir="./trained_model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error("Error loading model/tokenizer from ./trained_model. Check files and transformers/torch versions.")
        raise e

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # preserve languages but remove excessive control chars
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.strip()
    return text

def predict_comments_list(comments, model, tokenizer, device, inv_label_map):
    model.eval()
    preds = []
    batch_size = 16
    for i in range(0, len(comments), batch_size):
        batch_texts = [clean_text(t) for t in comments[i:i+batch_size]]
        enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        ids = torch.argmax(logits, dim=1).cpu().numpy()
        for idx in ids:
            preds.append(inv_label_map.get(int(idx), "unknown"))
    return preds

def fetch_youtube_comments(video_id, api_key, max_comments=500):

    # returns DataFrame with columns: user, comment
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    page_token = None
    fetched = 0
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - fetched),
            pageToken=page_token,
            textFormat="plainText"
        )
        response = request.execute()
        items = response.get("items", [])
        for item in items:
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "user": snip.get("authorDisplayName", "unknown"),
                "comment": snip.get("textDisplay", "")
            })
            fetched += 1
            if fetched >= max_comments:
                break
        page_token = response.get("nextPageToken")
        if not page_token or fetched >= max_comments:
            break
    return pd.DataFrame(comments)

def df_to_download_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# ---- Load model & labels -
# ---------------------------
try:
    label_map, inv_label_map = load_label_map()
    model, tokenizer, device = load_model_and_tokenizer()
except Exception:
    st.stop()

# ---------------------------
# ---- Layout: Inputs ------
# ---------------------------
st.sidebar.header("Input Mode")
input_mode = st.sidebar.selectbox("Choose input", ["YouTube URL", "Upload CSV", "Manual Comment"])

st.sidebar.markdown("---")
st.sidebar.info("Model loaded from ./trained_model. Ensure label_map.json exists and maps label->id.")

# ---------------------------
# ---- Main logic per mode -
# ---------------------------
df = None
if input_mode == "YouTube URL":
    st.header("🔗 Analyze YouTube Video Comments")
    video_url = st.text_input("Paste YouTube video URL (or video ID):")
    fetch_btn = st.button("Fetch Comments")

    if fetch_btn:
        if not video_url.strip():
            st.warning("Please enter a YouTube URL or ID.")
        else:
            # Extract video ID
            video_id = None
            if "v=" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = video_url.strip()

            load_dotenv()
            api_key = os.getenv("YOUTUBE_API_KEY")

            if not api_key:
                st.error("⚠️ YOUTUBE_API_KEY not found in .env file. Add it and restart.")
            else:
                with st.spinner("Fetching comments from YouTube..."):
                    try:
                        df = fetch_youtube_comments(video_id, api_key, max_comments=MAX_YT_COMMENTS)
                        if df.empty:
                            st.warning("No comments fetched. The video might have comments disabled.")
                        else:
                            df["comment"] = df["comment"].astype(str)
                            df["user"] = df.get("user", "unknown")
                            st.success(f"✅ Fetched {len(df)} comments successfully.")
                            st.session_state["df"] = df
                            st.session_state["source"] = "YouTube"
                    except Exception as e:
                        st.error(f"Error while fetching comments: {e}")

elif input_mode == "Upload CSV":
    st.header("📂 Upload CSV (must contain 'comment' column; optional 'user' or 'user_id')")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, on_bad_lines='skip', engine='python')
            # standardize column names
            cols = [c.lower() for c in df.columns]
            if "comment" not in cols and "text" not in cols and "body" not in cols:
                st.error("CSV must have a 'comment' or 'text' column. Rename and reupload.")
            else:
                # pick column
                if "comment" in cols:
                    comment_col = df.columns[cols.index("comment")]
                elif "text" in cols:
                    comment_col = df.columns[cols.index("text")]
                else:
                    comment_col = df.columns[cols.index("body")]
                df = df.rename(columns={comment_col: "comment"})
                # find user column
                user_col = None
                for candidate in ["user", "user_id", "author", "username", "name"]:
                    if candidate in cols:
                        user_col = df.columns[cols.index(candidate)]
                        df = df.rename(columns={user_col: "user"})
                        break
                if "user" not in df.columns:
                    df["user"] = "unknown"
                df = df[["user", "comment"]].dropna(subset=["comment"])
                st.success(f"Loaded {len(df)} rows from uploaded CSV.")
                st.session_state["df"] = df
                st.session_state["source"] = "Uploaded CSV"

        except Exception as e:
            st.error(f"Could not read CSV: {e}")

elif input_mode == "Manual Comment":
    st.header("✍️ Manual Single Comment Analysis")
    manual_comment = st.text_area("Enter the comment text here:")
    if st.button("Analyze Comment"):
        if manual_comment.strip() == "":
            st.warning("Please enter text to analyze.")
        else:
            preds = predict_comments_list([manual_comment], model, tokenizer, device, inv_label_map)
            st.subheader("Prediction Result")
            st.write(f"**Label:** {preds[0]}")
# Restore previously fetched/analyzed data from session
if "analyzed_df" in st.session_state:
    df = st.session_state["analyzed_df"]
elif "fetched_df" in st.session_state:
    df = st.session_state["fetched_df"]

if input_mode == "Manual Comment":
    # clear any previous session data
    for key in ["df", "fetched_df", "analyzed_df"]:
        if key in st.session_state:
            del st.session_state[key]


# ---------------------------
# ---- If df present -> analyze
# ---------------------------
if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]
    source = st.session_state.get("source", "Unknown Source")
    st.subheader(f"🧩 Analyzing Comments from {source}")

    # ensure columns exist
    if "user" not in df.columns:
        df["user"] = "unknown"
    df["comment"] = df["comment"].astype(str)
    st.subheader("Raw comments (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)

    # Predict
    if st.button("Run Classification on loaded comments"):
        with st.spinner("Classifying comments..."):
            comments = df["comment"].tolist()
            preds = predict_comments_list(comments, model, tokenizer, device, inv_label_map)
            df["prediction"] = preds
        st.success("Classification complete.")
        st.session_state["analyzed_df"] = df

        # Summary stats
        counts = df["prediction"].value_counts()
        total = len(df)
        toxic_count = total - counts.get("safe", 0) if "safe" in counts.index else total - counts.get("non_toxic", 0)
        toxic_percent = round((toxic_count / total) * 100, 2)

        st.markdown("### 📈 Summary")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.metric("Total comments", total)
            st.metric("Flagged (non-safe)", f"{toxic_count} ({toxic_percent}%)")
        with col2:
            st.metric("Distinct users", df["user"].nunique())
        with col3:
            st.metric("Unique labels", ", ".join(map(str, counts.index.tolist())))

        # Alert
        if toxic_percent >= ALERT_THRESHOLD_PERCENT:
            st.error(f"⚠️ ALERT: {toxic_percent}% comments flagged — exceeds threshold of {ALERT_THRESHOLD_PERCENT}%")
        elif toxic_percent > 0:
            st.warning(f"⚠️ Warning: {toxic_percent}% comments flagged")
        else:
            st.success("✅ No toxic comments detected.")

        # Visualizations
        st.markdown("### 📊 Visualizations")
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            fig1, ax1 = plt.subplots(figsize=(5,4))
            if counts.empty:
                ax1.text(0.5, 0.5, "No data", ha="center")
            else:
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                ax1.set_title("Label Distribution")
            st.pyplot(fig1)

        with vis_col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            if SEABORN_AVAILABLE:
                sns.countplot(data=df, x="prediction", order=counts.index, ax=ax2)
            else:
                ax2.bar(counts.index.astype(str), counts.values)
            ax2.set_title("Comments per Category")
            ax2.set_xlabel("Label")
            ax2.set_ylabel("Count")
            for rect in ax2.patches:
                height = rect.get_height()
                ax2.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
            st.pyplot(fig2)

        # Download full CSV
        csv_bytes = df_to_download_bytes(df)
        st.download_button("💾 Download full analysis CSV", csv_bytes, "comment_analysis_full.csv", "text/csv")

        # Group by user and top toxic users
        st.markdown("### 👥 Top toxic users")
        # define toxic labels (anything not 'safe')
        def is_toxic(lbl):
            if isinstance(lbl, str) and lbl.lower() in ("safe", "non_toxic", "clean", "benign"):
                return False
            return True
        df["is_toxic"] = df["prediction"].apply(is_toxic)
        user_toxic_counts = df.groupby("user")["is_toxic"].sum().sort_values(ascending=False)
        if not user_toxic_counts.empty:
            st.table(user_toxic_counts.head(TOP_USERS_N).reset_index().rename(columns={"is_toxic":"toxic_count"}))
            flagged_users = user_toxic_counts[user_toxic_counts > 0].index.tolist()
        else:
            st.info("No flagged users found.")

        # Show sample flagged comments
        st.markdown("### 🧾 Sample flagged comments")
        flagged_df = df[df["is_toxic"]].copy()
        if flagged_df.empty:
            st.info("No flagged comments to show.")
        else:
            st.dataframe(flagged_df[["user","comment","prediction"]].head(200), use_container_width=True)
            # allow filter by label
            selected_label = st.selectbox("Filter flagged by label", options=["all"] + sorted(df["prediction"].unique().tolist()))
            if selected_label != "all":
                st.dataframe(df[df["prediction"] == selected_label][["user","comment"]].head(200), use_container_width=True)

        # Create a PNG image (bar chart) bytes to embed in PDF
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        fig_bytes = buf.read()

        st.success("All done — review the charts and flagged comments above.")

# ---------------------------
# ---- Footer / tips -------
# ---------------------------
st.markdown("---")
st.caption("Tips: If predictions seem poor, consider retraining the MURIL model with more diverse Hinglish / abusive / hate-speech examples and save new model to ./trained_model.")
