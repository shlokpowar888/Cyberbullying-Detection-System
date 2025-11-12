import os
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

print("🚀 Script started...")

# Load environment variables
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    print("❌ API key not found. Make sure it's in your .env file as YOUTUBE_API_KEY=")
    exit()

youtube = build("youtube", "v3", developerKey=api_key)

# Replace with any public video ID
video_id = "89oSfqr7xWw"

print("🔍 Fetching comments for video:", video_id)

comments = []
request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=50,
    textFormat="plainText"
)
response = request.execute()

for item in response.get("items", []):
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
    comments.append({"author": author, "comment": comment})

print(f"✅ Fetched {len(comments)} comments.")

if len(comments) > 0:
    df = pd.DataFrame(comments)
    df.to_csv("youtube_comments.csv", index=False)
    print("💾 Saved to youtube_comments.csv")
else:
    print("⚠️ No comments found or invalid video ID.")
