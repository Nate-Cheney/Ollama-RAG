def extract_youtube_video_id(url):
    from pytube import extract
    return extract.video_id(url)


def extract_youtube_transcript(url):
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_youtube_video_id(url)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # gets a list of all transcripts for the provided video_id

        if transcript_list.find_manually_created_transcript(["en"]):
            return transcript_list.find_manually_created_transcript(["en"]).fetch()
        elif transcript_list.find_generated_transcript(["en"]):
            return transcript_list.find_generated_transcript(["en"]).fetch()
        else:
            print("No transcript found")
    except youtube_transcript_api._errors.NoTranscriptFound:
        return YouTubeTranscriptApi.get_transcript(video_id)


def get_playlist_videos(url: str):
    import requests
    import re
    video_list = list()
    response = requests.get(url)    
    if response.status_code==200:
        match = set(re.findall(r"watch\?v=([^\"&]+)", response.text))
        for video_id in match:
            if "index=" in video_id:
                video_list.append(f"https://www.youtube.com/watch?v={video_id}")

        return sorted(video_list, key=lambda x: int(x.split("index=")[1].split("\\")[0]))
    
    else:
        return None


def get_youtube_title(url: str) -> str:
    import requests
    import re
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Search for the title in the page HTML
            match = re.search(r'<title>(.*?)</title>', response.text)
            if match:
                # Clean up the title (YouTube titles often have "- YouTube" at the end)
                title = match.group(1).replace(" - YouTube", "").strip()
                return title
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def process_transcript(transcript_list):
    transcript = str()
    for dictionary in transcript_list:
        transcript += (dictionary["text"] + "\n")

    return transcript


def transcript_main(url: str):
    import os
    dir_path = r"transcripts"
    original_wd = os.getcwd()

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    os.chdir(dir_path)
    video_title = get_youtube_title(url)
    with open(f"{video_title}.txt", "w") as file:
        file.write(process_transcript(extract_youtube_transcript(url)))

    os.chdir(original_wd)

    return video_title