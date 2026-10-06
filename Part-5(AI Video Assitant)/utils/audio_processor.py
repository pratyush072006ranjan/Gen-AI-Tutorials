import yt_dlp
import os
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_video(url: str) -> str:
    """
    Download a YouTube video using format 18.

    Returns:
        str: Path to the downloaded MP4 file.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "18",

        "extractor_args": {
            "youtube": {
                "player_client": ["mweb"]
            }
        },

        "outtmpl": output_path,
        "noplaylist": True,

        "force_ipv4": True,
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 5,
        "continuedl": True,

        "keepvideo": True,

        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        filename = ydl.prepare_filename(info)

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Video was not downloaded: {filename}"
        )

    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to:
        - WAV
        - Mono
        - 16 kHz
    """

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    print("Converting media to 16 kHz mono WAV...")

    audio = AudioSegment.from_file(input_path)

    if len(audio) == 0:
        raise RuntimeError(
            "Input file contains no usable audio."
        )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"WAV file was not created: {output_path}"
        )

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    audio = AudioSegment.from_wav(wav_path)

    if len(audio) == 0:
        raise RuntimeError(
            "WAV file contains no usable audio."
        )

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:

    if source.startswith(
        ("https://", "http://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading video..."
        )

        # YouTube → MP4
        video_path = download_youtube_video(source)

        print(
            f"Video downloaded: {video_path}"
        )

        # MP4 → 16 kHz mono WAV
        wav_path = convert_to_wav(video_path)

    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        # Local file → 16 kHz mono WAV
        wav_path = convert_to_wav(source)

    print("Chunking Audio...")

    chunks = chunk_audio(wav_path)

    if not chunks:
        raise RuntimeError(
            "No audio chunks were created."
        )

    print(
        f"Audio ready - "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks


if __name__ == "__main__":

    url = "https://youtu.be/qkxuFKqJXWY"

    chunks = process_input(url)

    print("\nGenerated chunks:")

    for chunk in chunks:
        print(chunk)


# To start the POT server:
#
# cd "$HOME\bgutil-ytdlp-pot-provider\server"
# node build/main.js
#
#
# To run this file:
#
# uv run python utils/audio_processor.py