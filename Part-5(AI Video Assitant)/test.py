from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
import os


print(
    "KEY Loaded:",
    bool(os.getenv("SARVAM_API_KEY"))
)

print(
    "CWD:",
    os.getcwd()
)


source = "https://www.youtube.com/shorts/SwtNAXPud4g"

language = "hinglish"


chunks = process_input(source)

transcript = transcribe_all(
    chunks,
    language=language
)


print("\n=== Transcript ===\n")
print(transcript)