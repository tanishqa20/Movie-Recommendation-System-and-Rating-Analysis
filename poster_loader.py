
import zipfile
from pathlib import Path
from io import BytesIO
from PIL import Image
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent

ZIP_CANDIDATES = [
    *PARENT_DIR.glob("*.zip"),
    *PROJECT_DIR.glob("*.zip"),
]

ZIP_PATH = next(
    (p for p in ZIP_CANDIDATES if zipfile.is_zipfile(p)),
    None
)

@st.cache_data
def get_cover_paths(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        return {
            Path(name).stem: name
            for name in z.namelist()
            if name.startswith("covers/")
            and not name.endswith("/")
            and Path(name).suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp"}
        }

@st.cache_data
def load_poster(zip_path, cover_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        image_bytes = z.read(cover_path)

    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")
