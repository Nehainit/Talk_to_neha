import glob

def load_markdown_files(path_pattern: str = "data/*.md") -> str:
    combined = ""
    for p in glob.glob(path_pattern):
        with open(p, "r", encoding="utf-8") as f:
            combined += f.read() + "\n\n"
    return combined
