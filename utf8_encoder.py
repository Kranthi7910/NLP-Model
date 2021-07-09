
from pathlib import Path

files = Path("./test").glob("**/*.txt")

for file in files:
    _file = str(file)
    data = None
    with open(_file, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        _new_file = _file.replace("test", "test-UTF8")
        with open(_new_file, "w") as f:
            f.write(data)
