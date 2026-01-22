import io
import os

import rarfile
import tarfile
import tempfile
import zipfile
from typing import List, Tuple

rarfile.UNRAR_TOOL = "unrar"



def extract_content_from_zip(file: bytes, max_files_to_extract: int = None) -> List[Tuple[str, str]]:
    """Extract log contents from zip file attachment. returns list of (text, filename) tuples."""
    log_contents = []
    with zipfile.ZipFile(io.BytesIO(file), "r") as z:
        files = [f for f in z.namelist() if not f.endswith("/")]

        for f in files[:max_files_to_extract]:
            if not (f.endswith(".log") or f.endswith(".txt")):
                continue

            with z.open(f) as file:
                text = file.read().decode("utf-8", errors="ignore")
                log_contents.append((text, f))

    return log_contents


def extract_content_from_tar(file: bytes, suffix: str ,max_files_to_extract: int = None) -> List[Tuple[str, str]]:
    """Extract log contents from tar file attachment (.rar). returns list of (text, filename) tuples."""
    log_contents = []
    mode = 'r:gz' if suffix in ('gz', 'tgz') else 'r'
    with tarfile.open(fileobj=io.BytesIO(file), mode=mode) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

        for member in members[:max_files_to_extract]:
            if not (member.name.endswith(".log") or member.name.endswith(".txt")):
                continue

            extracted_file = tar.extractfile(member)
            if extracted_file:
                text = extracted_file.read().decode("utf-8", errors="ignore")
                log_contents.append((text, member.name))

    return log_contents


def extract_content_from_rar(file: bytes, max_files_to_extract: int = None) -> List[Tuple[str, str]]:
    """Extract log contents from rar file attachment. returns list of (text, filename) tuples."""
    log_contents: List[Tuple[str, str]] = []
    tmp_path = None
    extract_dir = None

    try:
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=".rar") as tmp:
            tmp.write(file)
            tmp_path = tmp.name

        extract_dir = tempfile.mkdtemp()

        with rarfile.RarFile(tmp_path) as rf:
            rf.extractall(extract_dir)

        count = 0
        for root, _dirs, filenames in os.walk(extract_dir):
            for fname in filenames:
                if max_files_to_extract is not None and count >= max_files_to_extract:
                    break
                if not (fname.endswith(".log") or fname.endswith(".txt")):
                    continue

                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                log_contents.append((text, fname))
                count += 1

        return log_contents

    except Exception:
        # Re-raise the original exception (preserves type + traceback)
        raise

    finally:
        # Cleanup only; do not raise here
        try:
            import shutil
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if extract_dir and os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
        except Exception:
            # Don't mask the real error with cleanup failures
            pass
