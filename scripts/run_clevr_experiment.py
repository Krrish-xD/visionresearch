"""Orchestrate the real CLEVR experiment end-to-end in the background.

Mirrors the MMVP testing flow but for the real CLEVR v1.0 dataset:
  1. Wait for the CLEVR download to complete.
  2. Extract the archive.
  3. Convert real CLEVR questions+scenes+images -> data/processed/clevr.jsonl (ITEM_SCHEMA).
  4. Create calibration/evaluation splits (data/splits/clevr_splits.json).
  5. Run VLM inference + the full calibrated-confidence experiment pipeline,
     persisting results under results/ (predictions, metrics tables, figures),
     exactly as is done for MMVP.
"""

import os
import sys
import time
import zipfile
import subprocess
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ZIP_PATH = os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip")
EXTRACT_DIR = os.path.join(ROOT, "data", "raw", "clevr")
EXPECTED_BYTES = 19_021_600_724  # official CLEVR_v1.0.zip content-length
LOG_PATH = os.path.join(ROOT, "results", "clevr_experiment.log")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_download():
    log("Waiting for CLEVR download to complete...")
    stable = 0
    last = 0
    while True:
        if not os.path.exists(ZIP_PATH):
            log(f"  zip not found yet ({ZIP_PATH}); sleeping 30s")
            time.sleep(30)
            continue
        size = os.path.getsize(ZIP_PATH)
        log(f"  zip size: {size/1e9:.3f} GB (target {EXPECTED_BYTES/1e9:.3f} GB)")
        if size >= EXPECTED_BYTES - 1_000_000:
            stable += 1
            if stable >= 2:
                log("Download appears complete.")
                return
        elif size == last:
            # not progressing; still wait (curl -C resumes)
            pass
        else:
            stable = 0
        last = size
        # safety timeout ~3h
        time.sleep(30)


def _images_ready(dest):
    images_dir = os.path.join(dest, "images")
    return (os.path.isdir(os.path.join(images_dir, "train"))
            and os.path.isdir(os.path.join(images_dir, "val")))


def extract():
    dest = os.path.join(EXTRACT_DIR, "CLEVR_v1.0")
    if _images_ready(dest):
        log("CLEVR_v1.0 images already extracted; skipping.")
        return
    log(f"Extracting {ZIP_PATH} -> {dest} ...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            # Resume-friendly: only extract image members that are missing.
            names = zf.namelist()
            image_names = [n for n in names if n.startswith("CLEVR_v1.0/images/")]
            to_do = []
            for n in image_names:
                target = os.path.join(EXTRACT_DIR, n)
                if not os.path.exists(target):
                    to_do.append(n)
            if to_do:
                log(f"  extracting {len(to_do)} image files (resume-safe)...")
                zf.extractall(EXTRACT_DIR, members=to_do)
            else:
                log("  all image files already present.")
            # Ensure non-image metadata is present (questions/scenes)
            for n in names:
                if n.startswith("CLEVR_v1.0/images/"):
                    continue
                target = os.path.join(EXTRACT_DIR, n)
                if not os.path.exists(target):
                    zf.extract(n, EXTRACT_DIR)
        log("Extraction complete.")
    except Exception as e:
        log(f"Extraction error: {e}")
        raise


def run(cmd):
    log(f"RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": ROOT},
                          cwd=ROOT, text=True)
    log(f"EXIT CODE: {proc.returncode}")
    return proc.returncode


def main():
    open(LOG_PATH, "w").close()  # reset log
    log("=== CLEVR real-dataset experiment orchestrator starting ===")
    log(f"Using python: {PY}")

    wait_for_download()
    extract()

    rc = run([PY, "src/datasets/clevr_real.py", "--limit", "500"])
    if rc != 0:
        log("Loader failed; aborting.")
        return rc

    rc = run([PY, "-m", "src.evaluation.make_splits"])
    if rc != 0:
        log("Splits failed; aborting.")
        return rc

    # Full pipeline: real GPU inference (evaluate.py auto-falls-back to mock
    # if the HF weights cannot be loaded). Results stored under results/.
    rc = run([PY, "-m", "src.evaluation.run_experiment",
              "--dataset", "clevr", "--model", "llava-1.5-7b"])
    log("=== CLEVR real-dataset experiment finished ===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
