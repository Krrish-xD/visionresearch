"""
Extract GQA images from Hugging Face Parquet shards into data/raw/gqa/images/
"""

import os
import io
import sys
import glob
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GQA_DIR = os.path.join(ROOT, "data", "raw", "gqa")
OUTPUT_IMG_DIR = os.path.join(GQA_DIR, "images")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def main():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow not installed. Please run: pip install pyarrow")
        sys.exit(1)

    # Find all parquet files in val_balanced_images, val_all_images, train_balanced_images, train_all_images
    parquet_patterns = [
        os.path.join(GQA_DIR, "val_balanced_images", "*.parquet"),
        os.path.join(GQA_DIR, "val_all_images", "*.parquet"),
        os.path.join(GQA_DIR, "train_balanced_images", "*.parquet"),
        os.path.join(GQA_DIR, "train_all_images", "*.parquet"),
    ]

    parquet_files = []
    for pat in parquet_patterns:
        files = sorted(glob.glob(pat))
        parquet_files.extend(files)

    if not parquet_files:
        print(f"No parquet files found under {GQA_DIR}/*_images/")
        return

    print(f"Found {len(parquet_files)} parquet file(s) across GQA image folders.")
    print(f"Extracting images to -> {OUTPUT_IMG_DIR}\n")

    total_extracted = 0
    skipped_existing = 0

    for p_path in parquet_files:
        rel_path = os.path.relpath(p_path, ROOT)
        print(f"Processing: {rel_path} ...")
        table = pq.read_table(p_path)
        schema_names = table.column_names

        # Identify image and ID column names
        id_col = next((c for c in ["id", "image_id", "imageId", "key", "name"] if c in schema_names), schema_names[0])
        img_col = next((c for c in ["image", "bytes", "data", "img"] if c in schema_names), None)

        pydict = table.to_pydict()
        ids = pydict.get(id_col, [])
        img_data = pydict.get(img_col, []) if img_col else []

        batch_saved = 0
        for i, item_id in enumerate(ids):
            # Format filename
            clean_id = str(item_id).replace(".jpg", "").replace(".png", "")
            out_filename = f"{clean_id}.jpg"
            out_path = os.path.join(OUTPUT_IMG_DIR, out_filename)

            if os.path.exists(out_path):
                skipped_existing += 1
                continue

            raw_img = img_data[i] if i < len(img_data) else None
            img_bytes = None

            if isinstance(raw_img, dict):
                img_bytes = raw_img.get("bytes")
            elif isinstance(raw_img, (bytes, bytearray)):
                img_bytes = raw_img

            if img_bytes:
                with open(out_path, "wb") as f_out:
                    f_out.write(img_bytes)
                batch_saved += 1
                total_extracted += 1
            else:
                # Fallback if image column is already a PIL object (when using datasets library)
                try:
                    if hasattr(raw_img, "save"):
                        raw_img.save(out_path, format="JPEG")
                        batch_saved += 1
                        total_extracted += 1
                except Exception:
                    pass

        print(f"  -> Extracted {batch_saved} images (Total so far: {total_extracted}, Skipped existing: {skipped_existing})")

    print("\n" + "=" * 60)
    print(f"✅ GQA Image Extraction Complete!")
    print(f"   Target Directory: {OUTPUT_IMG_DIR}")
    print(f"   Total Extracted:  {total_extracted} images")
    print(f"   Already Existed:  {skipped_existing} images")
    print("=" * 60)

if __name__ == "__main__":
    main()
