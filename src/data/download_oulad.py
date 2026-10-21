"""
Script tải và giải nén bộ dữ liệu OULAD (Open University Learning Analytics Dataset) từ Figshare.
Dữ liệu sẽ được lưu vào thư mục: data/raw/
"""

import os
import sys
import urllib.request
import zipfile

# Cấu hình encoding console UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Xác định đường dẫn gốc của dự án (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
URL = 'https://ndownloader.figshare.com/files/8606371'
ZIP_PATH = os.path.join(RAW_DIR, 'oulad.zip')


def download_and_extract(force=False):
    os.makedirs(RAW_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if len(csv_files) >= 7 and not force:
        print(f"ℹ️ Đã tìm thấy {len(csv_files)} file CSV trong '{RAW_DIR}'. Bỏ qua bước tải xuống.")
        print("  (Nếu muốn tải lại từ đầu, hãy gọi download_and_extract(force=True))")
    else:
        print(f"⏳ Đang tải bộ dữ liệu OULAD từ Figshare (~44.5 MB)...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("✅ Tải xuống hoàn tất! Đang giải nén tập tin...")

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("✅ Giải nén thành công!")

    print(f"\n📂 Danh sách file dữ liệu trong '{os.path.relpath(RAW_DIR, PROJECT_ROOT)}':")
    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith('.csv'):
            fpath = os.path.join(RAW_DIR, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as cf:
                lines = sum(1 for _ in cf)
            print(f"  - {fname:<25}: {lines:>10,} dòng ({size_mb:>6.2f} MB)")


if __name__ == '__main__':
    download_and_extract()
