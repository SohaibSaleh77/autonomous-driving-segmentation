
import os

from src.utils import download_data

def main():
    os.makedirs("data", exist_ok=True)
    
    # dataset url
    dataset_url = "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz"
    archive_name = "camvid.tgz"
    target_dir = "data/camvid"
    
    download_data(dataset_url, archive_name, target_dir)

if __name__ == "__main__":
    main()
