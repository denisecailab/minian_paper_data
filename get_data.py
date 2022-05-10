import os
import shutil
from zipfile import ZipFile

import requests
from tqdm.auto import tqdm

FILES = {
    "35077159": "./ezTrack_snapshot.zip",
    "35077162": "./minian_snapshot.zip",
    "35064955": "./data.zip",
}


def download_file(url, output):
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length", 0))
        if not total_length > 0:
            raise ValueError("Failed to download from {}".format(url))
        with tqdm.wrapattr(
            r.raw, "read", total=total_length, desc="downloading {}".format(output)
        ) as raw:
            with open(output, "wb") as out:
                shutil.copyfileobj(raw, out)


def download_figshare(file_id, output):
    download_file(
        "https://api.figshare.com/v2/file/download/{}".format(file_id), output
    )


if __name__ == "__main__":
    for fid, output in FILES.items():
        download_figshare(fid, output)
        if output.endswith(".zip"):
            outdir = output.rstrip(".zip")
            try:
                os.remove(outdir)
            except OSError:
                pass
            print("extracting {}".format(outdir))
            with ZipFile(output, "r") as zip:
                zip.extractall(outdir)
            os.remove(output)
