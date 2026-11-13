"""Download MovieLens Latest Small from its original publisher (educational use)."""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

def ensure_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dest = root / 'ml-latest-small'
    if all((dest / name).exists() for name in ('ratings.csv', 'movies.csv', 'tags.csv')):
        return dest
    archive = root / 'ml-latest-small.zip'
    print(f'Downloading {URL}')
    urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if '..' in Path(member).parts or Path(member).is_absolute():
                raise ValueError('Unsafe archive entry')
        zf.extractall(root)
    return dest

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data');a=p.parse_args()
    print(ensure_dataset(Path(a.root)))
