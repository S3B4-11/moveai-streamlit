"""Empaqueta código completo, sin videos, cachés ni secretos."""
from pathlib import Path
import zipfile
import argparse

def package(root,destination,include_asset=True):
    root=Path(root).resolve();destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    files=[p for p in root.iterdir() if p.is_file() and p.suffix in [".py",".txt",".md"]]
    files += [root/p for p in [".python-version",".gitignore",".streamlit/config.toml"] if (root/p).exists()]
    for folder in (["assets"] if include_asset else [])+["camera_modelos"]:
        files += [p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in [".task",".json",".npz"]]
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(root).as_posix())
    return destination

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default=Path(__file__).parent);p.add_argument("--output",default="MOVEAI_Streamlit.zip")
    a=p.parse_args();print(package(a.root,a.output))
