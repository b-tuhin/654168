"""Runs your three Colab cells (src/cell1.py, cell2.py, cell3.py) unchanged, outside Colab.

What this does for you:
  * replaces the Google Drive / /content paths with the repo's data/ folder
  * unzips data/euroc_small.zip (the old setup cell) and passes data_root to Cell 1
  * runs the cells in ONE shared namespace (Cell 2 and 3 need Cell 1's variables)
  * saves every figure to outputs/ instead of opening a window, and logs the text output

Usage:
    python run.py                         # uses SOURCE / TEST as written in cell1.py
    python run.py --source device --test square
    python run.py --source euroc
"""
import argparse, io, os, re, sys, tempfile, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display on a server / GitHub runner
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC, DATA, OUT = ROOT / "src", ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)


class Tee(io.TextIOBase):
    """Print to the screen and to outputs/log.txt at the same time."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams: st.flush()


def find_euroc_root():
    """Old setup cell: unzip euroc_small.zip and find the folder that contains mav0."""
    zip_path, out = DATA / "euroc_small.zip", DATA / "_euroc_extracted"
    if zip_path.exists() and not out.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    search = out if out.exists() else DATA / "euroc_small"
    for root, dirs, _ in os.walk(search):
        if "mav0" in dirs:
            return root
    return None


def patch(text, source, test, tmp):
    text = text.replace("/content/drive/MyDrive/colab_upload/phone/", (DATA / "phone").as_posix() + "/")
    text = text.replace("/content/device_unzipped/", Path(tmp).as_posix() + "/")
    if source:
        text = re.sub(r'^SOURCE = "[^"]*"', 'SOURCE = "%s"' % source, text, count=1, flags=re.M)
    if test:
        text = re.sub(r'^TEST = "[^"]*"', 'TEST = "%s"' % test, text, count=1, flags=re.M)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["euroc", "device"])
    ap.add_argument("--test", choices=["square", "slide", "none"])
    args = ap.parse_args()

    log = open(OUT / "log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log)

    counter = {"n": 0}
    def save_and_close(*a, **k):                      # replaces plt.show()
        for num in plt.get_fignums():
            counter["n"] += 1
            f = OUT / ("fig_%02d.png" % counter["n"])
            plt.figure(num).savefig(f, dpi=130, bbox_inches="tight")
            print("[saved %s]" % f.relative_to(ROOT))
        plt.close("all")
    plt.show = save_and_close

    ns = {"__name__": "__main__"}
    root = find_euroc_root()
    if root:
        ns["data_root"] = root                        # Cell 1 reads this if it exists
        print("EuRoC data folder:", root)

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("cell1.py", "cell2.py", "cell3.py"):
            print("\n" + "=" * 30, name, "=" * 30)
            code = patch((SRC / name).read_text(encoding="utf-8"), args.source, args.test, tmp)
            exec(compile(code, name, "exec"), ns)
    save_and_close()                                  # anything still open
    print("\nDone. Figures + log are in outputs/")


if __name__ == "__main__":
    main()
