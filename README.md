<<<<<<< HEAD
# IMU dead reckoning + ESKF

Runs the three Colab cells (`src/cell1.py`, `cell2.py`, `cell3.py`) as a normal Python project.

## Layout
- `src/` your three cells, unchanged (paste them in as cell1.py, cell2.py, cell3.py)
- `data/phone/` walk.zip, slide.zip, calibration.json
- `data/euroc_small.zip` EuRoC data (only for `--source euroc`; use Git LFS if over 50 MB)
- `outputs/` figures + log.txt written by each run

## Run
    pip install -r requirements.txt
    python run.py --source device --test square
    python run.py --source euroc
=======
# 654168
>>>>>>> c13e025b5519a7582f3b15b2b2062b291b42f626
