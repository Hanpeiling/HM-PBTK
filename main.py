import os
import sys
from pathlib import Path
import argparse
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
from batch_pipeline.main_pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description='batch prediction')
    parser.add_argument('--workdir', default='.')
    parser.add_argument('--input', default='Batch prediction.xlsx')
    args = parser.parse_args()

    final_output = run_pipeline(work_dir=Path(args.workdir), input_excel=args.input)
    print(f'final_output：{final_output}')


if __name__ == "__main__":
    main()
