import numpy as np
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--qp', nargs='+', type=int, required=True)
parser.add_argument('--cfmt', nargs='+', type=int, required=True)
parser.add_argument('--output_folder', type=str, required=True)
args = parser.parse_args()

dataset = list(Path(args.dataset).glob('**/*.png'))
bpps = {}
target_dir = lambda qp, cfmt: f'{args.output_folder}/qp{qp}_cfmt{cfmt}'

for img_name in tqdm(dataset):
    img_name = str(img_name)
    for qp in args.qp:
        for cfmt in args.cfmt:
            bpps[(qp, cfmt)] = []
            Path(img_name.replace(args.dataset, target_dir(qp, cfmt))).parent.mkdir(parents=True, exist_ok=True)
            target_name = img_name.replace('.png', '.bpg').replace(args.dataset,
                                                                   target_dir(qp, cfmt))
            subprocess.run(f'bpgenc -q {qp} -f {cfmt} -o {target_name} {img_name}', shell=True)
            subprocess.run(f'bpgdec -o {target_name.replace(".bpg", ".png")} {target_name}', shell=True)
