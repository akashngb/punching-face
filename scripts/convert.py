from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from reconstruction import reconstruct

parser=argparse.ArgumentParser(description="Convert a cropped, surface-aligned Gaussian PLY/SPLAT to an editable mesh JSON.")
parser.add_argument("input",type=Path)
parser.add_argument("output",type=Path)
parser.add_argument("--depth",type=int,choices=[6,7,8],default=8)
parser.add_argument("--min-opacity",type=float,default=.15,help="Discard Gaussian samples below this opacity, from 0 up to 1.")
args=parser.parse_args()
result=reconstruct(args.input.read_bytes(),args.input.suffix.lstrip('.'),args.depth,min_opacity=args.min_opacity)
args.output.write_text(json.dumps(result,allow_nan=False))
print(json.dumps(result['stats']))
