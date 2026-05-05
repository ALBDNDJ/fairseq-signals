# main.py
import argparse
from ecg_convert import run_convert          # 你的转换文件名是 ecg_convert.py
from preprocess import run_preprocess
from train import run_train
from eval_draw import run_eval               # 你的评估文件名是 eval_draw.py

def parse_args():
    p = argparse.ArgumentParser("ECG pipeline (Guitu)")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--preprocess", default=True)
    # p.add_argument("--preprocess", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--all", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.all:
        run_convert()
        run_preprocess()
        run_train()
        run_eval()
        return
    if args.convert:
        run_convert()
    if args.preprocess:
        run_preprocess()
    if args.train:
        run_train()
    if args.eval:
        run_eval()

if __name__ == "__main__":
    main()