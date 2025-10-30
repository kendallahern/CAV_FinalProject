# export_network.py
# Lightweight utility: copies a model JSON into an "exported" place (keeps consistent naming).
# May not use
# not strictly necessary bc models in week2/models/, but recommended for experimentation

import os
import shutil
import argparse

def export_model(src_path, dst_dir="models_exported"):
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src_path)
    dst = os.path.join(dst_dir, base)
    shutil.copy(src_path, dst)
    print(f"Copied {src_path} -> {dst}")
    return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Path to source model json (e.g. models/tiny_net_weights.json)")
    parser.add_argument("--dst", default="models_exported", help="Destination folder")
    args = parser.parse_args()
    export_model(args.src, args.dst)
