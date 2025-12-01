import torch
import argparse
from MiniNet import MiniNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", type=str, required=True)
    parser.add_argument("--out", type=str, default="MiniNet.onnx")
    args = parser.parse_args()

    model = MiniNet()
    state = torch.load(args.model_pth, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1,1,28,28)

    torch.onnx.export(
        model, dummy, args.out,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["logits"]
    )
    print("Saved ONNX →", args.out)

if __name__ == "__main__":
    main()
