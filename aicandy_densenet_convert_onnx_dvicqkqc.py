"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
from aicandy_model_src_bpvbytql.aicandy_model_densenet_ekktouop import DenseNet

# python aicandy_convert_to_onnx_dvicqkqc.py --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth --output_path aicandy_model_out_ddmalncc/aicandy_model_onnx_ptspsbyh.onnx

def convert_to_onnx(model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=12, num_classes=2)  # Change num_classes accordingly
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 160, 160, device=device)
    torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the ONNX model')

    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.output_path)
