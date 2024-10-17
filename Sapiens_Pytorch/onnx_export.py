import torch
import argparse

from .common import TaskType, download_hf_model
from .segmentation import SapiensSegmentationType
from .normal import SapiensNormalType
from .depth import SapiensDepthType
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type_dict = {
    "seg03b": (SapiensSegmentationType.SEGMENTATION_03B, TaskType.SEG),
    "seg06b": (SapiensSegmentationType.SEGMENTATION_06B, TaskType.SEG),
    "seg1b": (SapiensSegmentationType.SEGMENTATION_1B, TaskType.SEG),
    "normal03b": (SapiensNormalType.NORMAL_03B, TaskType.NORMAL),
    "normal06b": (SapiensNormalType.NORMAL_06B, TaskType.NORMAL),
    "normal1b": (SapiensNormalType.NORMAL_1B, TaskType.NORMAL),
    "normal2b": (SapiensNormalType.NORMAL_2B, TaskType.NORMAL),
    "depth03b": (SapiensDepthType.DEPTH_03B, TaskType.DEPTH),
    "depth06b": (SapiensDepthType.DEPTH_06B, TaskType.DEPTH),
    "depth1b": (SapiensDepthType.DEPTH_1B, TaskType.DEPTH),
    "depth2b": (SapiensDepthType.DEPTH_2B, TaskType.DEPTH)
}


@torch.no_grad()
def export_model(model_name: str, filename: str):
    type, task_type = model_type_dict[model_name]
    path = download_hf_model(type.value, TaskType.SEG)
    model = torch.jit.load(path)
    model = model.eval().to(device).to(torch.float32)
    input = torch.randn(1, 3, 1024, 768, dtype=torch.float32, device=device)  # Only this size seems to work well
    torch.onnx.export(model,
                      input,
                      filename,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=14,
                      input_names=["input"],
                      output_names=["output"])


def get_parser():
    parser = argparse.ArgumentParser(description="Export Sapiens models to ONNX")
    parser.add_argument("model_name", type=str, choices=model_type_dict.keys(), help="Model type to export")
    return parser

#
# if __name__ == "__main__":
#     args = get_parser().parse_args()
#     export_model(args.model, f"{args.model}.onnx")
