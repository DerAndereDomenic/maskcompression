import charonload
import pathlib

VSCODE_STUBS_DIRECTORY = pathlib.Path(__file__).parent / "typings"

charonload.module_config["maskcompression"] = charonload.Config(
    project_directory=pathlib.Path(__file__).parent / "maskcompression",
    build_directory=pathlib.Path(__file__).parent / "build",  # optional
    stubs_directory=VSCODE_STUBS_DIRECTORY,  # optional
)

import torch
import cv2
import maskcompression

def compress_mask(mask : torch.Tensor) -> torch.Tensor:
    compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32)
    return compressed

mask = cv2.imread("C:/Users/zingsheim/Documents/Repositories/VCI_data/Test2/mask_00_15.jpg", 0)

mask = torch.from_numpy(mask)

mask[(mask > 0) & (mask < 255)] = 255

compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32)

print(compressed)