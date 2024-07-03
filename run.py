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
import matplotlib.pyplot as plt
import time
import os


compressed_masks = []
masks = []
dir = "C:/Users/zingsheim/Documents/Repositories/VCI_data/dance_vci_simulator_v3/frame_00015/mask"
for filename in os.listdir(dir):
    mask = cv2.imread(os.path.join(dir, filename), 0)
    mask = torch.from_numpy(mask)
    mask[(mask > 0) & (mask < 255)] = 255

    compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32).to("cuda")
    compressed.unsqueeze_(0)

    compressed_masks.append(compressed)
    masks.append(mask.to("cuda").to(torch.float32) / 255.)


# warmup
for i in range(10):
    decompressed = maskcompression.decompress(compressed_masks[i], (masks[i].shape[0], masks[i].shape[1]))

decompressed_masks = []
start = time.time()
for compressed in compressed_masks:
    decompressed = maskcompression.decompress(compressed, (masks[0].shape[0], masks[0].shape[1]))
    decompressed_masks.append(decompressed)
end = time.time()

print(f"Single: {(end - start) * 1000.} ms")

total_error = 0.0

for mask1, mask2 in zip(masks, decompressed_masks):
    total_error += torch.mean((mask1 - mask2.squeeze_(0))**2)

print("Reconstruction error:", total_error.item())