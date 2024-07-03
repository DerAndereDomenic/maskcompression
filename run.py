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
import numpy as np


compressed_masks = []
masks = []
jpeg_sizes = []
compressed_sizes = []
dir = "C:/Users/zingsheim/Documents/Repositories/VCI_data/dance_vci_simulator_v3/frame_00015/mask"
for filename in os.listdir(dir):
    mask = cv2.imread(os.path.join(dir, filename), 0)
    mask = torch.from_numpy(mask)
    mask[(mask > 0) & (mask < 255)] = 255

    compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32).to("cuda")

    compressed_masks.append(compressed)
    masks.append(mask.to("cuda").to(torch.float32) / 255.)

    jpeg_sizes.append(os.stat(os.path.join(dir, filename)).st_size)
    compressed_sizes.append(compressed.shape[0] * 4)

masks = torch.stack(masks)

# warmup
for i in range(10):
    decompressed = maskcompression.decompress([torch.zeros_like(compressed) for compressed in compressed_masks], (mask.shape[0], mask.shape[1]))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
decompressed_masks = maskcompression.decompress(compressed_masks, (mask.shape[0], mask.shape[1]))
end.record()

torch.cuda.synchronize()
print("Elapsed time:", start.elapsed_time(end), "ms")


total_error = torch.sum((decompressed_masks - masks)**2)

print("Reconstruction error:", total_error.item())
print("Mean compression ratio:", np.mean(np.array(compressed_sizes) / np.array(jpeg_sizes)))