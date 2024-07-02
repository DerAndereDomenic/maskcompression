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

def compress_mask(mask : torch.Tensor) -> torch.Tensor:
    compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32)
    return compressed

mask = cv2.imread("C:/Users/zingsheim/Documents/Repositories/VCI_data/Test2/mask_00_15.png", 0)

mask = torch.from_numpy(mask)

mask[(mask > 0) & (mask < 255)] = 255

compressed = torch.unique_consecutive(mask.flatten(), return_counts=True)[1].to(torch.int32).to("cuda")

print(compressed)

# warmup
for i in range(10):
    decompressed = maskcompression.decompress(compressed, (mask.shape[0], mask.shape[1]))

n = 40

start = time.time()
for i in range(n):
    decompressed = maskcompression.decompress(compressed, (mask.shape[0], mask.shape[1]))
end = time.time()

print(f"{(end - start) * 1000.} ms")

decompressed = decompressed.cpu()

plt.imshow(mask)
plt.show()

plt.imshow(decompressed)
plt.show()

difference = mask.to(torch.float32) / 255. - decompressed

plt.imshow(difference.cpu())
plt.show()