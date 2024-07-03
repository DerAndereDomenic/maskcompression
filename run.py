import torch
import cv2
import maskcompression
import matplotlib.pyplot as plt
import os
import numpy as np


masks = []
jpeg_sizes = []
compressed_sizes = []
dir = "C:/Users/zingsheim/Documents/Repositories/VCI_data/dance_vci_simulator_v3/frame_00015/mask"
for filename in os.listdir(dir):
    if filename[-4:] != "jpeg":
            continue

    mask = cv2.imread(os.path.join(dir, filename), 0)
    mask = torch.from_numpy(mask)
    mask = torch.where(mask > 127, 255, 0)

    masks.append(mask.to("cuda").to(torch.float32) / 255.)

    jpeg_sizes.append(os.stat(os.path.join(dir, filename)).st_size)

masks = torch.stack(masks)

compressed_masks = maskcompression.compress(masks)

for compressed in compressed_masks:
    compressed_sizes.append(len(compressed) * 4)

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

plt.imshow(decompressed_masks[0].cpu())
plt.show()