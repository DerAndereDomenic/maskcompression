import torch
import maskcompression
import random


def create_mask():
    width = 1920
    height = 1080
    min_size = 20
    max_size = 200
    mask = torch.zeros((height, width), dtype=torch.float32, device="cuda:0")

    num_boxes = 10
    for _ in range(num_boxes):
        # Randomly choose the top-left corner of the box
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)

        # Randomly determine the width and height of the box
        box_width = random.randint(min_size, max_size)
        box_height = random.randint(min_size, max_size)

        # Ensure the box stays within the image bounds
        x2 = min(x1 + box_width, width)
        y2 = min(y1 + box_height, height)

        mask[y1:y2, x1:x2] = 1.0

    return mask


def test_compress_empty():
    empty = torch.zeros((300, 100), device="cuda:0", dtype=torch.uint8)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 0
    assert compressed[0][1] == empty.numel()


def test_compress_full():
    empty = torch.full((300, 100), 255, device="cuda:0", dtype=torch.uint8)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 1
    assert compressed[0][1] == empty.numel()


def test_compress_full_float():
    empty = torch.full((300, 100), 1.0, device="cuda:0", dtype=torch.float32)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 1
    assert compressed[0][1] == empty.numel()


def test_roundtrip():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1])
    )

    assert torch.allclose(decompressed[0], mask)


def test_roundtrip_batch():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2])
    )

    assert torch.allclose(decompressed, masks)
