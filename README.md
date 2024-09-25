# maskcompression

A simple library to encode and decode run length compressed binary mask images.

## Installation

Requires Python >= 3.8 and can be installed via:

```
python -m pip instlal --editable .
```

## Quick Start

```python
import maskcompression

masks = generate_masks() # (B,H,W), device=cuda

compressed = maskcompression.compress(masks) # list(torch.Tensor)
decompressed = maskcompression.decompress(compressed, 
                                          resolution, 
                                          vertical_flip=False) # (B,H,W), device=cuda, dtype=float
```

## Contact

Domenic Zingsheim - zingsheim@cs.uni-bonn.de