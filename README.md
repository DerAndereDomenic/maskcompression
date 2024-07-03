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

compressed = maskcompression.compress(masks)
decompressed = maskcompression.decompress(compressed, resolution)
```

## Contact

Domenic Zingsheim - zingsheim@cs.uni-bonn.de