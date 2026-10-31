"""Joint text-image vocabulary of the Halcyon checkpoints.

Layout (80,640 ids)::

    [0, 16_384)           text ids (byte-level BPE, 16,384 entries)
    [16_384, 80_384)      image ids: 16_384 + FSQ code, codebook 8*8*8*5*5*5
    [80_384, 80_640)      256 special ids ([SOV], [EOV], [PAD], [EOS] used)

Rows seen in training::

    text-to-image   {caption} [SOV] {1024 image ids} [EOV] [EOS]
    image-to-text   [SOV] {1024 image ids} [EOV] {caption} [EOS]

Image ids are a 32x32 grid in raster order.
"""

TEXT_VOCAB_SIZE = 16_384
IMG_CODEBOOK_SIZE = 64_000
IMG_OFFSET = TEXT_VOCAB_SIZE
SPECIALS_OFFSET = IMG_OFFSET + IMG_CODEBOOK_SIZE
TOTAL_VOCAB_SIZE = SPECIALS_OFFSET + 256

SOV_ID = SPECIALS_OFFSET + 0  # 80_384, start of image
EOV_ID = SPECIALS_OFFSET + 1  # 80_385, end of image
PAD_ID = SPECIALS_OFFSET + 2  # 80_386
EOS_ID = SPECIALS_OFFSET + 3  # 80_387

IMAGE_GRID = 32
IMAGE_TOKENS = IMAGE_GRID * IMAGE_GRID  # 1024

#: Half-open span of token ids an image-generation step may emit.
IMAGE_TOKEN_RANGE = (IMG_OFFSET, SPECIALS_OFFSET)


def max_caption_tokens(seq_length: int) -> int:
    """Longest caption that still leaves room for ``[SOV]`` + codes + ``[EOV]``."""
    return seq_length - IMAGE_TOKENS - 2
