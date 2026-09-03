"""Turn a photo or video of a bookshelf into model-ready image tiles.

The single most important thing this module does is *tiling*. A phone photo of a
full bookcase is maybe 4000px wide; the vision API works at about 1568px on the
long edge, so sending the whole frame throws away the resolution that spine text
actually lives in. Cutting the shelf into an overlapping grid sized from the
source resolution, and sending each tile at close to native resolution, is the
difference between "unreadable" and "legible".
"""

import base64
import io
import math
import os
from dataclasses import dataclass

from PIL import Image, ImageOps

from . import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class Tile:
    """One model-ready crop, tagged with where it came from."""

    source: str          # the file this originated in
    index: int           # position within that source, reading order
    total: int           # how many tiles the source produced
    image: Image.Image
    col: int = 0         # grid column, left to right
    row: int = 0         # grid row, top to bottom

    @property
    def label(self) -> str:
        return "{} [r{}c{}]".format(os.path.basename(self.source), self.row + 1,
                                    self.col + 1)

    def as_base64_png(self) -> str:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG", optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def estimated_tokens(self) -> int:
        """Anthropic's documented approximation: width * height / 750."""
        w, h = self.image.size
        return int(w * h / 750)


def _fit_to_max_edge(img: Image.Image) -> Image.Image:
    """Downscale so the long edge is at most MAX_EDGE. Never upscales."""
    w, h = img.size
    longest = max(w, h)
    if longest <= config.MAX_EDGE:
        return img
    scale = config.MAX_EDGE / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _axis_positions(length: int, divisions: int) -> list[tuple[int, int]]:
    """Split `length` into `divisions` overlapping spans covering it exactly."""
    if divisions <= 1:
        return [(0, length)]
    stride = length / (divisions - (divisions - 1) * config.TILE_OVERLAP)
    step = stride * (1 - config.TILE_OVERLAP)
    spans = []
    for i in range(divisions):
        start = int(round(i * step))
        end = int(round(min(length, start + stride)))
        if end - start >= 16:
            spans.append((start, end))
    return spans


def plan_grid(size: tuple[int, int], tile_count: int | None = None) -> tuple[int, int]:
    """Decide how many columns and rows of tiles an image needs.

    Driven by resolution, not aspect ratio. The goal is that no tile has to be
    downscaled much to reach MAX_EDGE, because every pixel thrown away here is
    spine text the model will not be able to read. A 4032x3024 phone photo needs
    a 3x2 grid to keep close to native resolution; deriving the count from the
    aspect ratio instead would call that 4:3 image "square" and send it whole.
    """
    w, h = size
    cols = max(1, math.ceil(w / config.MAX_EDGE))
    rows = max(1, math.ceil(h / config.MAX_EDGE))

    if tile_count is not None:
        # Honour an explicit budget by scaling the grid to fit it, keeping the
        # image's proportions rather than collapsing to a single axis.
        while cols * rows > max(1, tile_count):
            if cols >= rows and cols > 1:
                cols -= 1
            elif rows > 1:
                rows -= 1
            else:
                break

    # Guard against absurd grids on very large scans.
    while cols * rows > 12:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    return cols, rows


def tile_image(img: Image.Image, source: str, tile_count: int | None = None) -> list[Tile]:
    """Cut an image into an overlapping grid of model-ready tiles."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    cols, rows = plan_grid((w, h), tile_count)
    if cols == 1 and rows == 1:
        return [Tile(source=source, index=0, total=1, image=_fit_to_max_edge(img))]

    x_spans = _axis_positions(w, cols)
    y_spans = _axis_positions(h, rows)

    tiles: list[Tile] = []
    total = len(x_spans) * len(y_spans)
    index = 0
    for row, (top, bottom) in enumerate(y_spans):
        for col, (left, right) in enumerate(x_spans):
            crop = img.crop((left, top, right, bottom))
            tiles.append(
                Tile(source=source, index=index, total=total,
                     image=_fit_to_max_edge(crop), col=col, row=row)
            )
            index += 1
    return tiles


def tiles_from_photo(path: str, tile_count: int | None = None) -> list[Tile]:
    with Image.open(path) as img:
        return tile_image(img, source=path, tile_count=tile_count)


def tiles_from_video(path: str, every_seconds: float = 1.5,
                     tile_count: int | None = None) -> list[Tile]:
    """Sample frames from a pan-across-the-shelf video, then tile each frame.

    Requires opencv-python. Video is the more convenient capture story but the
    worse input: phone video is heavily compressed and motion-blurred compared
    with a still, so expect noticeably lower accuracy than photos.
    """
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "Video input needs opencv-python. Install it with:\n"
            "    python -m pip install opencv-python\n"
            "Or just use photos, which read more accurately anyway."
        ) from exc

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * every_seconds)))

    tiles: list[Tile] = []
    frame_no = 0
    kept = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_no % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                label = f"{path}#frame{kept:03d}"
                tiles.extend(tile_image(img, source=label, tile_count=tile_count))
                kept += 1
            frame_no += 1
    finally:
        capture.release()

    if not tiles:
        raise RuntimeError(f"No frames could be read from {path}")
    return tiles


def collect_tiles(paths: list[str], tile_count: int | None = None,
                  every_seconds: float = 1.5) -> list[Tile]:
    """Expand files and directories into tiles, dispatching on file type."""
    expanded: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if os.path.isfile(full):
                    expanded.append(full)
        else:
            expanded.append(path)

    tiles: list[Tile] = []
    for path in expanded:
        suffix = os.path.splitext(path)[1].lower()
        if suffix in IMAGE_SUFFIXES:
            tiles.extend(tiles_from_photo(path, tile_count=tile_count))
        elif suffix in VIDEO_SUFFIXES:
            tiles.extend(tiles_from_video(path, every_seconds=every_seconds,
                                          tile_count=tile_count))
        # Silently skip anything else, so pointing at a messy folder still works.
    return tiles
