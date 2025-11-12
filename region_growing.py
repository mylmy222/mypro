"""Simple region growing algorithm for 2-D grayscale images."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]
Image = Sequence[Sequence[Number]]


@dataclass(frozen=True)
class RegionGrowConfig:
    """Configuration for the region growing algorithm.

    Attributes
    ----------
    tolerance:
        Absolute difference threshold with respect to the average intensity of
        the region. A pixel is added to the region if the difference between the
        pixel value and the current region mean is less than or equal to this
        value.
    connectivity:
        Number of adjacent neighbors to consider when growing. Valid values are
        4 (Von Neumann neighborhood) or 8 (Moore neighborhood).
    max_size:
        Optional limit on the number of pixels that may belong to the region.
        ``None`` indicates that the region can grow to cover the entire image.
    """

    tolerance: float
    connectivity: int = 4
    max_size: int | None = None

    def __post_init__(self) -> None:
        if self.connectivity not in {4, 8}:
            msg = "connectivity must be either 4 or 8"
            raise ValueError(msg)
        if self.tolerance < 0:
            msg = "tolerance must be non-negative"
            raise ValueError(msg)
        if self.max_size is not None and self.max_size <= 0:
            msg = "max_size must be a positive integer or None"
            raise ValueError(msg)


def _neighbors(
    point: Tuple[int, int],
    shape: Tuple[int, int],
    connectivity: int,
) -> Iterable[Tuple[int, int]]:
    """Yield valid neighboring coordinates for ``point``."""

    row, col = point
    max_row, max_col = shape

    offsets: Sequence[Tuple[int, int]]
    if connectivity == 4:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:  # connectivity == 8
        offsets = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < max_row and 0 <= nc < max_col:
            yield nr, nc


def region_growing(
    image: Image,
    seed: Tuple[int, int],
    config: RegionGrowConfig,
) -> List[List[bool]]:
    """Perform region growing on a grayscale image.

    Parameters
    ----------
    image:
        2-D nested list or tuple containing the input image. The structure is
        not modified.
    seed:
        ``(row, col)`` tuple representing the starting pixel for the region.
    config:
        ``RegionGrowConfig`` instance that controls the growth behavior.

    Returns
    -------
    list[list[bool]]
        Boolean mask indicating which pixels belong to the grown region.

    Raises
    ------
    ValueError
        If the seed point is outside of the image bounds or the image is empty.
    """

    if not image or not image[0]:
        msg = "image must be a non-empty 2-D sequence"
        raise ValueError(msg)

    max_row = len(image)
    max_col = len(image[0])

    seed_row, seed_col = seed
    if not (0 <= seed_row < max_row and 0 <= seed_col < max_col):
        msg = "seed point is outside the image bounds"
        raise ValueError(msg)

    mask = [[False for _ in range(max_col)] for _ in range(max_row)]
    queue: deque[Tuple[int, int]] = deque()

    queue.append(seed)
    mask[seed_row][seed_col] = True
    region_sum = float(image[seed_row][seed_col])
    region_count = 1

    while queue:
        current = queue.popleft()
        current_mean = region_sum / region_count

        for neighbor in _neighbors(current, (max_row, max_col), config.connectivity):
            nr, nc = neighbor
            if mask[nr][nc]:
                continue

            pixel_value = float(image[nr][nc])
            if abs(pixel_value - current_mean) <= config.tolerance:
                queue.append(neighbor)
                mask[nr][nc] = True
                region_sum += pixel_value
                region_count += 1

                if config.max_size is not None and region_count >= config.max_size:
                    return mask

    return mask


def grow_from_seed(
    image: Image,
    seed: Tuple[int, int],
    tolerance: float,
    connectivity: int = 4,
    max_size: int | None = None,
) -> List[List[bool]]:
    """Convenience wrapper around :func:`region_growing`.

    Parameters
    ----------
    image:
        Input 2-D sequence.
    seed:
        ``(row, col)`` coordinates of the seed pixel.
    tolerance:
        Allowed deviation from the region mean when adding new pixels.
    connectivity:
        Neighbor definition, either 4 or 8.
    max_size:
        Optional cap on the number of pixels in the region.

    Returns
    -------
    list[list[bool]]
        Boolean mask marking the grown region.
    """

    config = RegionGrowConfig(
        tolerance=tolerance,
        connectivity=connectivity,
        max_size=max_size,
    )
    return region_growing(image, seed, config)


if __name__ == "__main__":
    img = [
        [10, 10, 10, 50, 50],
        [10, 10, 12, 55, 55],
        [10, 11, 15, 60, 60],
        [10, 10, 10, 60, 65],
        [10, 10, 10, 60, 65],
    ]
    seed_point = (1, 1)
    tolerance_value = 5

    mask = grow_from_seed(img, seed_point, tolerance_value, connectivity=4)
    print("Image:")
    for row in img:
        print(row)
    print("Seed:", seed_point)
    print("Mask:")
    for row in mask:
        print([int(value) for value in row])
    region_values = [img[r][c] for r in range(len(img)) for c in range(len(img[0])) if mask[r][c]]
    print("Region average:", sum(region_values) / len(region_values))
