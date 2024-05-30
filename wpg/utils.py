import collections.abc
import datetime
from typing import Generator, Iterable, Iterator, overload

import numpy as np
import skimage as ski

PHI = 1.618033988749895
PHI_A = 1 / PHI
PHI_B = 1 - PHI_A


class RingList[T](collections.abc.Sequence):
    _items: list[T]

    def __init__(self, items: Iterable[T]) -> None:
        self._items = list(items)

    def index(self, item: T) -> int:
        return self._items.index(item)

    def count(self, item: T) -> int:
        return self._items.count(item)

    def __contains__(self, item: T) -> bool:
        return item in self._items

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> list[T]: ...
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key % len(self._items)]
        elif isinstance(key, slice):
            return [
                self._items[i % len(self._items)]
                for i in range(
                    key.start or 0, key.stop or len(self._items), key.step or 1
                )
            ]
        else:
            raise TypeError

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._items)


def contour_path_from_threshold_img(
    threshold_img: np.ndarray,
    complete_paths: bool = True,
) -> Generator[np.ndarray, None, None]:
    """
    :param complete_paths: If True, generate complete paths by add border
        to image.
    """
    contour_img = threshold_img.copy()

    if complete_paths:
        # Add black border around edge, make contour generator give complete paths.
        contour_img[0:1, :] = 0
        contour_img[-1:, :] = 0
        contour_img[:, 0:1] = 0
        contour_img[:, -1:] = 0

    # Label islands so can get each island seperate.
    contour_img_labelled = ski.measure.label(contour_img)
    assert isinstance(contour_img_labelled, np.ndarray)
    contour_img_labels = np.unique(contour_img_labelled)

    for label in contour_img_labels:
        label_mask = contour_img_labelled == label
        contour_img_masked = contour_img * label_mask
        contours = ski.measure.find_contours(contour_img_masked)

        for points in contours:
            points = ski.measure.approximate_polygon(points, 1.0)
            yield points


def compute_beizer_points(points):
    bezier_points = []
    n = len(points)

    for i in range(n):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0

        bezier_points.append((p1, c1, c2, p2))

    return bezier_points


def date_as_seed() -> int:
    """Get the current date as a seed."""
    now = datetime.datetime.now()
    return int(now.strftime("%Y%m%d"))
