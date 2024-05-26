import dataclasses
from typing import Generator, Protocol, Sequence

import cairo
import numpy as np
import pyfastnoisesimd as fns
import skimage as ski

from wpg.utils import (
    RingList,
    contour_path_from_threshold_img,
    date_as_seed,
)


@dataclasses.dataclass
class NoiseSettings:
    type_: fns.NoiseType = fns.NoiseType.SimplexFractal
    frequency: float = 0.001

    fractal_octaves: int = 2
    fractal_lacunarity: float = 2.0
    fractal_gain: float = 0.5

    def get(self, shape: tuple[int, int], seed: int | None) -> np.ndarray:
        noise = fns.Noise(seed=seed)  # type: ignore
        noise.noiseType = self.type_
        noise.frequency = self.frequency

        noise.fractal.octaves = self.fractal_octaves
        noise.fractal.lacunarity = self.fractal_lacunarity
        noise.fractal.gain = self.fractal_gain

        noise_img = noise.genAsGrid(shape)
        noise_img = (noise_img + 1) / 2
        noise_img = ski.exposure.rescale_intensity(noise_img)

        return noise_img


@dataclasses.dataclass
class BandsSettings:
    count: int = 6
    mid: float = 0.5
    spread: float = 0.05
    thickness: float = 0.75

    @property
    def start(self) -> float:
        return self.mid - self.spread

    @property
    def end(self) -> float:
        return self.mid + self.spread

    @property
    def distance(self) -> float:
        return (self.end - self.start) / (self.count - 1)

    def offsets(self) -> Generator[float, None, None]:
        for i in range(self.count):
            yield self.start + i * self.distance

    def offsets_thickened(self) -> Generator[tuple[float, float, float], None, None]:
        for o in self.offsets():
            yield (
                o - self.distance * self.thickness / 2.0,
                o + self.distance * self.thickness / 2.0,
                o,
            )

    def norm(self) -> Generator[float, None, None]:
        for i in range(self.count):
            yield i / (self.count - 1)


type ColorValue = tuple[float, float, float, float]
type ColorSequence = Sequence[ColorValue]


class BandColorCallable(Protocol):
    def __call__(self, *, index: int, index_norm: float, count: int) -> ColorValue: ...


@dataclasses.dataclass
class StyleSettings:
    fill: bool = False

    line_width: float = 8.0
    dash: list[float] = dataclasses.field(default_factory=lambda: [])

    background_color: ColorValue = (0.0, 0.0, 0.0, 0.0)
    band_colors: ColorSequence | BandColorCallable = dataclasses.field(
        default_factory=lambda: [(1.0, 1.0, 1.0, 1.0)]
    )


@dataclasses.dataclass
class ContourSettings:
    size: int = 2048
    output: str = "output.png"
    seed: int | None = date_as_seed()

    noise: NoiseSettings = dataclasses.field(default_factory=NoiseSettings)
    bands: BandsSettings = dataclasses.field(default_factory=BandsSettings)
    style: StyleSettings = dataclasses.field(default_factory=StyleSettings)

    padding_pct: float = 0.0

    debug_show_points: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def shape_np(self) -> np.ndarray:
        return np.array(self.shape)


class ContourGenerator:
    settings: ContourSettings

    surface: cairo.ImageSurface
    cr: cairo.Context

    band_colors: Sequence[ColorValue]

    def __init__(self, settings: ContourSettings):
        self.settings = settings

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *settings.shape)
        self.cr = cairo.Context(self.surface)
        self.cr.set_antialias(cairo.ANTIALIAS_BEST)

        band_colors: ColorSequence
        if callable(settings.style.band_colors):
            band_colors = [
                settings.style.band_colors(
                    index=i, index_norm=n, count=settings.bands.count
                )
                for i, n in enumerate(settings.bands.norm())
            ]
        else:
            band_colors = list(settings.style.band_colors)
        self.band_colors = RingList(band_colors)

    def generate(self):
        settings = self.settings
        cr = self.cr

        cr.set_source_rgba(*settings.style.background_color)
        cr.paint()

        cr.translate(*settings.shape_np / 2)
        cr.scale(1 + settings.padding_pct, 1 + settings.padding_pct)
        cr.translate(*-settings.shape_np / 2)

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        if settings.style.fill:
            cr.set_line_width(1.0)
        else:
            cr.set_line_width(settings.style.line_width)
        cr.set_dash([d * settings.style.line_width for d in settings.style.dash])

        noise_img = settings.noise.get(settings.shape, settings.seed)

        for i, offsets in enumerate(settings.bands.offsets_thickened()):
            cr.set_source_rgba(*self.band_colors[i])

            threshold_masks = (
                [noise_img >= offsets[0], noise_img <= offsets[1]]
                if settings.style.fill
                else [noise_img > offsets[2]]
            )

            threshold_img = np.where(np.all(threshold_masks, axis=0), 1.0, 0.0)
            contours = list(contour_path_from_threshold_img(threshold_img))

            for points in contours:
                if len(points) < 2:
                    continue

                cr.move_to(*points[0])
                for p in points[1:]:
                    cr.line_to(*p)
                cr.close_path()

            if settings.style.fill:
                # Stroke for fill also to prevent gaps.
                cr.fill_preserve()
                cr.save()
                cr.set_line_width(1.0)
                cr.stroke()
                cr.restore()
            else:
                cr.stroke()

            if settings.debug_show_points:
                cr.save()
                cr.set_source_rgba(1.0, 0.0, 0.0, 1.0)
                for points in contours:
                    for p in points:
                        cr.arc(p[0], p[1], 2, 0, 2 * np.pi)
                        cr.fill()
                cr.restore()

    def save(self):
        self.surface.write_to_png(self.settings.output)