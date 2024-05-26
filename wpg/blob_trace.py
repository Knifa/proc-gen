# %%
import dataclasses
import random
from typing import Generator, Protocol, Sequence

import cairo
import numpy as np
import pyfastnoisesimd as fns
import skimage as ski
from coloraide import Color
from coloraide.spaces.okhsv import Okhsv

from wpg.utils import (
    PHI_A,
    PHI_B,
    RingList,
    contour_path_from_threshold_img,
    date_as_seed,
)

try:
    Color.register(Okhsv())
except ValueError:
    pass


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


# %%


def gen_1():
    """Inset with close lines."""
    settings = ContourSettings(
        padding_pct=-0.2,
        noise=NoiseSettings(
            frequency=0.001,
            fractal_gain=0.5,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            type_=fns.NoiseType.SimplexFractal,
        ),
        bands=BandsSettings(
            count=6,
            mid=PHI_B,
            spread=0.05,
        ),
        style=StyleSettings(
            fill=False,
            line_width=8.0,
            background_color=(0.1, 0.1, 0.1, 1.0),
        ),
    )

    band_colors = [
        Color("okhsv", [270.0 * i / (settings.bands.count - 1), 0.8, 1.0])
        for i in range(settings.bands.count)
    ]

    band_colors = Color.steps(
        band_colors,
        space="oklab",
        steps=settings.bands.count,
    )
    settings.style.band_colors = [  # type: ignore
        tuple(color.convert("srgb").coords()) for color in band_colors
    ]

    generator = ContourGenerator(settings)
    generator.generate()

    cr = generator.cr
    for i in range(3):
        cr.set_line_width(settings.style.line_width * 0.5)
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.05 * ((3 - i) / 3))

        cr.rectangle(
            -50 + i * -50,
            -50 + i * -50,
            settings.size + 100 + i * 100,
            settings.size + 100 + i * 100,
        )

        cr.stroke()

    generator.save()


gen_1()

# %%


def gen_2():
    """Filled waves."""

    rng = random.Random(date_as_seed())

    pallete_strs = [
        "#4a294d #cc8d85 #b56431 #a99245 #69822d #0d4841",
        "#2a2430 #45749e #9cb1b6 #edd8bb #ecb2a4 #e45453",
        "#e3813a #ecdd66 #eefff9 #48c8e5 #151521",
        "#efcfbc #f58835 #efbc05 #5aa14b #50b5a5 #302c28",
        "#1e2287 #373dcb #3c6bff #3caeff #ffac3c #ffefd9",
        "#9e5773 #b57d97 #d4c3e9 #b4a8f0 #9270ff",
        "#44933e #95c053 #dae346 #e6b917 #ec6e6c #d03d50",
        "#122a34 #214345 #a0a646 #d7bc77 #76604b",
    ]

    pallete_str = rng.choice(pallete_strs)

    pallete = [Color(hex_str) for hex_str in pallete_str.split()]
    pallete_rgba = [tuple(color.convert("srgb")) for color in pallete]  # type: ignore

    darkest_color = pallete[0]
    for c in pallete:
        if c.luminance() < darkest_color.luminance():
            darkest_color = c

    bg_color = Color(darkest_color).convert("okhsv")
    bg_color["v"] = bg_color["v"] * 0.5
    bg_color = tuple(bg_color.convert("srgb"))  # type: ignore

    settings = ContourSettings(
        noise=NoiseSettings(
            frequency=0.0015,
            type_=fns.NoiseType.Cubic,
        ),
        bands=BandsSettings(
            count=len(pallete),
            mid=rng.choice([0.5, PHI_A, PHI_B]),
            spread=rng.choice([0.1, 0.25, 0.5]),
            thickness=rng.choice([1.0, PHI_A]),
        ),
        padding_pct=0.01,
        style=StyleSettings(
            fill=True,
            background_color=bg_color,
            band_colors=pallete_rgba,  # type: ignore
        ),
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


gen_2()

# %%


def gen_3():
    """Dashed outline."""
    rng = random.Random(date_as_seed())
    dark = rng.choice([True, False])

    settings = ContourSettings(
        padding_pct=0.01,
        noise=NoiseSettings(frequency=0.001, type_=fns.NoiseType.Simplex),
        bands=BandsSettings(count=8, mid=0.5, spread=0.4),
        style=StyleSettings(
            fill=False,
            line_width=2.5,
            dash=[16.0 * PHI_A, 16.0 * PHI_B],
            band_colors=[(1.0, 1.0, 1.0, 0.25) if dark else (0.0, 0.0, 0.0, 0.25)],
            background_color=((0.1, 0.1, 0.1, 1.0) if dark else (0.8, 0.8, 0.8, 1.0)),
        ),
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


gen_3()

# %%
