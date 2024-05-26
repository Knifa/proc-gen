import random

import pyfastnoisesimd as fns
from coloraide import Color
from coloraide.spaces.okhsv import Okhsv

from ..utils import PHI_A, PHI_B, date_as_seed
from .generator import (
    BandsSettings,
    ContourGenerator,
    ContourSettings,
    NoiseSettings,
    StyleSettings,
)

try:
    Color.register(Okhsv())
except ValueError:
    pass


rng = random.Random(date_as_seed())


def gen_1():
    """Inset with close lines."""
    settings = ContourSettings(
        padding_pct=-0.1,
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
        output="out/contour_1.png",
    )

    band_colors = [
        Color("okhsv", [300.0 * i / (settings.bands.count - 1), 0.8, 1.0])
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

    # cr = generator.cr
    # for i in range(3):
    #     cr.set_line_width(settings.style.line_width * 0.5)
    #     cr.set_source_rgba(1.0, 1.0, 1.0, 0.05 * ((3 - i) / 3))

    #     cr.rectangle(
    #         -10 + i * -10,
    #         -10 + i * -10,
    #         settings.size + 20 + i * 20,
    #         settings.size + 20 + i * 20,
    #     )

    #     cr.stroke()

    generator.save()


def gen_2():
    """Filled waves."""
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
        output="out/contour_2.png",
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


def gen_3():
    """Dashed outline."""
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
        output="out/contour_3.png",
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


if __name__ == "__main__":
    gen_1()
    gen_2()
    gen_3()
