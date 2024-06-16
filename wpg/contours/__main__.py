import argparse
import random

import cairo
import pyfastnoisesimd as fns
from coloraide import Color

from ..colors import PALLETES, RgbaTuple, colors_to_rgba
from ..utils import PHI_A, PHI_B
from .generator import (
    BandsSettings,
    ContourGenerator,
    ContourSettings,
    NoiseSettings,
    StyleSettings,
)

# rng_seed = date_as_seed()
rng_seed = None
rng = random.Random(rng_seed)


class Args(argparse.Namespace):
    gen: int
    width: int
    height: int


def gen_0(args: Args) -> None:
    """Inset with close lines."""
    pallete = rng.choice(PALLETES)

    settings = ContourSettings(
        width=args.width,
        height=args.height,
        seed=rng_seed,
        padding_pct=-0.1,
        noise=NoiseSettings(
            fractal_gain=0.5,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            type_=fns.NoiseType.SimplexFractal,
        ),
        bands=BandsSettings(
            count=len(pallete),
            mid=PHI_B,
            spread=0.05,
        ),
        style=StyleSettings(
            fill=False,
            line_width=8.0,
            background_color=(0.1, 0.1, 0.1, 1.0),
            band_colors=colors_to_rgba(pallete),
        ),
        output="out/contour_0.png",
    )

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


def gen_1(args: Args) -> None:
    """Filled waves."""
    pallete = rng.choice(PALLETES)

    darkest_color = pallete[0]
    for c in pallete:
        if c.luminance() < darkest_color.luminance():
            darkest_color = c

    bg_color = Color(darkest_color).convert("okhsv")
    bg_color["v"] = bg_color["v"] * 0.5
    bg_color_tup: RgbaTuple = tuple(bg_color.convert("srgb"))  # type: ignore

    settings = ContourSettings(
        width=args.width,
        height=args.height,
        seed=rng_seed,
        noise=NoiseSettings(
            frequency=4.0,
            type_=fns.NoiseType.Cubic,
        ),
        bands=BandsSettings(
            count=len(pallete),
            mid=rng.choice([0.5, PHI_A, PHI_B]),
            spread=rng.choice([0.1, 0.25, 0.5]),
            # thickness=rng.choice([1.0, PHI_A]),
            thickness=1.0,
        ),
        padding_pct=0.01,
        style=StyleSettings(
            fill=True,
            background_color=bg_color_tup,
            band_colors=colors_to_rgba(pallete),
        ),
        output="out/contour_1.png",
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


def gen_2(args: Args) -> None:
    """Dashed outline."""
    dark = rng.choice([True, False])

    settings = ContourSettings(
        width=args.width,
        height=args.height,
        seed=rng_seed,
        padding_pct=0.01,
        noise=NoiseSettings(
            type_=fns.NoiseType.Simplex,
        ),
        bands=BandsSettings(
            count=8,
            mid=0.5,
            spread=0.4,
        ),
        style=StyleSettings(
            fill=False,
            line_width=2.5,
            dash=[16.0 * PHI_A, 16.0 * PHI_B],
            band_colors=[(1.0, 1.0, 1.0, 0.25) if dark else (0.0, 0.0, 0.0, 0.25)],
            background_color=((0.1, 0.1, 0.1, 1.0) if dark else (0.8, 0.8, 0.8, 1.0)),
        ),
        output="out/contour_2.png",
    )

    generator = ContourGenerator(settings)
    generator.generate()
    generator.save()


def gen_3(args: Args) -> None:
    """A lovely gradient."""
    settings = ContourSettings(
        width=args.width,
        height=args.height,
        seed=rng_seed,
        padding_pct=0.025,
        noise=NoiseSettings(
            frequency=4.0,
            type_=fns.NoiseType.SimplexFractal,
        ),
        bands=BandsSettings(
            count=10,
            mid=0.5,
            spread=0.4,
        ),
        style=StyleSettings(
            fill=False,
            line_width=8.0,
            band_colors=[(0.4, 0.4, 0.4, 0.5)],
        ),
        output="out/contour_3.png",
    )

    generator = ContourGenerator(settings)

    cr = generator.cr
    surf = generator.surface

    t = rng.uniform(0.0, 360.0)
    a = Color("okhsv", [t, 0.7, 0.4])
    b = Color("okhsv", [t + 90.0, 0.7, 0.4])
    a_b = Color.interpolate([a, b], space="oklab", out_space="srgb")
    a_b_y = [a_b(i / settings.height) for i in range(settings.height)]

    surf_data = surf.get_data()
    stride = surf.get_stride()
    for y in range(settings.height):
        col = a_b_y[y]
        for x in range(settings.width):
            i = y * stride + x * 4
            surf_data[i + 0] = int(col[0] * 255)
            surf_data[i + 1] = int(col[1] * 255)
            surf_data[i + 2] = int(col[2] * 255)
            surf_data[i + 3] = 255

    cr.set_operator(cairo.Operator.HSL_LUMINOSITY)
    generator.generate()
    generator.save()


if __name__ == "__main__":
    gens = [gen_0, gen_1, gen_2, gen_3]

    parser = argparse.ArgumentParser(description="Generate contour images.")
    parser.add_argument(
        "--gen",
        type=int,
        help="The generator to run.",
        required=False,
        choices=range(len(gens)),
    )

    parser.add_argument(
        "--width",
        type=int,
        help="The width of the image.",
        required=False,
        default=2048,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="The height of the image.",
        required=False,
        default=2048,
    )

    args = parser.parse_args(namespace=Args())

    if args.gen is not None:
        gens[args.gen](args)
    else:
        for gen in gens:
            gen()
