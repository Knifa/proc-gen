from coloraide import Color
from coloraide.spaces.okhsv import Okhsv

try:
    Color.register(Okhsv())
except ValueError:
    pass

RgbaTuple = tuple[float, float, float, float]

PALLETE_STRINGS = [
    "#122a34 #214345 #a0a646 #d7bc77 #76604b",
    "#1e2287 #373dcb #3c6bff #3caeff #ffac3c #ffefd9",
    "#2a2430 #45749e #9cb1b6 #edd8bb #ecb2a4 #e45453",
    "#44933e #95c053 #dae346 #e6b917 #ec6e6c #d03d50",
    "#4a294d #cc8d85 #b56431 #a99245 #69822d #0d4841",
    "#9e5773 #b57d97 #d4c3e9 #b4a8f0 #9270ff",
    "#e3813a #ecdd66 #eefff9 #48c8e5 #151521",
    "#efcfbc #f58835 #efbc05 #5aa14b #50b5a5 #302c28",
]


def colors_from_pallete_str(pallete_str: str) -> list[Color]:
    return [Color(hex_str) for hex_str in pallete_str.split()]


PALLETES = [
    *[colors_from_pallete_str(pallete_str) for pallete_str in PALLETE_STRINGS],
    # Rainbow
    [Color("okhsv", [300.0 * i / (8 - 1), 0.8, 1.0]) for i in range(8)],
]


def colors_to_rgba(pallete: list[Color]) -> list[RgbaTuple]:
    return [tuple(color.convert("srgb")) for color in pallete]  # type: ignore
