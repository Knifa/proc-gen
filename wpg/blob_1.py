# %%
import numpy as np
import pyfastnoisesimd as fns
import skimage as ski
from coloraide import Color as Base
from coloraide.spaces.okhsv import Okhsv
from PIL import Image


class Color(Base): ...


Color.register(Okhsv())

# %%

WIDTH = 256
HEIGHT = 512

noise = fns.Noise()
noise.noiseType = fns.NoiseType.Perlin
noise.frequency = 0.0025

noise_img = noise.genAsGrid((HEIGHT, WIDTH))
noise_img = (noise_img + 1) / 2
noise_img_orig = noise_img.copy()

BANDS = 16
BANDS_WIDTH = 0.025
BANDS_START = 0.0 + BANDS_WIDTH / 2
BANDS_END = 1.0 - BANDS_WIDTH / 2
BAND_STEPS = (BANDS_END - BANDS_START) / (BANDS)

band_colors = []
for i in range(BANDS):
    hue = (i / BANDS) * 360
    color = Color("okhsv", [hue, 1, 1])
    band_colors.append(color.convert("srgb"))

# for j in range(10 * 30):
# noise_img = noise_img_orig.copy()
# noise_img = (noise_img + j * (1 / 30) * 0.05) % 1

bands_img = np.zeros((HEIGHT, WIDTH, 3), dtype=noise_img.dtype)
bands_img[:, ::, :] = 0.1


for i in range(BANDS):
    offset = BANDS_START + i * BAND_STEPS
    band_color = np.array(band_colors[i].coords())

    mask = (noise_img >= offset - BANDS_WIDTH / 2) & (
        noise_img <= offset + BANDS_WIDTH / 2
    )

    band_img = np.where(mask, 1, 0)
    bands_img[mask] = band_color

bands_img_rgb = (bands_img * 255).astype(np.uint8)
img = Image.fromarray(bands_img_rgb)
img

# %%

WIDTH = 1920
HEIGHT = 1080


def alpha_composite(over: np.ndarray, under: np.ndarray) -> np.ndarray:
    over_a = over[:, :, 3]
    under_a = under[:, :, 3]

    out_a = over_a + under_a * (1.0 - over_a)
    out_rgb = (
        over[:, :, :3] * over_a[:, :, None]
        + under[:, :, :3] * under_a[:, :, None] * (1.0 - over_a[:, :, None])
    ) / out_a[:, :, None]

    out = np.zeros_like(over)
    out[:, :, :3] = out_rgb
    out[:, :, 3] = out_a

    return out


noise = fns.Noise()
noise.noiseType = fns.NoiseType.SimplexFractal
noise.fractal.octaves = 2
noise.frequency = 0.002

noise_img = noise.genAsGrid((HEIGHT, WIDTH))
noise_img = (noise_img + 1) / 2
noise_img = ski.exposure.rescale_intensity(noise_img)
noise_img = noise_img**1.0

BANDS = 4
BANDS_MID = 0.5
BANDS_SPREAD = 0.15
BANDS_START = BANDS_MID - BANDS_SPREAD / 2
BANDS_END = BANDS_MID + BANDS_SPREAD / 2
BAND_STEPS = (BANDS_END - BANDS_START) / (BANDS - 1)

band_colors = []
band_color_offset = np.random.random() * 360
band_color_range = np.random.random() * 360

band_color_offset = 0
band_color_range = 360

for i in range(BANDS):
    hue = band_color_offset + (i / BANDS) * band_color_range
    color = Color("okhsv", [hue, 0.5, 1.0])
    # color = Color("srgb", [1.0, 1.0, 1.0])
    band_colors.append(color.convert("srgb"))

bands_img = np.zeros((HEIGHT, WIDTH, 4), dtype=noise_img.dtype)

grad_a = Color("okhsv", [band_color_offset + 0.0, 0.75, 0.2])
grad_b = Color("okhsv", [band_color_offset + band_color_range, 0.75, 0.2])
grad_interp = Color.interpolate([grad_a, grad_b], space="lch")

grad_ys = [
    (grad_interp(y / bands_img.shape[0])).convert("srgb").coords()
    for y in range(bands_img.shape[0])
]
grad_ys = np.array(grad_ys)
bands_img[:, :, :3] = grad_ys[:, None, :]

bands_img[:] = 0.1
bands_img[:, :, 3] = 1.0

for i in range(BANDS):
    band_color = np.array(band_colors[i].coords() + [1.0])

    offset = BANDS_START + i * BAND_STEPS
    threshold_img = np.where(noise_img >= offset, 1.0, 0.0)

    edge_img = threshold_img
    # edge_img = ski.filters.gaussian(edge_img, sigma=1.0)
    # edge_img = ski.filters.sobel(edge_img)
    # edge_img = ski.filters.difference_of_gaussians(edge_img, low_sigma=1.0, high_sigma=8.0)
    # edge_img = ski.exposure.rescale_intensity(edge_img)
    edge_img = ski.feature.canny(edge_img)

    edge_img = ski.morphology.binary_dilation(
        edge_img, footprint=ski.morphology.disk(1)
    )

    # anti-alias
    aa_img = ski.filters.gaussian(edge_img, sigma=1.0)
    edge_img = edge_img + aa_img * 10.0
    edge_img = np.clip(edge_img, 0.0, 1.0)

    edge_rgb = np.stack([edge_img] * 4, axis=-1)
    edge_rgb = edge_rgb * band_color
    edge_rgb[:, :, :3] = band_color[:3]
    bands_img = alpha_composite(edge_rgb, bands_img)

# bands_img = ski.filters.unsharp_mask(
#     bands_img[:, :, :3],
#     radius=4.0,
#     amount=1.0,
# )

bands_img = np.clip(bands_img, 0.0, 1.0)

bands_img_rgb = (bands_img * 255).astype(np.uint8)
img = Image.fromarray(bands_img_rgb)
img
# %%
