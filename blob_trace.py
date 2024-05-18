# %%
import numpy as np
import pyfastnoisesimd as fns
import scipy as sp
import skimage as ski
from PIL import Image, ImageDraw

from IPython.display import display

# %%

WIDTH = 512
HEIGHT = 512

SHAPE_L = (HEIGHT, WIDTH)
SHAPE_RGB = (HEIGHT, WIDTH, 3)

noise = fns.Noise()
noise.noiseType = fns.NoiseType.SimplexFractal
noise.frequency = 0.005
noise.fractal.octaves = 2

noise_img = noise.genAsGrid(SHAPE_L)
noise_img = (noise_img + 1) / 2
noise_img = ski.exposure.rescale_intensity(noise_img)

BANDS = 4
BANDS_MID = 0.5
BANDS_SPREAD = 0.25
BANDS_START = BANDS_MID - BANDS_SPREAD / 2
BANDS_END = BANDS_MID + BANDS_SPREAD / 2
BAND_STEPS = (BANDS_END - BANDS_START) / (BANDS - 1)

img = Image.new("RGB", (WIDTH * 8, HEIGHT * 8), "black")
draw = ImageDraw.Draw(img)

for i in range(BANDS):
    offset = BANDS_START + i * BAND_STEPS
    threshold_img = np.where(noise_img >= offset, 1.0, 0.0)
    contours = ski.measure.find_contours(threshold_img, 0.8)

    for contour in contours:
        # contour = ski.measure.subdivide_polygon(contour, degree=5, preserve_ends=True)
        draw.line(
            [(x * 8, y * 8) for y, x in contour],
            fill="red",
            width=4 * 8,
            joint="curve",
        )

img = img.resize(SHAPE_L, Image.BILINEAR)
display(img)

# %%
