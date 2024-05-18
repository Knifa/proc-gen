import math

from coloraide import Color as Base
from coloraide.spaces.okhsv import Okhsv
from perlin_noise import PerlinNoise
from PIL import Image, ImageChops, ImageOps


class Color(Base): ...


Color.register(Okhsv())


def add_band(img: Image, threshold: int, band_width: int):
    band_img = Image.new("L", img.size)

    for x in range(img.width):
        for y in range(img.height):
            pixel = img.getpixel((x, y))
            if pixel > threshold - band_width and pixel < threshold + band_width:
                band_img.putpixel((x, y), 255)
            else:
                band_img.putpixel((x, y), 0)

    return band_img


def main(output_i: int = 0):
    print(output_i)

    noise = PerlinNoise(octaves=1, seed=0)
    noise_img = Image.new("L", (256, 512))

    for x in range(noise_img.width):
        for y in range(noise_img.height):
            n = noise((x / noise_img.width, y / noise_img.width))
            n = n / 2 + 0.5
            n = math.pow(n, 2)
            n *= 255

            noise_img.putpixel((x, y), int(n))

    noise_img = ImageOps.autocontrast(noise_img)
    noise_img.save("noise.png")

    band_width = 8
    band_start = band_width / 2
    band_end = 255 - band_width / 2
    band_count = 8

    band_range = band_end - band_start

    threshold = band_start

    band_imgs = []
    for i in range(band_count):
        band_img = add_band(noise_img, threshold, band_width)
        band_imgs.append(band_img)
        band_img.save(f"band_{i:04}.png")

        threshold += band_range / (band_count - 1)

    final_image = Image.new("RGB", noise_img.size, (33, 33, 33))
    for i, band_img in enumerate(band_imgs):
        col = Color("okhsv", (i / band_count * 360, 0.8, 1)).convert("srgb")

        band_img_c = Image.new(
            "RGB",
            band_img.size,
            (
                int(col[0] * 255),
                int(col[1] * 255),
                int(col[2] * 255),
            ),
        )

        final_image = ImageChops.composite(band_img_c, final_image, band_img)

    final_image_rgb = final_image.convert("RGB")
    final_image_rgb.save(f"final_{output_i:04}.png")


# pool = multiprocessing.Pool(16)
# pool.map(main, range(5 * 20))

main()
