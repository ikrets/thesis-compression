import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def image_from_batch(batch_img):
    imgs = np.transpose(batch_img, [1, 0, 2, 3])
    imgs = np.reshape(imgs, [imgs.shape[0], -1, imgs.shape[3]])
    return imgs


def draw_text_line(text_batch, background_color, font_color, font_size, cell_dimension):
    line = np.ones((cell_dimension[0], cell_dimension[1] * len(text_batch), 3)) * background_color
    image = Image.fromarray(line.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('visualization/Roboto-Regular.ttf', size=font_size)

    for i, text in enumerate(text_batch):
        draw.text((i * cell_dimension[1], 0), text, fill=font_color, font=font)

    return image


def original_reconstruction_comparison(original_batch, reconstruction_batch, size, alphas=None, lambdas=None):
    float_to_uint8 = lambda X: np.clip(X * 255, 0, 255).astype(np.uint8)

    original_batch = float_to_uint8(original_batch[:size])
    reconstruction_batch = float_to_uint8(reconstruction_batch[:size])
    if alphas is not None and lambdas is not None:
        alphas = alphas[:size]
        lambdas = lambdas[:size]

    elif alphas is not None or lambdas is not None:
        raise ValueError('Provide either both alphas and lambdas, or neither!')

    original_batch_line = image_from_batch(original_batch)
    reconstruction_batch_line = image_from_batch(reconstruction_batch)
    lines = [original_batch_line, reconstruction_batch_line]

    if alphas is not None:
        text_line = draw_text_line([f'λ={lambdas[i]:0.3f}\n'
                                    f'α={alphas[i]:0.3f}' for i in
                                    range(len(alphas))],
                                   background_color=255,
                                   font_color=0,
                                   font_size=8,
                                   cell_dimension=(30, original_batch.shape[2]))
        lines.append(text_line)

    return np.concatenate(lines, axis=0)
