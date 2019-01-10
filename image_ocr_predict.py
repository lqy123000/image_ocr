'''Predict OCR images using keras model.

'''
import os
import itertools
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

# digit classes
# alphabet = u'0123456789 '

# English characters classes
alphabet = u'abcdefghijklmnopqrstuvwxyz '

OUT_DIR = r"E:\Users\Desktop\download\1"

# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

# creates larger "blotches" of noise
def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck

# paints text on canvas with random rotation and noise
def paint_text(text, w, h, rotate=True, ud=True, multi_fonts=True, save_path=None):
    surface = Image.new('L', (w, h), 255)  # "L" represents gray
    border_w_h = (2, 2)
    box_width = 0
    box_height = 0
    for font_size in range(32, 20, -1):
        if multi_fonts:
            fonts = ["simsun.ttc", "simhei.ttf", "msyh.ttc", "msyhbd.ttc", "msyhl.ttc",
                     "simfang.ttf", "simkai.ttf"]
            font = ImageFont.truetype(np.random.choice(fonts), font_size)
        else:
            font = ImageFont.truetype("msyh.ttc", font_size)
        box_width, box_height = font.getsize(text)
        if box_width <= (w - 2 * border_w_h[0]) and box_height <= (h - border_w_h[1]):
            break
        elif font_size <= 21:
            raise IOError(('Could not fit string into image.'
                           'Max char count is too large for given image width.'))

    max_shift_x = w - box_width - border_w_h[0]
    max_shift_y = h - box_height - border_w_h[1]
    top_left_x = np.random.randint(border_w_h[0], int(max_shift_x) + 1)
    if ud:
        top_left_y = np.random.randint(0, int(max_shift_y) + 1)
    else:
        top_left_y = max_shift_y // 2
    draw = ImageDraw.Draw(surface)
    draw.text((top_left_x, top_left_y), text, fill=0, font=font)

    a = np.array(surface, np.uint8)
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 70 * (np.min([top_left_y, h - box_height - top_left_y])) / w)
    a = speckle(a)

    if save_path:
        save_array = a[0, :, :] * 255
        save_array = save_array.astype(np.uint8)
        save_image = Image.fromarray(save_array)
        save_image.save(save_path)

    return a

# predict sigle image in file
def predict_image(model, image_path, img_w, img_h):
    img = image.load_img(image_path, color_mode="grayscale", target_size=(img_h, img_w))
    inputs = image.img_to_array(img)
    inputs = inputs.transpose(1, 0, 2)
    inputs = inputs.astype(np.float32) / 255
    inputs = np.expand_dims(inputs, 0)
    out = model.predict(inputs)
    out_best = list(np.argmax(out[0, 2:], 1))
    # print(list(np.max(out[0, 2:], 1)))
    # print(out_best)
    out_best = [k for k, g in itertools.groupby(out_best)]
    # print(out_best)
    outstr = labels_to_text(out_best)
    print(outstr)

# Predict generated random images in batch.
# Generated images can be saved to file.
def predict_generate(model, img_w, img_h, batch, batch_size, max_string_len, save=False):
    paint_func = lambda text, save_path: paint_text(
        text, img_w, img_h,
        True, True, True, save_path)
    right_num = 0

    for i in range(batch):
        data = np.ones([batch_size, img_w, img_h, 1])
        word_list = []

        for j in range(batch_size):
            word_len = np.random.randint(3, max_string_len + 1)  # max_string_len // 2
            word = ""
            for k in range(word_len):
                word += alphabet[np.random.randint(0, len(alphabet) - 1)]
            if save:
                save_path = os.path.join(OUT_DIR, "{}.png".format(i * batch_size + j))
            else:
                save_path = None
            data[j, :, :, 0] = paint_func(word, save_path)[0, :, :].T
            word_list.append(word)

        out = model.predict(data)

        for j in range(batch_size):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            if outstr == word_list[j]:
                right_num += 1
            else:
                print("error index: {}, right: {}, predict: {}"
                      .format(i * batch_size + j, word_list[j], outstr))
    print("right/total: {}/{}, accuracy: {}%"
          .format(right_num, batch * batch_size, right_num / (batch * batch_size) *100))

def main():
    img_w = 512
    img_h = 64
    model = keras.models.load_model("./model/image_ocr_word.h5")
    # model.summary()
    # predict_image(model, os.path.join(OUT_DIR, "1.png"), img_w, img_h)
    predict_generate(model, img_w, img_h, 32, 32, 25, False)

if __name__ == '__main__':
    main()
    