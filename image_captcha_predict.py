'''Predict captcha images using keras model.

'''
import os
import itertools
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
from captcha.image import ImageCaptcha
from PIL import Image

alphabet = u'0123456789abcdefghijklmnopqrstuvwxyz'

OUT_DIR = r"E:\Users\Desktop\download\1"

def labels_to_text(labels):
    """Reverse translation of numerical classes back to characters
    """
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def paint_captcha(generator, text, save_path=None):
    """Paint captcha via ImageCaptcha libs
    """
    img = generator.generate_image(text)
    img = img.convert("L")
    img_array = np.array(img, np.uint8)
    img_array = img_array.astype(np.float32) / 255

    if save_path:
        img.save(save_path)

    return img_array

def predict_image(model, image_path, img_w, img_h):
    """Predict sigle image in file
    """
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


def predict_generate(model, img_w, img_h, batch, batch_size, string_len, save=False):
    """Predict generated random images in batch.

    Generated images can be saved to file.
    """
    generator = ImageCaptcha(width=img_w, height=img_h)
    right_num = 0

    for i in range(batch):
        data = np.ones([batch_size, img_w, img_h, 1])
        word_list = []

        for j in range(batch_size):
            word_len = string_len
            word = ""
            for k in range(word_len):
                word += alphabet[np.random.randint(0, len(alphabet))]
            if save:
                save_path = os.path.join(OUT_DIR, "{}.png".format(i * batch_size + j))
            else:
                save_path = None
            data[j, :, :, 0] = paint_captcha(generator, word, save_path).T
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
    img_w = 128
    img_h = 64
    model = keras.models.load_model("./model/image_captcha.h5")
    # model.summary()
    # predict_image(model, os.path.join(OUT_DIR, "1.png"), img_w, img_h)
    predict_generate(model, img_w, img_h, 32, 32, 4, False)

if __name__ == '__main__':
    main()
    