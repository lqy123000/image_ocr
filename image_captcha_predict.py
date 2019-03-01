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

def decode_ctc(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

def decode_cnn(out):
    ret = []
    outlen = np.argmax(out[-1], 1)
    for i in range(out[0].shape[0]):
        onestr = ''
        for j in range(outlen[i]):
            out_best = np.argmax(out[j][i])
            outstr = alphabet[out_best]
            onestr += outstr
        ret.append(onestr)
    return ret

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


class Preditcor(object):
    def __init__(self, model_path, img_w, img_h, mode):
        if mode not in ("CNN", "CTC"):
            raise ValueError("illegal mode")
        self.model = keras.models.load_model(model_path)
        self.img_w = img_w
        self.img_h = img_h
        self.mode = mode

        # a tensorflow backend bug when using flask.
        # ref: https://www.jianshu.com/p/c84ae0527a3f
        # ref: https://github.com/keras-team/keras/issues/2397
        self.model.predict(np.zeros([1, self.img_w, self.img_h, 1]))
        # self.graph = tf.get_default_graph()

    def predict_image(self, image_path):
        """Predict sigle image in file
        """
        img = image.load_img(image_path, color_mode="grayscale", target_size=(self.img_h, self.img_w))
        inputs = image.img_to_array(img)
        inputs = inputs.transpose(1, 0, 2)
        inputs = inputs.astype(np.float32) / 255
        inputs = np.expand_dims(inputs, 0)
        out = self.model.predict(inputs)
        out_best = list(np.argmax(out[0, 2:], 1))
        # print(list(np.max(out[0, 2:], 1)))
        # print(out_best)
        out_best = [k for k, g in itertools.groupby(out_best)]
        # print(out_best)
        outstr = labels_to_text(out_best)
        return outstr

    def predict_generate(self, batch, batch_size, string_len, save=False):
        """Predict generated random images in batch.

        Generated images can be saved to file.
        """
        generator = ImageCaptcha(width=self.img_w, height=self.img_h)
        right_num = 0

        for i in range(batch):
            data = np.ones([batch_size, self.img_w, self.img_h, 1])
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

            out = self.model.predict(data)
            if self.mode == "CTC":
                outstr = decode_ctc(out)
            elif self.mode == "CNN":
                outstr = decode_cnn(out)

            for j in range(batch_size):
                if outstr[j] == word_list[j]:
                    right_num += 1
                else:
                    print("error index: {}, right: {}, predict: {}"
                        .format(i * batch_size + j, word_list[j], outstr[j]))
        print("right/total: {}/{}, accuracy: {}%"
            .format(right_num, batch * batch_size, right_num / (batch * batch_size) *100))

def main():
    img_w = 128
    img_h = 64
    model_path = "./model/image_captcha_cnn.h5"
    predictor = Preditcor(model_path, img_w, img_h, mode="CNN")
    # print(predictor.predict_image(r"E:\1.jpg"))
    predictor.predict_generate(320, 32, 4, False)

if __name__ == '__main__':
    main()
    