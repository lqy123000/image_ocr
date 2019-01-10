# -*- coding: utf-8 -*-
'''CNN+GRU+CTC network for optical character recognition of generated characters images.

This source code bases on "image_ocr.py" in keras examples.
Some main changes are listed as follows:
* Generate characters in random rather than loading words from file.
* Change network architecture. BN is added after
Conv2D and one BiGRU is removed, since the origin network
is found hard to converge.
* Reduce epochs since BN accelerates training.
* The function "on_train_begin" seems working in parallel with
"fit_generator", so built word list on constructor.
* Replace cairocffi with PIL. cairocffi relies on GTK which is not easy
to install on windows.
* If font size is too large to paint, try smaller size rather than
throw a exception immediately.
* Rotation range is tuned to avoid string exceeding canvas.
* Support saving "predict_model".

The origin [keras example](https://github.com/keras-team/keras/blob/master/examples/image_ocr.py)
was created by [Mike Henry](https://github.com/mbhenry/)
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import itertools
import codecs
import re
import datetime
import numpy as np
from scipy import ndimage
import pylab
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.layers import Reshape, Lambda
from tensorflow.python.keras.layers import add, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import callbacks


OUTPUT_DIR = './keras/ocr/image_ocr'

# digit classes
# alphabet = u'0123456789 '

# English characters classes
alphabet = u'abcdefghijklmnopqrstuvwxyz '

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
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
    # teach the RNN translational invariance by
    # fitting text box randomly on canvas, with some room to rotate
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
    if rotate:  # avoid string exceeding canvas
        a = image.random_rotation(a, 70 * (np.min([top_left_y, h - box_height - top_left_y])) / w)
    a = speckle(a)

    return a


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(callbacks.Callback):

    def __init__(self, minibatch_size, img_w, img_h, 
                 downsample_factor, val_split, build_word_count,
                 max_string_len=25, mono_fraction=1):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.max_string_len = max_string_len
        self.mono_fraction = mono_fraction
        self.cur_val_index = 0
        self.cur_train_index = 0
        self.build_word_list(build_word_count)
        self.paint_func = lambda text: paint_text(
            text, self.img_w, self.img_h,
            rotate=False, ud=False, multi_fonts=False)

    def get_output_size(self):
        return len(alphabet) + 1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words):
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.Y_data = np.ones([self.num_words, self.max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        def _is_length_of_word_valid(word):
            return (self.max_string_len == -1 or
                    self.max_string_len is None or
                    len(word) <= self.max_string_len)

        # generate characters in random
        for _ in range(int(self.num_words * self.mono_fraction)):
            word_len = np.random.randint(self.max_string_len // 2, self.max_string_len + 1)
            word = ""
            for _ in range(word_len):
                word += alphabet[np.random.randint(0, len(alphabet) - 1)]
            if _is_length_of_word_valid(word):
                tmp_string_list.append(word)

        # generate characters with black in random (seems hard to train)
        for _ in range(self.num_words - int(self.num_words * self.mono_fraction)):
            word_len = np.random.randint(self.max_string_len // 2, self.max_string_len + 1)
            word = ""
            for _ in range(word_len):
                word += alphabet[np.random.randint(0, len(alphabet))]
            if _is_length_of_word_valid(word):
                tmp_string_list.append(word)

        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words'
                          'from supplied monogram and bigram files.')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word)
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 2:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = (
                        self.paint_func(self.X_text[index + i])[0, :, :].T)
                else:
                    X_data[i, 0:self.img_w, :, 0] = (
                        self.paint_func(self.X_text[index + i])[0, :, :].T)
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index,
                                 self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index,
                                 self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 2 <= epoch < 4:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=False, ud=True, multi_fonts=False)
        elif 4 <= epoch < 6:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 6:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=True, ud=True, multi_fonts=True)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs={}):
        weight_file = os.path.join(self.output_dir, 'weights%02d.h5' % (epoch))
        self.model.save_weights(weight_file)
        print("save_weight: ", weight_file)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func,
                           word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel(
                'Truth = \'{}\'\nPredict = \'{}\' ({})'
                .format(word_batch['source_str'][i], res[i], word_batch['source_str'][i] == res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()


def train(run_name, start_epoch, stop_epoch, img_w, build_word_count,
          max_string_len, mono_fraction, save_model_path=None):
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    # GRU output NAN when rnn_size=512 with my GPU, but CPU or rnn_size=256 is ok. 
    # Tensorflow 1.10 appears, but vanishes in 1.12!
    rnn_size = 512
    minibatch_size = 32

    # if start_epoch >= 12:
    #     minibatch_size = 8  # 32 is to large for my poor GPU

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words,
        build_word_count=build_word_count,
        max_string_len=max_string_len,
        mono_fraction=mono_fraction)
    act = 'relu'
    kernel_init = 'he_normal'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer=kernel_init,
                   name='conv1')(input_data)
    inner = BatchNormalization(axis=3, scale=False, name='bn1')(inner)
    inner = Activation(activation=act)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer=kernel_init,
                   name='conv2')(inner)
    inner = BatchNormalization(axis=3, scale=False, name='bn2')(inner)
    inner = Activation(activation=act)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # bidirectional GRU, GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer=kernel_init, name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer=kernel_init,
                 name='gru1_b')(inner)
    gru1_merged = concatenate([gru_1, gru_1b])

    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer=kernel_init,
                  name='dense2')(gru1_merged)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels',
                   shape=[img_gen.max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
        optimizer=sgd,
        metrics=['accuracy'])
    if start_epoch > 0:
        weight_file = os.path.join(
            OUTPUT_DIR,
            os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
        print("load_weight: ", weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    model.fit_generator(
        generator=img_gen.next_train(),
        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
        epochs=stop_epoch,
        validation_data=img_gen.next_val(),
        validation_steps=val_words // minibatch_size,
        callbacks=[viz_cb, img_gen],
        initial_epoch=start_epoch,
        verbose=1)

    if save_model_path:
        predict_model = Model(inputs=input_data, outputs=y_pred)
        predict_model.save(save_model_path)

if __name__ == '__main__':
    RUN_NAME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # RUN_NAME = "EnglishWord_GRU"
    SAVE_MODEL_PATH = "./model/image_ocr_word.h5"
    train(run_name=RUN_NAME,
          start_epoch=0,
          stop_epoch=12,
          img_w=128,
          build_word_count=16000,
          max_string_len=5,
          mono_fraction=1)
    # increase to wider images and start at epoch 12.
    # The learned weights are reloaded
    train(run_name=RUN_NAME,
          start_epoch=12,
          stop_epoch=20,
          img_w=512,
          build_word_count=32000,
          max_string_len=25,
          mono_fraction=1,
          save_model_path=SAVE_MODEL_PATH)
