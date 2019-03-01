# -*- coding: utf-8 -*-
'''CNN network for captcha recognition of generated captcha images.

References:
    - [Multi-digit Number Recognition from Street View Imagery using 
      Deep Convolutional Neural Networks]()
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import re
import datetime
import numpy as np
import pylab
from captcha.image import ImageCaptcha
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.layers import Reshape, Lambda, Flatten
from tensorflow.python.keras.layers import add, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import callbacks

OUTPUT_DIR = './keras/ocr/image_captcha'

alphabet = u'0123456789abcdefghijklmnopqrstuvwxyz'

def paint_captcha(generator, text):
    """Paint captcha via ImageCaptcha libs
    """
    img = generator.generate_image(text)
    img = img.convert("L")
    img_array = np.array(img, np.uint8)
    img_array = img_array.astype(np.float32) / 255
    return img_array


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
        ret.append(alphabet[c])
    return "".join(ret)

# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(callbacks.Callback):

    def __init__(self, minibatch_size, img_w, img_h, 
                 val_split, build_word_count,
                 max_string_len=25):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.val_split = val_split
        self.max_string_len = max_string_len
        self.cur_val_index = 0
        self.cur_train_index = 0
        self.build_word_list(build_word_count)
        self.generator = ImageCaptcha(width=img_w, height=img_h)

    def get_output_size(self):
        return len(alphabet)

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words):
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = []
        self.Y_data = np.ones([self.num_words, self.max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        def _is_length_of_word_valid(word):
            return (self.max_string_len == -1 or
                    self.max_string_len is None or
                    len(word) <= self.max_string_len)

        # generate characters in random
        for _ in range(self.num_words):
            word_len = self.max_string_len
            word = ""
            for _ in range(word_len):
                word += alphabet[np.random.randint(0, len(alphabet))]
            if _is_length_of_word_valid(word):
                self.string_list.append(word)

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
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = [np.ones(size) for i in range(self.max_string_len + 1)]  # N + 1
        source_str = []
        for i in range(size):
            if K.image_data_format() == 'channels_first':
                X_data[i, 0, 0:self.img_w, :] = (
                    paint_captcha(self.generator, self.X_text[index + i]).T)
            else:
                X_data[i, 0:self.img_w, :, 0] = (
                    paint_captcha(self.generator, self.X_text[index + i]).T)
            for j in range(self.max_string_len):
                labels[j][i] = self.Y_data[index + i, j]     
            labels[self.max_string_len][i] = self.Y_len[index + i][0]
            source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = labels
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

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])
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


# this two block function are borrowed from (https://github.com/matterport/Mask_RCNN)
def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = add([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = add([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def train(run_name, start_epoch, stop_epoch, img_w, img_h, build_word_count,
          max_string_len, save_model_path=None):
    # Input Parameters
    words_per_epoch = 96000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    pool_size = 2
    minibatch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        val_split=words_per_epoch - val_words,
        build_word_count=build_word_count,
        max_string_len=max_string_len)
    act = 'relu'
    kernel_init = 'glorot_uniform'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(64, (3, 3), padding='same',
                   activation=None, kernel_initializer=kernel_init,
                   name='conv1')(input_data)
    inner = BatchNormalization(axis=3, scale=False, name='bn1')(inner)
    inner = Activation(activation=act)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    inner = identity_block(inner, 3, [16, 16, 64], stage=2, block='a', train_bn=True)
    inner = conv_block(inner, 3, [32, 32, 128], stage=3, block='a', train_bn=True)
    inner = conv_block(inner, 3, [64, 64, 256], stage=4, block='a', train_bn=True)

    inner = Flatten(name='reshape')(inner)
    inner = Dense(512, activation=act, name='dense1')(inner)

    outputs = []
    for i in range(max_string_len):
        output = Dense(img_gen.get_output_size(), activation='softmax',
                       kernel_initializer=kernel_init, name='output{}'.format(i))(inner)
        outputs.append(output)
    output = Dense(max_string_len + 1, activation='softmax', kernel_initializer=kernel_init,
                   name='output{}'.format(max_string_len))(inner)
    outputs.append(output)

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data],
                  outputs=outputs)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss=['sparse_categorical_crossentropy'] * (max_string_len + 1), 
                  optimizer=sgd,
                  metrics=['accuracy'] * (max_string_len + 1))
    if start_epoch > 0:
        weight_file = os.path.join(
            OUTPUT_DIR,
            os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
        print("load_weight: ", weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], outputs)

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
        model.save(save_model_path)

if __name__ == '__main__':
    RUN_NAME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # RUN_NAME = "2019-02-28_15-54-19"
    SAVE_MODEL_PATH = "./model/image_captcha_cnn.h5"
    train(run_name=RUN_NAME,
          start_epoch=0,
          stop_epoch=10,
          img_w=128,
          img_h=64,
          build_word_count=96000,
          max_string_len=4,
          save_model_path=SAVE_MODEL_PATH)
