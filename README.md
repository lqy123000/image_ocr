# image_ocr
Image OCR in Keras.

## Introduction
This project is inspired by image_ocr.py in keras examples.

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

## Results
Test 10000 random images.
* **digits** test acc: 99.76% for training 15 epochs 
* **English character** test acc: 96.16% for training 16 epochs 

## References
* [keras example "image_ocr.py"](https://github.com/keras-team/keras/blob/master/examples/image_ocr.py)
