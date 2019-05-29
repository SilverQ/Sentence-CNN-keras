# CNN-text-classification-keras

It is simplified implementation of [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) in Keras as functional api
The original code is from Bhavesh Vinod Oswal's github (https://github.com/bhaveshoswal/CNN-text-classification-keras)
# Requirements
- [Python 3.6.7](https://www.python.org/)
- [Tensorflow 1.13.1](https://www.tensorflow.org/)

# Modifications
1. Adjusted the codes to Tensorflow 1.13.1
2. Add layers (see the file "Modified Model.txt")
3. Reduced the epochs to 10
4. Change the checkpoint folder

# Traning Command
`python model.py`

# Result
Overfitted so much. But I can't fix it.
- loss: 0.0052
- acc: 0.9995
- val_loss: 1.5170
- val_acc: 0.7459


# For new data
You have to rebuild the vocabulary and then train.

# For Citation
```
@misc{bhaveshoswal,
  author = {Bhavesh Vinod Oswal},
  title = {CNN-text-classification-keras},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bhaveshoswal/CNN-text-classification-keras}},
}
```
