import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
from keras.models import model_from_json

def tensormemory():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

def drowplt(hist, filename):
    plt.subplot(2, 1, 1)
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history['acc'], linewidth=3, label='train')
    plt.plot(hist.history['val_acc'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title("acc")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def save_models(model, filename):
    plot(model, to_file=filename + "_structure.png", show_shapes=True)
    json_string = model.to_json()
    open(filename + '.json', 'w').write(json_string)
    model.save_weights(filename + ".h5")
