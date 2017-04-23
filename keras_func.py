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
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(hist.history['loss'], 'b-', marker='.', label='train')
    plt.plot(hist.history['val_loss'], 'g-', marker='.', label='valid')
    plt.grid()
    plt.legend()
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history['acc'], 'b-', marker='.', label='train')
    plt.plot(hist.history['val_acc'], 'g-', marker='.', label='valid')
    plt.grid()
    plt.legend()
    plt.title("acc")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')

    plt.tight_layout()
    plt.savefig(filename)
    return plt
#    plt.show()

def save_models(model, filename):
    plot(model, to_file=filename + "_structure.png", show_shapes=True)
    json_string = model.to_json()
    open(filename + '.json', 'w').write(json_string)
    model.save_weights(filename + ".h5")


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

