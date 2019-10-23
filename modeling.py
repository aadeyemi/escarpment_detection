import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, concatenate, MaxPooling2D
from keras.layers import Conv2DTranspose, BatchNormalization
from keras.layers import Activation, Add, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)))
# K.tensorflow_backend._get_available_gpus()


def mean_iou(y_true, y_pred):
    """
    compute mean IOU metric
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = K.tf.to_int32(y_pred > t)
        score, up_opt = K.tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(K.tf.local_variables_initializer())
        with K.tf.control_dependencies([up_opt]):
            score = K.tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def residual_block(inp, num_filters=16):
    """
    resnet block to be inserted within larger unet
    """
    x = Activation('relu')(inp)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inp])
    return x


def conv_segment(inp, num_filters, kernel_size, pool_size, dropout_ratio=0.5):
    """
    convolution branch of u-net
    """
    c = Conv2D(num_filters, kernel_size, activation=None, padding="same")(inp)
    c = residual_block(c, num_filters)
    c = Activation('relu')(c)
    p = MaxPooling2D(pool_size)(c)
    p = Dropout(dropout_ratio)(p)
    return p, c


def deconv_segment(c1, c0, num_filters, deconv_kernel_size,
                   conv_kernel_size, dropout_ratio=0.5):
    """
    convolution-transpose branch of u-net
    """
    u = Conv2DTranspose(num_filters, deconv_kernel_size,
                        strides=(2, 2), padding="same")(c1)
    u = concatenate([u, c0])
    u = Dropout(dropout_ratio)(u)
    u = Conv2D(num_filters, conv_kernel_size,
               activation=None, padding="same")(u)
    u = residual_block(u, num_filters)
    u = Activation('relu')(u)
    return u


def build_model(N=128, channels=1):
    """
    build unet (with resnet blocks) model
    """
    inp = Input((N, N, channels))
    start_neurons = 16

    p1, c1 = conv_segment(inp, start_neurons*1, (3, 3), (2, 2), 0.25)
    p2, c2 = conv_segment(p1, start_neurons*2, (3, 3), (2, 2), 0.50)
    p3, c3 = conv_segment(p2, start_neurons*4, (3, 3), (2, 2), 0.50)
    p4, c4 = conv_segment(p3, start_neurons*8, (3, 3), (2, 2), 0.50)
    _, c5 = conv_segment(p4, start_neurons*16, (3, 3), (2, 2), 0.50)

    c6 = deconv_segment(c5, c4, start_neurons*8, (2, 2), (3, 3), 0.50)
    c7 = deconv_segment(c6, c3, start_neurons*4, (2, 2), (3, 3), 0.50)
    c8 = deconv_segment(c7, c2, start_neurons*2, (2, 2), (3, 3), 0.50)
    c9 = deconv_segment(c8, c1, start_neurons*1, (2, 2), (3, 3), 0.50)
    c9 = Dropout(0.25)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inp], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[mean_iou])
    model.summary()
    return model


def model_fit(X_tr, y_tr, X_vd, y_vd, fname='graph.h5', N=128, channels=1):
    """
    fit model to data
    """
    model = build_model(N, channels)
    early_stop = EarlyStopping(patience=5, verbose=1)
    check_point = ModelCheckpoint(fname, verbose=1, save_best_only=True)
    history = model.fit(X_tr, y_tr,
                        epochs=50,
                        validation_data=(X_vd, y_vd),
                        callbacks=[early_stop, check_point],
                        batch_size=4,
                        initial_epoch=0)
    return model, history


def train_model(images, masks, model_fname='model.h5', N=128, channels=1):
    """
    split training data and train model
    """
    X_tr, X_vd, y_tr, y_vd = train_test_split(
        images, masks, random_state=23, test_size=0.2)
    X_tr = [X_tr]
    X_vd = [X_vd]
    model, history = model_fit(X_tr, y_tr, X_vd, y_vd,
                               model_fname, N, channels)
    return model, history
