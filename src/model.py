from tensorflow.keras import layers, models, regularizers, initializers
def build_autoencoder(img_shape, l2_reg=1e-4):
    he_init = initializers.HeNormal()

    input_img = layers.Input(shape=img_shape)

    # ---------------- Encoder ----------------
    x = layers.Conv2D(
        32, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(
        64, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(
        128, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # ---------------- Decoder ----------------
    x = layers.Conv2D(
        128, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(
        64, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(
        32, (3, 3),
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(
        3, (3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    model = models.Model(input_img, decoded, name="conv_autoencoder")
    return model
