from libs import *
from data_processing import *


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


def create_model(input_shape, n_classes, head_size, num_heads, ff_dim,
                num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(n_classes, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def train_model(X_tr, y_tr, input_shape, n_classes, name_saved='../models/transformer.keras'):
    model = create_model(
        input_shape, n_classes,
        head_size=64, num_heads=4,
        ff_dim=4, num_transformer_blocks=4,
        mlp_units=[64, 32],
        mlp_dropout=0.2,
        dropout=0.25,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    print(model.summary())

    callbacks = [EarlyStopping( 
        monitor='val_loss', min_delta=5e-3, 
        patience=10, verbose=0, mode='min',
        restore_best_weights=True
        )
    ]

    model.fit(
        X_tr, y_tr,
        validation_split=0.2, epochs=10,
        batch_size=128, callbacks=callbacks,
    )

    model.save(name_saved)


if __name__ == '__main__':
    train, test = load_data(PATH_TRAIN, PATH_TEST)

    scaler = get_scaling(train, features, PATH_SCALER)
    print('x_train')
    print(train.shape)
    X_train = data_transform(
        train, features, label,
        scaler, period, is_label=False
    )
    print('y_train')
    y_train = data_transform(
        train, features, label,
        scaler, period, is_label=True
    )
    print('x_test')
    print(test.shape)
    X_test = data_transform(
        test, features, label,
        scaler, period, is_label=False
    )
    print('y_test')
    y_test = data_transform(
        test, features, label,
        scaler, period, is_label=True
    )

    y_train = y_train[:, 0].reshape((-1,1))

    print(X_train.shape)
    print(y_train.shape)
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]

    train_model(X_train, y_train, input_shape, n_classes)
    model = keras.models.load_model('../models/transformer.keras')
    print(model.evaluate(X_test, y_test, verbose=1))
    preds = np.argmax(model.predict(X_test), axis=1)
    #for idx, preds in zip(X_test, preds):
    print(np.unique(preds))