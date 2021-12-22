from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import Model

from tensorflow.keras import Model
pre_trained_model = ResNet50(input_shape=(224,224,3), weights='imagenet')
ResNet = Model(inputs=pre_trained_model.input,outputs=pre_trained_model.layers[-2].output)
#ResNet.summary()

# First time, train without adjusting the ResNet weights to train the final dense layer first
ResNet.trainable =  False
model = tf.keras.Sequential([
                             ResNet,
                             tf.keras.layers.Dense(224, activation='softmax')
                                      ])
model.summary()

model.fit(x_train, y_train, batch_size=24, epochs=1, validation_split=0.2)

ResNet.trainable =  True
model.add_loss(lambda: 0.01*tf.reduce_sum(tf.square(model.layers[-1].kernel)))

model.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=tf.keras.metrics.CategoricalAccuracy())

model.fit(x_train, y_train, batch_size=24, epochs=15, validation_split=0.2)

early_stop = tf.keras.callbacks.EarlyStopping(patience=2)
model.fit(x_train, y_train, batch_size=24, epochs=20, validation_split=0.2, callbacks=[early_stop])

model.evaluate(x_test, y_test)

