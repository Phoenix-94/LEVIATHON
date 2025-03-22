import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load MobileNetV2 as base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dense(4, activation='softmax')  # Change number of classes as per dataset

# Create model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                                   rotation_range=20, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_data = train_datagen.flow_from_directory('dataset', target_size=(224, 224), batch_size=32, subset='training')
val_data = train_datagen.flow_from_directory('dataset', target_size=(224, 224), batch_size=32, subset='validation')

# Train model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("models/marine_species.keras")  # Use .keras instead of .h5

print("Model training complete and saved!")