import pandas as pd
import mlflow
import os
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import ModelTrainerConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


class ModelTrainer: 
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.model = None

    def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
        model = Sequential()

        input_shape = (48, 48, 1)   # Explicit tuple, not an object

        # Block 1
        model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Block 2
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Block 3
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Dense Layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))  # 7 classes (happy, disgust, etc.)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.AUC(name="auc")]
        )
        return model
    

    def train_model(self):
        
        with mlflow.start_run():
            
            
            model = self.build_emotion_model()
            logger.info("Model architecture created.")
            history = model.fit(
            self.train_data,  
            validation_data = self.val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
            )

            # Save model after training
            model_path = os.path.join(self.config.root_dir, "emotion_model.h5")
            model.save(model_path)
            logger.info(f"Model saved at: {model_path}")

            mlflow.log_param("epochs", self.config.epochs)
            mlflow.log_param("batch_size", self.config.batch_size)
            mlflow.log_param("learning_rate", 0.0005)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("loss_function", "categorical_crossentropy")

            mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
            mlflow.log_metric("Precision", history.history['precision'][-1])
            mlflow.log_metric("recall", history.history['recall'][-1])
            mlflow.log_metric("auc", history.history['auc'][-1])

        
    def load_training_data(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        self.train_data = datagen.flow_from_directory(
            self.config.train_data_path,
            target_size=(48,48),
            color_mode="grayscale",
            class_mode="categorical",
            batch_size=self.config.batch_size,
            subset="training"
        )
        self.val_data = datagen.flow_from_directory(
            self.config.test_data_path,
            target_size=(48,48),
            color_mode="grayscale",
            class_mode="categorical",
            batch_size=self.config.batch_size,
            subset="validation"
        )

