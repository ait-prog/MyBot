import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing import image, text
from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, MobileNetV2,
    EfficientNetB0, DenseNet121
)
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten,
    Dropout, BatchNormalization, LSTM, GRU,
    Bidirectional, Attention, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import logging
from typing import Tuple, List, Dict, Union, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralNetwork:
    def __init__(self, model_type: str = "custom"):
        """
        Initialize neural network class

        Args:
            model_type: Type of model ("custom", "resnet", "vgg", "inception", "mobilenet", "efficientnet", "densenet")
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.input_shape = None
        self.num_classes = None
        self.class_names = None

        # Create directories for saving
        self.create_directories()

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "models",
            "logs",
            "data",
            "results",
            "checkpoints"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def build_custom_cnn(self, input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
        """
        Build custom CNN model

        Args:
            input_shape: Input data dimensions (height, width, channels)
            num_classes: Number of classes

        Returns:
            Compiled model
        """
        model = models.Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_transformer(self, input_shape: Tuple[int, int], num_classes: int) -> models.Model:
        """
        Build Transformer-based model

        Args:
            input_shape: Input data dimensions (sequence length, feature size)
            num_classes: Number of classes

        Returns:
            Compiled model
        """
        inputs = layers.Input(shape=input_shape)

        # Positional encoding
        x = layers.Dense(256)(inputs)
        x = LayerNormalization()(x)

        # Transformer blocks
        for _ in range(4):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=8, key_dim=32
            )(x, x)
            x = layers.Add()([x, attention_output])
            x = LayerNormalization()(x)

            # Feed-forward network
            ffn = layers.Dense(512, activation='relu')(x)
            ffn = layers.Dense(256)(ffn)
            x = layers.Add()([x, ffn])
            x = LayerNormalization()(x)

        # Output layer
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_lstm(self, input_shape: Tuple[int, int], num_classes: int) -> models.Model:
        """
        Build LSTM model

        Args:
            input_shape: Input data dimensions (sequence length, feature size)
            num_classes: Number of classes

        Returns:
            Compiled model
        """
        model = models.Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_autoencoder(self, input_shape: Tuple[int, int, int]) -> models.Model:
        """
        Build autoencoder model

        Args:
            input_shape: Input data dimensions (height, width, channels)

        Returns:
            Compiled model
        """
        # Encoder
        encoder = models.Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2))
        ])

        # Decoder
        decoder = models.Sequential([
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')
        ])

        # Full model
        model = models.Sequential([encoder, decoder])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )

        return model

    def build_gan(self, input_shape: Tuple[int, int, int], latent_dim: int = 100) -> Tuple[models.Model, models.Model]:
        """
        Build GAN (Generative Adversarial Network)

        Args:
            input_shape: Input data dimensions (height, width, channels)
            latent_dim: Latent space dimension

        Returns:
            Tuple (generator, discriminator)
        """
        # Generator
        generator = models.Sequential([
            Dense(256 * 4 * 4, input_dim=latent_dim),
            layers.Reshape((4, 4, 256)),
            layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
            BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same'),
            BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same'),
            BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(input_shape[2], (4, 4), strides=2, padding='same', activation='tanh')
        ])

        # Discriminator
        discriminator = models.Sequential([
            Conv2D(64, (3, 3), strides=2, padding='same', input_shape=input_shape),
            layers.LeakyReLU(0.2),
            Dropout(0.3),
            Conv2D(128, (3, 3), strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            Dropout(0.3),
            Conv2D(256, (3, 3), strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            Dropout(0.3),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        # Compile discriminator
        discriminator.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Compile GAN
        discriminator.trainable = False
        gan_input = layers.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = models.Model(gan_input, gan_output)
        gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

        return generator, discriminator

    def load_pretrained_model(self, model_name: str, input_shape: Tuple[int, int, int],
                              num_classes: int) -> models.Model:
        """
        Load pretrained model

        Args:
            model_name: Model name
            input_shape: Input data dimensions
            num_classes: Number of classes

        Returns:
            Loaded model
        """
        base_models = {
            "resnet": ResNet50,
            "vgg": VGG16,
            "inception": InceptionV3,
            "mobilenet": MobileNetV2,
            "efficientnet": EfficientNetB0,
            "densenet": DenseNet121
        }

        if model_name not in base_models:
            raise ValueError(f"Unknown model: {model_name}")

        base_model = base_models[model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Freeze base model weights
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            batch_size: int = 32,
            epochs: int = 100,
            callbacks_list: Optional[List[callbacks.Callback]] = None
    ) -> Dict:
        """
        Train model

        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            callbacks_list: List of callbacks

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model first.")

        # Create callbacks
        if callbacks_list is None:
            callbacks_list = [
                ModelCheckpoint(
                    'checkpoints/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                ),
                TensorBoard(
                    log_dir=f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                ),
                CSVLogger('logs/training.log')
            ]

        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )

        return self.history.history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model

        Args:
            x_test: Test data
            y_test: Test labels

        Returns:
            Dictionary with metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model first.")

        return self.model.evaluate(x_test, y_test, return_dict=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            x: Input data

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model first.")

        return self.model.predict(x)

    def save_model(self, filepath: str):
        """
        Save model

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model first.")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model

        Args:
            filepath: Path to model file
        """
        self.model = models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history

        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            raise ValueError("No training history available")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

    def plot_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            save_path: Path to save plot
        """
        cm = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()

    def plot_feature_importance(
            self,
            feature_names: List[str],
            importance_scores: np.ndarray,
            save_path: Optional[str] = None
    ):
        """
        Plot feature importance

        Args:
            feature_names: Feature names
            importance_scores: Importance scores
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def generate_latent_points(self, latent_dim: int, n_samples: int) -> np.ndarray:
        """
        Generate random points in latent space

        Args:
            latent_dim: Latent space dimension
            n_samples: Number of points

        Returns:
            Generated points
        """
        return np.random.randn(n_samples, latent_dim)

    def generate_fake_samples(
            self,
            generator: models.Model,
            latent_dim: int,
            n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fake samples

        Args:
            generator: Generator model
            latent_dim: Latent space dimension
            n_samples: Number of samples

        Returns:
            Tuple (generated images, labels)
        """
        x_input = self.generate_latent_points(latent_dim, n_samples)
        x = generator.predict(x_input)
        y = np.zeros((n_samples, 1))
        return x, y

    def train_gan(
            self,
            dataset: np.ndarray,
            generator: models.Model,
            discriminator: models.Model,
            gan: models.Model,
            latent_dim: int,
            n_epochs: int = 100,
            n_batch: int = 128
    ):
        """
        Train GAN

        Args:
            dataset: Dataset
            generator: Generator model
            discriminator: Discriminator model
            gan: Full GAN model
            latent_dim: Latent space dimension
            n_epochs: Number of epochs
            n_batch: Batch size
        """
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)

        for i in range(n_epochs):
            for j in range(bat_per_epo):
                # Train discriminator
                x_real, y_real = dataset[np.random.randint(0, dataset.shape[0], half_batch)], np.ones((half_batch, 1))
                x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, half_batch)

                d_loss_real = discriminator.train_on_batch(x_real, y_real)
                d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                x_gan = self.generate_latent_points(latent_dim, n_batch)
                y_gan = np.ones((n_batch, 1))
                g_loss = gan.train_on_batch(x_gan, y_gan)

                logger.info(f'Epoch {i + 1}/{n_epochs}, Batch {j + 1}/{bat_per_epo}, '
                            f'D_loss: {d_loss[0]:.4f}, G_loss: {g_loss:.4f}')

    def save_gan_samples(
            self,
            generator: models.Model,
            latent_dim: int,
            n_samples: int = 16,
            save_path: str = 'results/generated_samples.png'
    ):
        """
        Save generated samples

        Args:
            generator: Generator model
            latent_dim: Latent space dimension
            n_samples: Number of samples
            save_path: Path to save samples
        """
        x_input = self.generate_latent_points(latent_dim, n_samples)
        x = generator.predict(x_input)

        plt.figure(figsize=(10, 10))
        for i in range(n_samples):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')
            plt.imshow((x[i] + 1) / 2.0)

        plt.savefig(save_path)
        logger.info(f"Generated samples saved to {save_path}")

    def save_training_config(self, config: Dict, save_path: str = 'results/training_config.json'):
        """
        Save training configuration

        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Training configuration saved to {save_path}")

    def load_training_config(self, load_path: str = 'results/training_config.json') -> Dict:
        """
        Load training configuration

        Args:
            load_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(load_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Training configuration loaded from {load_path}")
        return config

    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image

        Args:
            image_path: Path to image
            target_size: Target size of image

        Returns:
            Preprocessed image
        """
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Augment image

        Args:
            img: Input image

        Returns:
            Augmented image
        """
        datagen = image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        return datagen.random_transform(img)

    def create_data_generator(
            self,
            batch_size: int = 32,
            target_size: Tuple[int, int] = (224, 224),
            data_dir: str = None,
            class_mode: str = 'categorical'
    ) -> image.ImageDataGenerator:
        """
        Create data generator

        Args:
            batch_size: Batch size
            target_size: Target image size
            data_dir: Data directory
            class_mode: Classification mode

        Returns:
            Data generator
        """
        datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        if data_dir:
            train_generator = datagen.flow_from_directory(
                data_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='training'
            )

            validation_generator = datagen.flow_from_directory(
                data_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='validation'
            )

            return train_generator, validation_generator

        return datagen

    def prepare_text_data(
            self,
            texts: List[str],
            max_words: int = 10000,
            max_len: int = 200
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Prepare text data

        Args:
            texts: List of texts
            max_words: Maximum number of words
            max_len: Maximum sequence length

        Returns:
            Tuple (prepared data, token dictionary)
        """
        tokenizer = text.Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        x = text.pad_sequences(sequences, maxlen=max_len)
        return x, tokenizer.word_index

    def create_embedding_matrix(
            self,
            word_index: Dict[str, int],
            embedding_dim: int = 100,
            glove_path: str = None
    ) -> np.ndarray:
        """
        Create embedding matrix

        Args:
            word_index: Token dictionary
            embedding_dim: Embedding dimension
            glove_path: Path to GloVe file

        Returns:
            Embedding matrix
        """
        embeddings_index = {}
        if glove_path:
            with open(glove_path, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def prepare_time_series_data(
            self,
            data: np.ndarray,
            look_back: int = 10,
            look_forward: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data

        Args:
            data: Input data
            look_back: Number of previous points
            look_forward: Number of predicted points

        Returns:
            Tuple (X, y) for training
        """
        X, y = [], []
        for i in range(len(data) - look_back - look_forward + 1):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back:i + look_back + look_forward])
        return np.array(X), np.array(y)

    def normalize_data(
            self,
            data: np.ndarray,
            method: str = 'standard'
    ) -> Tuple[np.ndarray, object]:
        """
        Normalize data

        Args:
            data: Input data
            method: Normalization method ('standard' or 'minmax')

        Returns:
            Tuple (normalized data, scaler)
        """
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        normalized_data = scaler.fit_transform(data)
        return normalized_data, scaler

    def split_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
            test_size: float = 0.2,
            val_size: float = 0.2,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation and test sets

        Args:
            X: Features
            y: Labels
            test_size: Test set size
            val_size: Validation set size
            random_state: Random seed for reproducibility

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_balanced_batch_generator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create balanced batch generator

        Args:
            X: Features
            y: Labels
            batch_size: Batch size

        Returns:
            Batch generator
        """
        n_classes = y.shape[1]
        class_indices = [np.where(y[:, i] == 1)[0] for i in range(n_classes)]
        n_samples_per_class = batch_size // n_classes

        while True:
            batch_indices = []
            for indices in class_indices:
                batch_indices.extend(np.random.choice(indices, n_samples_per_class))
            np.random.shuffle(batch_indices)

            yield X[batch_indices], y[batch_indices]

    def create_mixed_precision_model(self, model: models.Model) -> models.Model:
        """
        Create mixed precision model

        Args:
            model: Base model

        Returns:
            Mixed precision model
        """
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        return model

    def create_distributed_model(
            self,
            model: models.Model,
            strategy: tf.distribute.Strategy
    ) -> models.Model:
        """
        Create distributed model

        Args:
            model: Base model
            strategy: Distribution strategy

        Returns:
            Distributed model
        """
        with strategy.scope():
            model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
        return model
