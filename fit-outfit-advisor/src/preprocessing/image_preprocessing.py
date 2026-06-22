from PIL import Image
import numpy as np


DEFAULT_IMAGE_SIZE = (128, 128)


def preprocess_image_for_cnn(image: Image.Image, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """
    Prépare une image pour un futur CNN TensorFlow/Keras.

    Sortie attendue : batch de shape (1, height, width, 3), normalisé entre 0 et 1.
    """
    if image is None:
        raise ValueError("Aucune image fournie au preprocessing.")

    image = image.convert("RGB")
    image = image.resize(image_size)
    array = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)
