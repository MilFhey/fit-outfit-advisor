from PIL import Image
import numpy as np


DEFAULT_IMAGE_SIZE = (128, 128)
SIMPLE_CNN_ARCHITECTURE = "simple_cnn"
MOBILENET_V2_ARCHITECTURE = "mobilenet_v2"


def get_image_preprocessing_mode(architecture: str) -> str:
    if architecture == SIMPLE_CNN_ARCHITECTURE:
        return "rescale_1_over_255"
    if architecture == MOBILENET_V2_ARCHITECTURE:
        return "mobilenet_v2_preprocess_input"
    raise ValueError(f"Architecture image inconnue : {architecture}")


def preprocess_image_for_cnn(
    image: Image.Image,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    architecture: str = SIMPLE_CNN_ARCHITECTURE,
) -> np.ndarray:
    """
    Prépare une image pour un futur CNN TensorFlow/Keras.

    Sortie attendue : batch de shape (1, height, width, 3).
    `simple_cnn` applique uniquement /255.
    `mobilenet_v2` applique uniquement preprocess_input MobileNetV2.
    """
    if image is None:
        raise ValueError("Aucune image fournie au preprocessing.")

    image = image.convert("RGB")
    image = image.resize(image_size)
    array = np.asarray(image).astype("float32")

    preprocessing_mode = get_image_preprocessing_mode(architecture)
    if preprocessing_mode == "rescale_1_over_255":
        array = array / 255.0
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        array = preprocess_input(array)

    return np.expand_dims(array, axis=0)
