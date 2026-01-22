import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize


OUTPUT_EMB_FILE = "embeddings.npy"
OUTPUT_LABELS_FILE = "labels.npy"
NUM_SAMPLES = 1000
TARGET_SIZE = (224, 224)

def load_data(num_samples=NUM_SAMPLES):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    x = x[:num_samples]
    y = y[:num_samples]
    return x, y

def build_feature_extractor():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    )
    base_model.trainable = False
    return base_model

def preprocess_images(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = resize(x, TARGET_SIZE)
    x = preprocess_input(x)
    return x

def extract_embeddings():
    print("[INFO] Loading data...")
    x, y = load_data()
    print(f"[INFO] Data shape: {x.shape}, labels shape: {y.shape}")

    print("[INFO] Building feature extractor (ResNet50)...")
    feature_extractor = build_feature_extractor()

    print("[INFO] Preprocessing images...")
    x_pp = preprocess_images(x)

    batch_size = 64
    num_samples = x_pp.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    all_embeddings = []

    print("[INFO] Extracting embeddings...")
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_samples)
        batch = x_pp[start:end]

        emb = feature_extractor(batch, training=False).numpy()
        all_embeddings.append(emb)

        print(f"  Batch {b+1}/{num_batches} done, shape={emb.shape}")

    embeddings = np.vstack(all_embeddings)
    print(f"[INFO] Final embeddings shape: {embeddings.shape}")

    np.save(OUTPUT_EMB_FILE, embeddings)
    np.save(OUTPUT_LABELS_FILE, y)

    print(f"[INFO] Saved embeddings to {OUTPUT_EMB_FILE}")
    print(f"[INFO] Saved labels to {OUTPUT_LABELS_FILE}")

if __name__ == "__main__":
    extract_embeddings()
