import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import json

class KeypointPrecomputedDatasetTF:
    def __init__(self, root_dir, keypoint_file, img_size):
        self.class_to_label = {"Belly": 0, "Back": 1, "Right_side": 2, "Left_side": 3}
        self.img_size = img_size
        self.root_dir = Path(root_dir)
        self.keypoints_data = self._load_keypoints_json(keypoint_file)
        self.samples = self._load_samples()

    def _load_keypoints_json(self, keypoint_file):
        with open(keypoint_file, "r") as f:
            data = json.load(f)
            # Преобразуем список словарей в словарь: путь -> {keypoints, label}
            return {
                item["path"]: {
                    "keypoints": item["keypoints"],
                    "label": item["label"]
                }
                for item in data
            }

    def _load_samples(self):
        samples = []
    
        for class_path in self.root_dir.iterdir():
            if not class_path.is_dir():
                continue
    
            label = self.class_to_label.get(class_path.name)
            if label is None:
                continue
    
            for ext in ("*.jpg", "*.png"):
                for image_path in class_path.glob(ext):
                    # Строим относительный путь от корня
                    relative_path = image_path.relative_to(self.root_dir)
    
                    # Добавляем префикс "depth/" вручную, чтобы соответствовать JSON
                    key = Path("depth") / relative_path
                    key = key.as_posix()  # Приведение к формату с '/' вместо '\'
    
                    if key not in self.keypoints_data:
                        
                        continue
    
                    keypoints = self.keypoints_data[key]["keypoints"]


                    samples.append((str(image_path), keypoints, label))

        return samples


    def _load_image(self, image_path):
        image_path = image_path.decode("utf-8")  # tf.numpy_function передаёт bytes
        img = Image.open(image_path).convert("L")
        img = img.resize((self.img_size, self.img_size))
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    def _make_tf_dataset(self, samples, batch_size=64, shuffle=True):
        image_paths, keypoints, labels = zip(*samples)

        def load_fn(image_path, kps, label):
            img = tf.numpy_function(
                func=self._load_image,
                inp=[image_path],
                Tout=tf.float32
            )
            img.set_shape((self.img_size, self.img_size, 1))
            kps = tf.convert_to_tensor(kps, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.int32)
            return (img, kps), label

        ds = tf.data.Dataset.from_tensor_slices((list(image_paths), list(keypoints), list(labels)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(samples))
        ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_tf_dataset(self, batch_size=64, shuffle=True):
        return self._make_tf_dataset(self.samples, batch_size=batch_size, shuffle=shuffle)

    def get_tf_dataset_from_samples(self, samples, batch_size=64, shuffle=True):
        return self._make_tf_dataset(samples, batch_size=batch_size, shuffle=shuffle)

    def get_all_data(self):
        """Возвращает numpy-массивы всех данных для прямого обучения без tf.data.Dataset"""
        image_paths, keypoints, labels = zip(*self.samples)
        images = np.stack([self._load_image(p) for p in image_paths])
        keypoints = np.array(keypoints, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return images, keypoints, labels
