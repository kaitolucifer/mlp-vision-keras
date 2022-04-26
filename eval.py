import argparse
import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from self_attention import SelfAttention
from patch import Patch
from blocks import MLPMixerLayer, FNetLayer, gMLPLayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./model/",
                        help="File path to load model")
    parser.add_argument("--eval-dir", default="",
                        help="Image dataset from directory for evalutaing")
    parser.add_argument("-o", "--output-path", default="./output/",
                        help="File path to save output results")
    parser.add_argument("-is", "--image-size", required=True, type=int, help="Size of image")
    args = parser.parse_args()

    classifier = tf.keras.models.load_model(args.model_path, custom_objects={"SelfAttention": SelfAttention, "Patch": Patch,
                                                                             "MLPMixerLayer": MLPMixerLayer, "FNetLayer": FNetLayer, "gMLPLayer": gMLPLayer, "AdamW": tfa.optimizers.AdamW})
    eval = tf.keras.preprocessing.image_dataset_from_directory(
                args.eval_dir, label_mode=None, image_size=(args.image_size, args.image_size))
    preds = np.argmax(classifier.predict(eval), axis=1)
    np.savetxt(os.path.join(args.output_path, "output.txt"), preds, fmt='%d')