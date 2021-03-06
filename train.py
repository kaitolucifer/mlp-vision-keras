import argparse
import sys

import tensorflow as tf
import tensorflow_addons as tfa

from classifier import Classifier
from data_augumentation import get_data_augmentation

from utils import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mb", "--mlp-block", type=str, default="gmlp", choices=[
                        "mlp-mixer", "fnet", "gmlp"], help="Choose which block type to use in classifier (mlp-mixer, fnet, gmlp)")
    parser.add_argument("-nb", "--num-blocks", type=int,
                        default=config.num_blocks, help="Num of blocks to use in classifier")
    parser.add_argument("-bs", "--batch-size",
                        default=config.batch_size, type=int, help="Batch size")
    parser.add_argument("-ne", "--num-epochs",
                        default=config.num_epochs, type=int, help="Num of epochs")
    parser.add_argument("-lr", "--learning-rate", type=float,
                        default=config.learning_rate, help="Learning rate of AdamW optimizer")
    parser.add_argument("-wd", "--weight-decay", default=config.weight_decay,
                        help="Weight decay of AdamW optimizer")
    parser.add_argument("-is", "--image-size",
                        default=config.image_size, type=int, help="Size of image")
    parser.add_argument("-ps", "--patch-size",
                        default=config.patch_size, type=int, help="Size of patches")
    parser.add_argument("-emb", "--embedding-dim",
                        default=config.embedding_dim, type=int, help="Dimension of embedding layers")
    parser.add_argument("-dr", "--dropout-rate",
                        default=config.dropout_rate, type=float, help="Dropout rate")
    parser.add_argument("-pe", "--positional-encoding",
                        action="store_true", help="Use positional encoding")
    parser.add_argument("-sa", "--self-attention",
                        action="store_true", help="Use self attention")
    parser.add_argument("-nh", "--num-heads", type=int,
                        default=config.num_heads, help="Num of heads to use in self attention")
    parser.add_argument("--train-dir", default="./train/",
                        help="Image dataset from directory for training (train_dir/label_no(0, 1, ...)/image.jpg)")
    parser.add_argument("--val-dir", default="",
                        help="Image dataset from directory for validating (val_dir/label_no(0, 1, ...)/image.jpg)")
    parser.add_argument("-s", "--save-path", default="./model/",
                        help="File path to save model")
    args = parser.parse_args()

    if args.self_attention:
        if args.embedding_dim % 3:
            print("embedding dim must be multiples of 3 if using self attention")
            sys.exit(1)
        elif args.embedding_dim % args.num_heads != 0:
            print("embedding dim must be be exactly divisible by num heads")
            sys.exit(1)

    print(f"MLP block type: {args.mlp_block}")
    num_patches = (args.image_size // args.patch_size) ** 2

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.save_path,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    if args.val_dir:
        train = tf.keras.preprocessing.image_dataset_from_directory(
            args.train_dir, label_mode="int", image_size=(args.image_size, args.image_size))
        val = tf.keras.preprocessing.image_dataset_from_directory(
            args.val_dir, label_mode="int", image_size=(args.image_size, args.image_size))
    else:
        train = tf.keras.preprocessing.image_dataset_from_directory(
            args.train_dir, label_mode="int", image_size=(args.image_size, args.image_size),
            validation_split=0.1, subset="training", seed=42)
        val = tf.keras.preprocessing.image_dataset_from_directory(
            args.train_dir, label_mode="int", image_size=(args.image_size, args.image_size),
            validation_split=0.1, subset="validation", seed=42)

    augmentation = get_data_augmentation()
    classifier = Classifier(mlp_block=args.mlp_block, num_blocks=args.num_blocks, embedding_dim=args.embedding_dim, dropout_rate=args.dropout_rate,
                            patch_size=args.patch_size, num_patches=num_patches, num_classes=len(train.class_names), augmentation=augmentation,
                            positional_encoding=args.positional_encoding, num_heads=args.num_heads)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
    )
    classifier.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                5, name="top5-acc"),
        ],
    )
    history = classifier.fit(
        train,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        validation_data=val,
    )
    _, accuracy, top_5_accuracy = classifier.evaluate(val)

    print(f"Val accuracy: {round(accuracy * 100, 2)}%")
    print(f"Val top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
