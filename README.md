# mlp-vision-keras
### Usage

#### training

```bash
python ./train.py --image-size 160 --train-dir ./train --save-path ./model --num-epochs 10 --embedding-dim 384 --mlp-block gmlp --positional-encoding --self-attention
```

#### testing
```bash
python ./predict.py --test-dir ./test --image-size 160 --output-path ./output
```

### Training data folder structure

```bash
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

### License
Distributed under an [MIT license](https://github.com/kaitolucifer/mlp-vision-keras/blob/main/LICENSE).