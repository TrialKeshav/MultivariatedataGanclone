
# Code from <br>"Generative Adversarial Networks to solve data scarcity problem in future networks"

This repository contains information, code and models from the paper [Generative Adversarial Networks to solve data scarcity problem in future networks] by Keshav Bharadwaj, [M.Sc Faiaz Nazmetdinov,] and [Prof. Andreas Mitschele-Thiel]. Please visit the [project webpage here](https://github.com/Keshav-Bharadwaj/DeepFreq/). 

## Code and Pre-trained Models

Please refer to [`requirements.txt`](requirements.txt) for required packages. 

### Train

[`train.py`](train.py) provides the code for training a DNN model from scratch. An example usage of the script with some options is given below:

```shell
python train.py \
  --output_dir ./pretrained_models/ \
  --sample_data ./example_data/sample_data.txt \
  --batch_size 25 \
  --learning_rate 0.001 \
  --epochs 25 \ 
```


[`trainRnn`](trainRnn.py) provides the code for training a RNN model from scratch. An example usage of the script with some options is given below:

```shell
python trainRnn.py \
  --output_dir ./pretrainedRNN_models/ \
  --sample_data ./example_data/sample_data.txt \
  --cell_type lstm \
  --num_layers 2 \
  --hidden_size 128 \
  --batch_size 25 \
  --learning_rate 0.001 \
  --epochs 25 \
```

Please refer to the `argparse` module in [`train.py`](train.py) for additional training options.


### Evaluate

[`evaluation.py`](test.py) provides the script used for evaluation of the generated data.


```shell
python evaluation.py \
  --gendata_dir ./pretrained_models/finalSyntheticData.txt \
  --sample_data ./example_data/sample_data.txt \
  --n_estimators 125 \
  --random_state 42 \
  --bins 50 \
  --pathsave ./pretrained_models/residuals_kdeplot \
```

