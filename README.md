# MultivariatedataGan

Multivariate data generation for SON

# Code and Pretrained Networks from <br>"Generative Adversarial Networks to solve data scarcity problem in future networks"

This repository contains information, code and models from the paper [Generative Adversarial Networks to solve data scarcity problem in future networks] by Keshav Bharadwaj, [M.Sc Faiaz Nazmetdinov,] and [Prof. Andreas Mitschele-Thiel]. Please visit the [project webpage here](https://github.com/Keshav-Bharadwaj/DeepFreq/). 

## Code and Pre-trained Models

Please refer to [`requirements.txt`](requirements.txt) for required packages. 

### pre-trained models
The directory [`pretrained_models`](pretrained_models) contains the pretained models of Complex DeepFreq. 

### Train

[`train.py`](train.py) provides the code for training a DNN model from scratch. An example usage of the script with some options is given below:

```shell
python train.py 
```


[`trainRnn`](trainRnn.py) provides the code for training a RNN model from scratch. An example usage of the script with some options is given below:

```shell
python trainRnn.py 
```

Please refer to the `argparse` module in [`train.py`](train.py) for additional training options.
