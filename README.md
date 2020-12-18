# HealthcareMachineLearning

Course Project of Machine Learning for Healthcare, Spring 2020

Blog post: [https://sohamtamba.github.io/projects/deep-learning/2020/12/17/mlhc-tf.html](https://sohamtamba.github.io/projects/deep-learning/2020/12/17/mlhc-tf.html)


## Requirements

* Python: 3.8.5
* Pytorch: 1.7.0
* Torchvision: 0.8.1
* Pytorch-Lightning: 1.0.8
* PIL: 8.0.1

## Run the code yourself

To train a CBR-Large-Wide model on the entire dataset to solve the Consolidationt detection task, execute:

`python train_scratch --model_type cbr_large_wide -- task Consolidation --num_train_samples -1`


To train a CBR-Large-Wide model on 5,000 data points to solve the Edema detection task, execute:

`python train_scratch --model_type cbr_large_wide -- task Edema --num_train_samples 5000`

To fine-tune a CBR model to solve the  Edema detection task using 5,000 datapoints, execute:

`python finetune.py --task Edema --num_train_samples 5000 --backbone_path <Path to checkpoint created by train_scratch.py>`
