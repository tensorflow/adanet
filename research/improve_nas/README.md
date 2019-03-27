**This is not a Google product.**

Improving Neural Architecture Search Image Classifiers via Ensemble Learning
============================

<div>
  <div style="float:left">
    <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/search_space.png" alt="adanet_search_space"><br><br>
  </div>

  <div style="float:left">
    <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/ensemble.png" alt="ensemble"><br><br>
  </div>
</div>


## Introduction
We present an algorithm that can improve the performance of NASNet models by learning an ensemble of smaller models with minimal hyperparameter tuning.
Interestingly, a simple ensemble of identical architectures trained independently with a uniform averaged output performs better than the baseline single large model.
Conversely, our adaptive methods show performance gains for applications where we can afford to train ensemble sequentially.
We were able to achieve near state-of-the-art results by using a combination of learning mixture weights and applying Adaptive Knowledge Distillation.

This paper was done as a part of Google AI residency.


## Paper results

<div>
  <div style="float:left">
    <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/ensemble_accuracy_cif10.png" alt="ensemble_accuracy_cifar_10"><br><br>
    Accuracy of ensemble in CIFAR-10.
  </div>

  <div style="float:left">
    <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/ensemble_accuracy_cif100.png" alt="ensemble_accuracy_cifar_100"><br><br>
    Accuracy of ensemble in CIFAR-100.
  </div>
</div>

Our experiments demonstrate that ensembles of subnetworks improve accuracy upon a single neural network with the same number of parameters.
On CIFAR-10 our algorithm achieves error 2.26 and on CIFAR-100 it achieves error 14.58.
To our knowledge, our technique achieves a new state-of-the-art on CIFAR-100 without using additional regularization or data augmentation (e.g., [Shake-Drop](https://arxiv.org/abs/1802.02375) or [AutoAugment](https://arxiv.org/abs/1805.09501)).


<div align="center">
  <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/cif10_caption.png" alt="accuracy_improvements_cifar10"><br><br>
</div>

<div align="center">
  <img src="https://tensorflow.github.io/adanet/research/improve_nas/images/cif100_caption.png" alt="accuracy_improvements_cifar100"><br><br>
</div>




This is the code accompanying paper: [Improving Neural Architecture Search Image Classifiers via Ensemble Learning](https://arxiv.org/abs/1903.06236), currently under review for ICML 2019.

For instructions on running the code on google cloud, follow [sararob/adanet-ml-engine](https://github.com/sararob/adanet-ml-engine)


## Prerequisites
Follow instructions in [sararob/adanet-ml-engine](https://github.com/sararob/adanet-ml-engine) to set up your Cloud project, install gcloud CLI, setup a storage bucket and run the job.


## Reproduction
To reproduce a simple experiment (ensemble of 10 NASNet(6@768)), setup a few environment variables on your local machine:

```
export JOBNAME=<unique_job_name>  # improvenastest1
export REGION=us-east1;
export MODULE=trainer.model;
export PACKAGE_PATH=trainer/;
export JOB_DIR=<path_to_your_bucket>;  # gs://improve_nas_bucket/test1
```

then go to `improve_nas` directory and run (still on your local machine):

```
gcloud ml-engine jobs submit training $JOBNAME --package-path trainer/ \
--module-name trainer.trainer --job-dir $JOB_DIR/$JOBNAME --region $REGION \
--config config.yaml -- --batch_size=1  --data_params=augmentation=basic,cutout=True \
--dataset=cifar10  --train_steps=10000000   \
--hparams="adanet_beta=0.0,adanet_lambda=0.0,boosting_iterations=10,force_grow=True,\
knowledge_distillation=none,generator=simple,learn_mixture_weights=False,\
initial_learning_rate=0.025,learning_rate_schedule=cosine,aux_head_weight=0.4,\
clip_gradients=5,data_format=NHWC,dense_dropout_keep_prob=1.0,drop_path_keep_prob=0.6,\
filter_scaling_rate=2.0,label_smoothing=0.1,model_version=cifar,num_cells=6,\
num_conv_filters=32,num_reduction_layers=2,optimizer=momentum,skip_reduction_layer_input=0,\
stem_multiplier=3.0,use_aux_head=False,weight_decay=0.0005" \
--save_summary_steps 10000 --save_checkpoints_secs 600 --model_dir=$JOB_DIR/$JOBNAME
```

To train mixture weights, set hparam `learn_mixture_weights=True`.
To use knowledge distillation, set hparam `knowledge_distillation` to `adaptive` or `born_again`.
Finally, to perform architecture search (dynamic generator) set hparam `generator` to `dynamic` and adjust `num_cells` and `num_conv_filters` to set the initial architecture.

For testing, use `--config config_test.yaml` (uses only one GPU), change the eval steps `--eval_steps=1` and set the `--dataset=fake`.

## Cite

```
@article{macko2019improving,
  title={Improving Neural Architecture Search Image Classifiers via Ensemble Learning},
  author={Macko, Vladimir and Weill, Charles and Mazzawi, Hanna and Gonzalvo, Javier},
  journal={arXiv preprint arXiv:1903.06236},
  year={2019}
}
```

