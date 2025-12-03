# COMP6630 Project

Requirement
```
scikit-learn
pytorch
matplotlib
scanpy

```

### To download data:

```
python data_download.py
```

### To process data and do train/val/test split:

```
python data_processing.py
```

### To train and evaluate the model

```
python train.py --gpu [GPU_ID]
```

### To run hyperparameters tuning with random search 

```
python hyperparameter_tuning.py --gpu [GPU_ID] --n_trials [NUMBER OF RANDOM TRIALS]
```

### To evaluate the best model

```
python evaluate_best_model.py --gpu [GPU_ID] 
```
#### Umap visualization of the features extracted by the trained model, colored by groundtruth labels and predicted labels.
![UMAP](figures/umap_test_embeddings_ondata.png)


    
