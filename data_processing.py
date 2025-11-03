#%%
import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read("./data/dataset.h5ad")
print(adata)
## remove low variance genes
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
print(adata)
# %%
from sklearn.model_selection import train_test_split

## get all donor id
donor_ids = adata.obs['donor_id'].unique().tolist()
print(donor_ids)
train_ids, test_ids = train_test_split(donor_ids, test_size=0.2, random_state=42)

train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)  
print("train_ids:", train_ids)
print("val_ids:", val_ids)
print("test_ids:", test_ids)

### train_ids: ['postnatB_7', 'postnatB_8', 'postnatB_4', 'postnatB_9', 'postnatB_1', 'postnatB_5']
### val_ids: ['postnatB_3', 'postnatB_2']
### test_ids: ['postnatB_6', 'postnatB_10']


train_adata = adata[adata.obs['donor_id'].isin(train_ids)].copy()
val_adata = adata[adata.obs['donor_id'].isin(val_ids)].copy()
test_adata = adata[adata.obs['donor_id'].isin(test_ids)].copy()

## filter cells with common cell types in train and test set
common_cell_types = np.intersect1d(
    np.intersect1d(train_adata.obs['cell_type'].unique(), val_adata.obs['cell_type'].unique()),
    test_adata.obs['cell_type'].unique()
)

train_adata = train_adata[train_adata.obs['cell_type'].isin(common_cell_types)].copy()
val_adata   = val_adata[val_adata.obs['cell_type'].isin(common_cell_types)].copy()
test_adata = test_adata[test_adata.obs['cell_type'].isin(common_cell_types)].copy() 

## write the matrices as numpy array to /data/dungp/projects/COMP6630/project/processed_data
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/train_X.npy", train_adata.X.toarray())
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/train_y.npy", train_adata.obs['cell_type'].values)
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/val_X.npy", val_adata.X.toarray())
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/val_y.npy", val_adata.obs['cell_type'].values)
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/test_X.npy", test_adata.X.toarray())
np.save("/data/dungp/projects/COMP6630/project/data/processed_data/test_y.npy", test_adata.obs['cell_type'].values)


print("Number of training cells:", train_adata.n_obs)
print("Number of validation cells:", val_adata.n_obs)
print("Number of testing cells:", test_adata.n_obs)
print("Common cell types:", common_cell_types)
# %%
