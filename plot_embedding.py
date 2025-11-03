import scanpy as sc


adata_emb = sc.read_h5ad('./results/test_embeddings.h5ad')
sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X')
sc.tl.umap(adata_emb)
sc.pl.umap(adata_emb, color=['label', 'predicted_label'], save='_test_embeddings.png')