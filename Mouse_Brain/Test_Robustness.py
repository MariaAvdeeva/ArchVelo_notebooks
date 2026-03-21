import ArchVelo as av
import numpy as np
import pandas as pd
import os
import scanpy as sc
import anndata
import pickle
data_outdir = 'processed_data/'
model_outdir = 'modeling_results/test_robustness/'
arch_outdir = 'modeling_results/choose_k/archetypes/'

os.makedirs(model_outdir, exist_ok = True)

n_jobs = -1
n_neigh = 50
n_pcs = 30

# preprocessed ATAC for ArchVelo
adata_atac_raw = sc.read_h5ad(data_outdir+'adata_atac_raw.h5ad')
# preprocessed ATAC for ArchVelo
adata_rna = sc.read_h5ad(data_outdir+'adata_rna.h5ad')
# aggregated atac for MultiVelo
adata_atac = sc.read_h5ad(data_outdir+'adata_atac.h5ad')

peak_annotation = pd.read_csv('data/outs/our_peaks/nearest_genes_summits_correct_annot.csv', index_col = [0])

np.random.seed(42)

for k in range(6,20,2):
    nn_idx = None
    nn_dist = None
    cur_outdir = arch_outdir+str(k)+'_comps/'
    cur_model_outdir = model_outdir+str(k)+'_comps/'
    os.makedirs(cur_outdir, exist_ok = True)
    os.makedirs(cur_model_outdir, exist_ok = True)
    f = open(cur_outdir+'res.p', 'rb')
    res = pickle.load(f)
    f.close()
    XC_raw, S_raw, _, _, _ = res
    XC_raw = pd.DataFrame(XC_raw, index = adata_atac_raw.obs.index)
    S_raw = pd.DataFrame(S_raw, columns = adata_atac_raw.var.index)
    _, gene_weights = av.annotate_and_summarize(S_raw, 
                                             peak_annotation, 
                                             outdir = cur_model_outdir)
    gene_weights = gene_weights.loc[:, adata_rna.var_names]
    # Save the result for use later on
    atac_AA_denoised = av.create_denoised_atac(adata_rna,
                                               gene_weights, 
                                            XC_raw,
                                            nn_idx = nn_idx, nn_dist = nn_dist,
                                            model_outdir = cur_model_outdir,
                                            n_pcs=n_pcs, 
                                            n_neighbors=n_neigh
                               )
    smooth_arches = sc.read_h5ad(cur_model_outdir+"arches.h5ad")
    avel = av.apply_ArchVelo_full(adata_rna,
                    atac_AA_denoised,
                    smooth_arches,
                    gene_weights,
                    cur_model_outdir,
                    n_jobs = n_jobs,
                    n_neighbors = n_neigh,
                    n_pcs = n_pcs)
    avel.write(cur_model_outdir+'avel.h5ad')