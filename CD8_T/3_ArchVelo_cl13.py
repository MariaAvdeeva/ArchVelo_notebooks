import numpy as np
import pandas as pd
import scanpy as sc

import scvelo as scv
import multivelo as mv
import ArchVelo as av

import os
import pickle

from scipy.sparse import csr_matrix

#>>>>> FILL: which models would you like to run
#scVelo?
run_scv = False
#MultiVelo?
run_mv = False
#AA on ATAC?
run_archetypal = False
#MultiVelo-AA?
run_mv_aa =True
#ArchVelo?
run_aa = True


#>>>>> FILL: would you like to benchmark them for different w_c
indiv = True
benchmark = False

#>>>>> FILL parameters for models
num_comps = 9

n_jobs = -1
n_neighbors = 50
n_pcs=50

#>>>>> FILL: suffix for input directories
suff = 'cl13/'

#>>>>> FILL: point to peak annotation
peak_annotation = pd.read_csv('data/full_LCMV_T_cell_dataset/nearest_genes_to_summits.distances.csv', index_col = [0,1])#.droplevel(0)
peak_annotation = peak_annotation.reset_index().set_index('name')

#>>>>> DO NOT CHANGE ANYTHING BELOW

# data
data_outdir = 'processed_data/'+suff+'/'
# results of AA
arch_dir = 'modeling_results/2_clones/archetypes/'

nn_idx = None
nn_dist = None

#>>>>> output directory
model_outdir = 'modeling_results/'+suff+'/joint_arches/'+str(num_comps)+'_comps/'
os.makedirs(model_outdir, exist_ok = True)

print('Reading data...')
# load processed RNA
adata_rna = sc.read_h5ad(data_outdir+'adata_rna.h5ad')
# full processed atac for ArchVelo
adata_atac_raw = sc.read_h5ad(data_outdir+'adata_atac_raw.h5ad')
# aggregated atac for MultiVelo
adata_atac = sc.read_h5ad(data_outdir+'adata_atac.h5ad')

#Run methods

if run_archetypal:
    print('Running AA...')
    XC, S = av.apply_AA_no_test(adata_atac_raw, 
                  k = num_comps,
                  outdir = arch_dir, 
                  )

XC_raw = pd.read_csv(arch_dir+'cell_on_peaks_'+str(num_comps)+'_comps.csv', index_col = [0])
S_raw = pd.read_csv(arch_dir+'peak_on_peaks_'+str(num_comps)+'_comps.csv', index_col = [0])
XC_raw = XC_raw.loc[adata_rna.obs_names,:]


try:
    adata_atac.layers['Mc'] = csr_matrix(np.array(adata_atac.layers['Mc'].todense()))
except:
    pass
try:
    adata_atac_raw.layers['Mc'] = csr_matrix(np.array(adata_atac_raw.layers['Mc']))
except:
    pass

if run_scv:
    print('Running scVelo...')
    rna_copy = adata_rna.copy()
    scv.tl.recover_dynamics(rna_copy, 
                            #var_names = rna_copy.var_names,
                            n_jobs = -1)
    scv.tl.velocity(rna_copy, mode='dynamical')
    scv.tl.velocity_graph(rna_copy)
    scv.tl.latent_time(rna_copy)
    rna_copy.write(model_outdir+'scvelo_result.h5ad')

if run_mv:
    print('Running MultiVelo...')
    if indiv:
        adata_result = mv.recover_dynamics_chrom(adata_rna, 
                                             adata_atac,
                                             n_jobs = n_jobs,
                                             n_neighbors = n_neighbors,
                                             n_pcs = n_pcs
                                            )
        adata_result.write(model_outdir+'multivelo_result.h5ad')
    
    if benchmark:
        for wc in np.linspace(0,1,11):
            adata_result = mv.recover_dynamics_chrom(adata_rna, 
                                                 adata_atac,
                                                 weight_c = wc,
                                                 n_jobs = n_jobs,
                                                 n_neighbors = n_neighbors,
                                                 n_pcs = n_pcs
                                                )
            adata_result.write(model_outdir+'multivelo_result_weight_c_'+str(np.round(wc, 1))+'.h5ad')

if run_mv_aa:
    print('Running MultiVelo_AA...')
    if indiv:
        full_res_denoised = av.apply_MultiVelo_AA(adata_rna, 
                                               XC_raw, S_raw, 
                                               peak_annotation,
                                               nn_idx, 
                                               nn_dist,
                                               model_outdir = model_outdir,
                                               n_jobs = n_jobs,
                                               n_neighbors = n_neighbors,
                                               n_pcs = n_pcs
                                       )
        full_res_denoised.write(model_outdir+'multivelo_result_denoised_chrom.h5ad')

    if benchmark:
        for wc in np.linspace(0,1,11):
            full_res_denoised = av.apply_MultiVelo_AA(adata_rna, 
                                                   XC_raw, S_raw,
                                                   peak_annotation,
                                                   nn_idx, nn_dist,
                                                   weight_c = wc, 
                                                   model_outdir = model_outdir,
                                                   n_jobs = n_jobs, 
                                                   n_neighbors = n_neighbors,
                                                   n_pcs = n_pcs    
                                   )
            full_res_denoised.write(model_outdir+'multivelo_result_denoised_chrom_weight_c_'+str(np.round(wc, 1))+'.h5ad')

if run_aa:
    print('Running ArchVelo...')
    adata_result = sc.read_h5ad(model_outdir+'multivelo_result.h5ad')
    full_res_denoised = sc.read_h5ad(model_outdir+'multivelo_result_denoised_chrom.h5ad')
    smooth_arch = sc.read_h5ad(model_outdir+'arches.h5ad')
    gene_weights = pd.read_csv(model_outdir+'gene_weights.csv', index_col = [0])
    rna_conn = full_res_denoised.obsp['_RNA_conn']
    min_c, max_c = av.extract_minmax(smooth_arch)
    # will run ArchVelo on top MultiVelo likelihood genes
    top_lik = full_res_denoised.var['fit_likelihood'].sort_values(ascending = False).index
    # save these genes
    f = open(model_outdir+'top_lik.p', 'wb')
    pickle.dump(top_lik, f)
    f.close()
    
    if benchmark:
        wcs = np.linspace(0,1,11)
    
        for wc in wcs:
            print('Weight_c = '+str(np.round(wc,1)))
            av_pars = av.extract_ArchVelo_pars(adata_rna, 
                          full_res_denoised, 
                          smooth_arch,
                          gene_weights,
                          weight_c = wc, 
                          n_jobs = n_jobs)
            f = open(model_outdir+'archevelo_results_weight_c_'+str(np.round(wc,1))+'.p', 'wb')
            pickle.dump(av_pars, f)
            f.close()
    if indiv:
        avel = av.apply_ArchVelo(adata_rna, 
                   full_res_denoised,
                   smooth_arch,
                   gene_weights,
                   model_outdir)
        

        avel.write(model_outdir+'archvelo_result.h5ad')
    

