# ArchVelo_notebooks
This repository contains notebooks for generating figures in the ArchVelo paper.

ArchVelo is a method for modeling gene regulation and inferring cell trajectories using single-cell simultaneous chromatin accessibility (scATAC-seq) and transcriptomic (scRNA-seq) profiling. ArchVelo represents chromatin accessibility as a set of archetypes---shared regulatory programs---and models their dynamic influence on transcription. As a result, ArchVelo improves inference accuracy compared to previous methods and decomposes the velocity field into components, each potentially corresponding to a specific regulatory program.

In this repository, we demonstrate ArchVelo on 3 different single-cell multi-omic datasets: mouse embryonic brain, human hematopoietic stem cells and CD8 T cells responding to acute and chronic viral infections. The notebooks for every dataset are numbered according to their order in the pipeline. The current core structure of every folder after downloading required raw data will be as follows:

|--**Dataset_name**
|&emsp; |-- **data**: raw data that requires preprocessing \
|&emsp; |-- **processed_data**: processed data to run the notebooks \
|&emsp; |-- **seurat_wnn**: auxiliary weighted nearest neighbor information \
|&emsp; |-- **1_Data_preparation.ipynb**: process data for ArchVelo analysis and benchmarking \
|&emsp; |-- **2_Create_archetypes.ipynb**: apply AA to the dataset \
|&emsp; |-- **3_ArchVelo_apply.ipynb**: apply ArchVelo to the dataset \
|&emsp; |-- **4_Compare_latent_times.ipynb**: benchmark ArchVelo against MultiVelo and scVelo \
|&emsp; |-- **5_CBDir.ipynb**: another benchmark of ArchVelo against MultiVelo and scVelo \
|&emsp; |-- **6_Trajectory_components.ipynb**: trajectory decomposition and interpretation of ArchVelo results \
|-- **archetypal_regression**: Archetypal Analysis for the ATAC modality and ATAC-to-RNA regression code \
|&emsp; |-- **archetypes.py**: delta-AA analysis (see https://github.com/ulfaslak/py_pcha) \
|&emsp; |-- **archetypes_regression.py**: module for ATAC-to-RNA regression \
|&emsp; |-- **util.py**: utility methods \
|&emsp; |-- **util_atac.py**: utility methods for the ATAC component \
|&emsp; |-- **util_regression.py**: utility for the regression \
|-- **ArchVelo.py**: ArchVelo methods \
|-- **UTV_metrics.py**: Metrics required for benchmarking \



