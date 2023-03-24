pip install -r requirements_PatchCore.txt


python bin/run_patchcore.py \
		--gpu 0 --seed 0 # Set GPU-id & reproducibility seed.
		--save_patchcore_model --save_segmentation_images # If set, saves the patchcore model(s) and segmentation images.
		--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results # Logging details: Name of the run & Name of the overall project folder.

	patch_core # We now pass all PatchCore-related parameters.
		-b wideresnet50 # Which backbone to use.
		-le layer2 -le layer3 # Which layers to extract features from.
		--pretrain_embed_dimension 1024 --target_embed_dimension 1024 # Dimensionality of features extracted from backbone layer(s) and final aggregated PatchCore Dimensionality
		--anomaly_scorer_num_nn 1 --patchsize 3 # Num. nearest neighbours to use for anomaly detection & neighbourhoodsize for local aggregation.

	sampler  # We now pass all the (Coreset-)subsampling parameters.
		-p 0.1 approx_greedy_coreset # Subsampling percentage & exact subsampling method.

	dataset # We now pass all the Dataset-relevant parameters.
		--resize 256 --imagesize 224 --subdatasets "class_name (ex: bottle)" mvtec "path/to/dataset/mvtec/" # Initial resizing shape and final imagesize (centercropped) as well as the MVTec subdatasets to use.