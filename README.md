# Visual Place Recognition project

This repository provides a starting code for the **Visual Place Recognition** project of the Advanced Machine Learning / Data analysis and Artificial Intelligence Course.

The following commands are meant to be run locally. If you plan to use Colab, upload the notebook [start_your_project.ipynb](./start_your_project.ipynb) and start from there.

> [!NOTE]  
> ### About datasets format
> The adopted convention is that the names of the files with the images are:
> ```
> @ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension
> ```
> Note that some of these values can be empty (e.g. the timestamp might be unknown), and the only required values are UTM coordinates (obtained from latitude and longitude).

> [!WARNING]  
> Some models require code implementation. You should identify which models require them, where they should be implemented, and then implement them.

## Install the repo

```sh
git clone --recursive https://github.com/FarInHeight/Visual-Place-Recognition-Project.git
```

## Install dependencies

```sh
cd Visual-Place-Recognition-Project/image-matching-models
pip install -e .[all]
pip install faiss-cpu
```

## Download Datasets

```sh
cd ..
python download_datasets.py
```

## Run VPR Evaluation

```sh
python VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--image_size 512 512 \
--database_folder '<path-to-database-folder>' \
--queries_folder '<path-to-queries-folder>' \
--num_preds_to_save 20 \
--recall_values 1 5 10 20 \
--save_for_uncertainty
```

## Run Image Matching on Retrieval Results

```sh
python match_queries_preds.py \
--preds-dir '<path-to-predictions-folder>' \
--matcher 'superpoint-lg' \
--device 'cuda' \
--num-preds 20
```

## Check Re-ranking Performance

```sh
python reranking.py \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--num-preds 20 \
--recall-values 1 5 10 20
```

## Perform Uncertainty Evalutation [only for AML students]

```sh
python -m vpr_uncertainty.eval \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--z-data-path '<path-to-z-data-file>'
```
