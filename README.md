# Physics-Driven Pseudo-Lesion Synthesis for Joint Brain Tissue and Lesion Segmentation

This repository provides the pseudo-lesion generation, training integration, baseline adaptations, ablations, data processing, and metric computation code used in the paper, together with labels for 3,996 cases across nine datasets, case mappings, and image processing scripts. The training integration files are based on MedNeXt.

## Repository structure

| Location | Purpose |
|---|---|
| [code/](code/) | Method implementation, training adaptations, inference, evaluation, and figure generation |
| [labels/labels/](labels/labels/) | Primary labels organized by dataset; see below for coexisting-lesion supplements for 606 GLI training cases |
| [labels/case_manifest.csv](labels/case_manifest.csv) | Case identifiers and training/test assignments |
| [labels/processing/USAGE.md](labels/processing/USAGE.md) | Image preparation, label alignment, and Upper Bound label generation |
| [labels/reproduction_resources/](labels/reproduction_resources/) | Registration transforms, GLI repair masks, and supplementary labels; download the SRI24 template separately |

All repository paths below are relative to the repository root. Target paths within MedNeXt are relative to `$MEDNEXT_ROOT/`.

Example commands use Linux Bash. Replace `/path/to/...` placeholders in the code with your own paths before running the corresponding scripts.

## 1. Obtain the base projects

Obtain the base project from the [official MedNeXt repository](https://github.com/MIC-DKFZ/MedNeXt) and install it in your Python environment.

FDAR uses `FluidAnomaly` from the [official UNA repository](https://github.com/jhuldr/UNA). After installing its dependencies, add the **parent directory containing `FluidAnomaly/`** to the Python search path:

```bash
export UNA_ROOT=/path/to/UNA
export PYTHONPATH="$UNA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

The interfaces used are `FluidAnomaly.DiffEqs.pde.AdvDiffPDE`, `FluidAnomaly.DiffEqs.adjoint.odeint_adjoint`, and `FluidAnomaly.perlin3d`. Retain the upstream licenses and copyright notices when using these projects.

## 2. Main method files

All target paths in the middle column are relative to `$MEDNEXT_ROOT/`.

| Repository file | Target location in MedNeXt | Purpose |
|---|---|---|
| `code/method/online/pseudo_lesions.py` | `nnunet_mednext/training/data_augmentation/pseudo_lesions.py` | Online pseudo-lesion generation and blending |
| `code/method/online/dataset_loading.py` | `nnunet_mednext/training/dataloading/dataset_loading.py` | Invoke augmentation during data loading |
| `code/method/online/nnUNetTrainer.py` | `nnunet_mednext/training/network_training/nnUNetTrainer.py` | Pass asset paths to the data loader |
| `code/evaluation/metrics/metrics.py` | `nnunet_mednext/evaluation/metrics.py` | Compute evaluation metrics |
| `code/evaluation/metrics/evaluator.py` | `nnunet_mednext/evaluation/evaluator.py` | Invoke metrics and load and evaluate predictions |

## 3. Morphology and texture generation

These scripts run outside MedNeXt. Keep each configuration file in the same directory as the script that uses it.

| Purpose | Code location |
|---|---|
| Simplex texture generation | `code/method/simplex/generate_simplex_textures.py`, function `generate_sharp_npy()` |
| Random FDAR morphology generation | `code/method/fdar/calx0.py` and `config_randomizer.py` |
| Convert real lesions to probability maps | `code/method/fdar/map.py` |
| FDAR evolution of real lesions | `code/method/fdar/caltrue.py` and `config_random_true.py` |

### Connect the asset paths

1. Set `save_dir` in `generate_simplex_textures.py` to generate texture `.npy` files.
2. Set `output_dir` in `calx0.py` to generate random morphology `.npy` files. For the real-lesion branch, run `map.py` followed by `caltrue.py`, setting the label, probability-map, and evolved-morphology directories as appropriate.
3. In `get_basic_generators()` of the `nnUNetTrainer.py` **copied into MedNeXt**, point `lesion_dir_path` to the morphology library and `noise_dir_path` to the texture library. Also set the corresponding default paths in `dataset_loading.py`.

## 4. Switch between baselines and ablations

Files with the same name in different experiments replace one another. Separate MedNeXt copies are recommended. In the table below, "the corresponding location" refers to the target paths in Section 2.

| Experiment | File placement |
|---|---|
| Ours | Use the three files in `code/method/online/` listed in Section 2. |
| w/o Simplex | After integrating Ours, replace the corresponding file with `code/ablations/without_simplex/pseudo_lesions.py`. |
| w/o FDAR | After integrating Ours, replace the corresponding file with `code/ablations/without_fdar/dataset_loading.py`; change the morphology directory passed by the trainer to the ellipsoid library. |
| w/o MIPB | After integrating Ours, replace the corresponding files with `pseudo_lesions.py` and `dataset_loading.py` from `code/ablations/without_mipb/`. |
| UNA | Replace the three corresponding files from Section 2 with those in `code/baselines/una/`; set the trainer's `prob_dir_path` and the loader's default path. Probability-map generation scripts and configurations are in `code/baselines/una/asset_generation/`. |
| WMH-source | Replace the three corresponding files from Section 2 with those in `code/wmh_source/`. |
| MedLesSynth-LD | Keep `code/baselines/medlessynth_ld/control_pi.py` and `fun_pi.py` in the same directory. Generate the augmented data offline, then train with a MedNeXt project that does not load the Ours online augmentation. |

## 5. Data, training, and inference

The inputs are T1 and FLAIR, named `case_0000.nii.gz` and `case_0001.nii.gz`, respectively. The shared label scheme is: 0 background, 1 GM, 2 BG, 3 WM, 4 lesion, 5 ventricles, 6 cerebellum, and 7 brainstem.

The main method is trained on 394 clean GLI cases. The corresponding labels, case manifest, and image preparation scripts are in `labels/`; `primary_clean394=true` identifies these training cases. Obtain the original MRI images from the official BraTS resources.

`code/preprocessing/` contains dataset-specific processing code, including label fusion, registration, resampling, and N4 correction.

Follow the MedNeXt/nnU-Net inference workflow. Additional inference scripts are in `code/inference/`.

## 6. Metric computation

For raw BrainSeg outputs, construct the lesion mask using `(raw == 0) & (support > 0)`.

`code/figures/generate_combined_radars.py` generates the combined radar charts and contains the numerical results reported in the paper.

## 7. Labels and image preparation

### Case counts

| Dataset | Training pool | Test | Total |
|---|---:|---:|---:|
| GLI | 1,000 | 251 | 1,251 |
| MEN | 800 | 200 | 1,000 |
| WMH | 60 | 110 | 170 |
| Stroke / SOOP | 826 | 207 | 1,033 |
| MET | 190 | 48 | 238 |
| MSLesSeg | 93 | 22 | 115 |
| PED | 79 | 20 | 99 |
| SSA | 48 | 12 | 60 |
| MRBrainS18 | 0 | 30 | 30 |
| **Total** | **3,096** | **900** | **3,996** |

The case manifest contains five fields: `dataset`, `case_id`, `split`, `primary_clean394`, and `historical_case_id`. These record the dataset, case identifier, training/test split, membership in the 394 clean GLI cases used by the main method, and historical internal identifier, respectively.

### Labels and experiment assignments

Joint labels use `0=background, 1=GM, 2=BG, 3=WM, 4=lesion, 5=Ven, 6=Cer, 7=BS`. GLI/MEN/WMH/MRBrainS18 provide joint labels; MET/MS/PED/SSA/Stroke provide lesion labels with values 0 and 4.

| Experiment | Selection in the case manifest |
|---|---|
| Main method, matched synthesis baselines, and ablations | `primary_clean394=true`: 394 GLI cases |
| Joint-Baseline | `dataset=GLI` and `split=train`: 1,000 cases |
| Joint Upper Bound | `split=train`: 3,096 cases |
| WMH-Baseline / WMH-Ours supplementary experiments | `dataset=WMH` and `split=train`: 60 WMH cases |
| Paper evaluation | `split=test`: 900 cases |

The GLI repair masks in `labels/reproduction_resources/gli_repair_masks/` are sourced from [johncolby/svid_paper](https://github.com/johncolby/svid_paper).

**GLI training labels:** Among the 1,000 training cases in `labels/labels/GLI/`, the primary labels for the 606 cases outside the 394 clean cases used by the main method do not include added coexisting lesions. The versions with coexisting lesions are in `labels/reproduction_resources/upper_bound_supplements/`: 606 files named `GLI_<case_id>.nii.gz`.

These supplementary files contain only lesion labels (`0=background, 1=lesion`) and no brain tissue labels. To obtain joint labels that include coexisting lesions, overlay the regions with value 1 in each supplementary label onto the corresponding joint label in `labels/labels/GLI/`, assigning value 4 to those regions and retaining the brain tissue labels elsewhere.

Joint Upper Bound uses lesion value 1. Of its 3,096 training labels, 2,490 are generated by extracting regions with value 4 from the primary labels; the remaining 606 GLI cases use the supplementary labels described above.

### Workflow

1. Download images from the official sources and match them using the original case identifiers in the case manifest.
2. Prepare the images following the [image and label processing guide](labels/processing/USAGE.md). Before processing WMH, MS, Stroke, or MRBrainS18, download the SRI24 template as described in that guide. The main method uses T1/FLAIR; the four-modality jSTABL comparison additionally uses T1ce/T2.
3. Pair the output images with the labels sharing the same case identifiers.
4. Select training and test cases using the case manifest.

### Download the original images

| Dataset | Official source and version details |
|---|---|
| GLI / MEN / MET / PED / SSA | [BraTS 2023](https://www.synapse.org/Synapse:syn51156910); MET uses the 165-case main package plus 73 Additional cases; SSA uses the 60-case V2 release; PED uses the 99-case 2023 release |
| WMH | [WMH challenge](https://wmh.isi.uu.nl/); use `pre/T1` and `pre/FLAIR` in the corresponding directories |
| MSLesSeg | [Figshare DOI](https://doi.org/10.6084/m9.figshare.27919209.v1) |
| Stroke / SOOP | [OpenNeuro ds004889 v1.1.2](https://openneuro.org/datasets/ds004889/versions/1.1.2); select the 1,033 cases listed in the manifest |
| MRBrainS18 | [Official website](https://mrbrains18.isi.uu.nl/) and [DataverseNL DOI](https://doi.org/10.34894/E0U32Q); all 30 cases are used for paper evaluation |

## Copyright and licensing

Original code, additional annotations, and modifications in this repository that the author is entitled to license are released under the [MIT License](LICENSE). The copyright notice is `Copyright (c) 2026 Xingyu Xiang`. Author: Xingyu Xiang, Xi'an Jiaotong University.

Third-party code, source annotations, and other upstream materials retain their original licenses and terms of use. Related derivative content remains subject to applicable upstream requirements, including attribution, citation, and noncommercial restrictions where applicable. The MIT License does not replace or waive these requirements.

When using this project for research, please cite this paper and the relevant original works.
