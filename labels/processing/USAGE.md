# Image and Label Processing

All repository paths below are relative to the repository root.

Skull stripping uses [HD-BET](https://github.com/MIC-DKFZ/HD-BET), which produces the required brain masks during processing. GLI repair uses NiftySeg's `seg_maths` and `seg_FillLesions`.

## BraTS: GLI, MEN, MET, PED, SSA

Use `labels/processing/prepare_brats_inputs.py` to read the official images for the selected dataset according to `labels/case_manifest.csv`. Specify the dataset name, image root, and output directory before processing. MEN, MET, PED, and SSA follow the same procedure, using their respective dataset names and image roots.

| Output suffix | Modality | Original image suffix |
|---|---|---|
| `_0000.nii.gz` | T1 | `-t1n.nii.gz` |
| `_0001.nii.gz` | FLAIR | `-t2f.nii.gz` |
| `_0002.nii.gz` | T1ce | `-t1c.nii.gz` |
| `_0003.nii.gz` | T2 | `-t2w.nii.gz` |

T1/FLAIR are exported by default. To prepare the four-modality jSTABL comparison data, add `--four-modalities` to also export T1ce/T2. Read the model inputs according to the modality mapping in the table.

For GLI, use `labels/processing/prepare_brats_inputs.py` with `labels/case_manifest.csv` and `labels/reproduction_resources/gli_repair_manifest.csv` to process the 116 cases requiring repair.

Cases requiring repair use all four modalities. The repair mask is defined as `1=whole tumor, 2=WMH`; the tumor dilation parameter is 3, and the WMH repair parameters are `-dil 2 -size 4`. Use the final segmentation labels from `labels/labels/GLI/`.

In `labels/labels/GLI/`, the 606 training cases outside the 394 clean cases do not include added coexisting lesions. The versions with coexisting lesions are in `labels/reproduction_resources/upper_bound_supplements/`. These contain only lesion labels; overlay the corresponding lesion regions onto the joint labels as needed.

## WMH, MS, Stroke, MRBrainS18

The SRI24 template is not bundled with this repository. Download the anatomical NIfTI package (`sri24_anatomy_nifti.zip`) from the [official SRI24 download page](https://www.nitrc.org/frs/?group_id=214), select the T1/SPGR template, and extract it to `labels/reproduction_resources/` with the filename `SRI24_spgr.nii`. Alternatively, store it elsewhere and pass the `.nii` or `.nii.gz` file through the `--template` option of `labels/processing/preprocess_images.py`.

The script uses the 1 mm, `240 x 240 x 155` SRI24 template grid and also accepts a four-dimensional file with a singleton final dimension. Preserve the downloaded template's spatial information for compatibility with the bundled registration transforms.

First use `labels/processing/make_image_manifest.py` to generate an input manifest from the dataset name and image root. Then use `labels/processing/preprocess_images.py` to read that manifest and process the images into the specified output directory.

The code locates the original images using the `dataset` and `case_id` fields in `labels/case_manifest.csv`; no additional manifest fields are needed. `labels/reproduction_resources/registration_transforms/` provides 1,348 registration matrices to SRI24 (WMH 170, MS 115, Stroke 1,033, MRBrainS18 30). The input manifest automatically links the matching matrices. FLAIR-to-T1 transforms for MS and Stroke are computed during processing.

| Dataset | Processing steps |
|---|---|
| WMH | Skull-strip T1/FLAIR; rigidly register T1 to SRI24; apply the same transform to FLAIR using linear interpolation |
| MRBrainS18 | Read `pre/reg_T1` and `pre/FLAIR`; process both modalities using the T1 skull-stripping result; rigidly register T1 to SRI24 and apply the same transform to FLAIR |
| MS | Skull stripping, N4 correction, rigid FLAIR-to-T1 registration, and template registration; resample FLAIR once using the composed transforms |
| Stroke | Orient to LPS, skull-strip, and apply N4 correction; resample T1 axes with spacing below 1 mm to 1 mm; register FLAIR to T1 and then to the template, retaining two interpolation steps |

## Generate Joint Upper Bound labels

Use `labels/processing/make_upper_bound_labels.py` with the desired output directory to generate 3,096 binary (0/1) lesion labels: 2,490 by extracting regions with value 4 from the primary labels, and 606 GLI labels from the supplements.

## Align label orientation

Use `labels/processing/align_labels.py`, specifying the input label, its corresponding processed reference image, and the output path, to permute and flip the label axes to match the reference image.
