# Contrsative Discrepancy: a Label-free Metric for Deformable Image Registration Supporting Testing-Time Hyperparameter Selection

## Prepare the DIRLab dataset
- Download from the official [website](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/index.html)
- Process .img into .nii.gz
- Segment lung masks into the "Lungs" folder under each case using the repo [lungmask](https://github.com/JoHof/lungmask/blob/master/setup.py)

Organize the structure as

```bash
└── data
    └── DIRLab
        ├── Case1Pack
        │   ├── Images
        │   │   ├── case1_T00.nii.gz
        │   │   └── case1_T50.nii.gz
        │   ├── ExtremPhases
        │   ├── Sampled4D
        │   └── Lungs
        │       ├── case1_T00.nii.gz
        │       └── case1_T50.nii.gz
        ├── Case2Pack
        │       :
        └── Case10Pack
```

## Run the following commend to calculate CD
`bash scripts/run.sh`

## Run the following commend to conduct hyperparameter selection
`bash scripts/run_hyper.sh`
