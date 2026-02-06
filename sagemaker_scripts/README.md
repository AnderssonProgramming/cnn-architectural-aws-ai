# SageMaker Deployment Scripts

This folder contains Python scripts for deploying the CNN Mars Classification model to AWS SageMaker.

## Scripts Overview

| Script | Purpose | Learner Lab Compatible |
|--------|---------|------------------------|
| `demo_training.py` | Demo that runs everything except endpoint creation | ✅ YES |
| `deploy.py` | Full deployment with real endpoint | ❌ NO (blocked) |
| `test_endpoint.py` | Test a deployed endpoint | ❌ NO (needs endpoint) |
| `cleanup.py` | Delete endpoint to avoid charges | ❌ NO (needs endpoint) |

## Which Script to Use?

### 🎓 In AWS Learner Lab (Academy)

Use `demo_training.py`:
```bash
python demo_training.py
```

This script will:
- ✅ Initialize SageMaker session
- ✅ Load Mars Surface Image dataset
- ✅ Build CNN model
- ✅ Train for 5 epochs
- ✅ Evaluate on test set
- ✅ Save model artifacts
- ❌ Skip actual endpoint creation (blocked by Lab policy)

### 💼 In Full AWS Account

Use the full deployment flow:
```bash
# 1. Deploy the model
python deploy.py

# 2. Test the endpoint
python test_endpoint.py

# 3. IMPORTANT: Clean up when done!
python cleanup.py
```

## Learner Lab Limitations

AWS Academy Learner Labs have restricted IAM policies (`VocLabPolicy`) that block:

| Action | Status |
|--------|--------|
| `sagemaker:CreateEndpointConfig` | ❌ Blocked |
| `sagemaker:CreateEndpoint` | ❌ Blocked |
| `sagemaker:ListEndpoints` | ✅ Allowed |
| `s3:PutObject` | ✅ Allowed |
| `sagemaker:CreateModel` | ✅ Allowed |

This is a cost-control measure by AWS Academy.

## Required Files

Before running any script, ensure these files are in your working directory:

```
cnn-architectural-aws-ai/
├── cnn_exploration.ipynb     # Main notebook
├── msl-images/               # Dataset folder
│   ├── calibrated/           # Image files
│   ├── train-calibrated-shuffled.txt
│   ├── val-calibrated-shuffled.txt
│   ├── test-calibrated-shuffled.txt
│   └── msl_synset_words-indexed.txt
└── sagemaker_scripts/        # This folder
    ├── demo_training.py
    ├── deploy.py
    ├── test_endpoint.py
    └── cleanup.py
```

## Instance Types

For **SageMaker Endpoints**, use:
- ✅ `ml.t2.medium` (recommended, cheapest)
- ✅ `ml.t2.large`
- ✅ `ml.m5.large`

**DO NOT USE** for endpoints:
- ❌ `ml.t3.*` (only for Studio/Notebooks, not for inference)
