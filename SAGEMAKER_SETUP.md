# AWS SageMaker Setup Guide

Step-by-step guide for running the CNN Exploration notebook on AWS SageMaker.

---

## ⚠️ Important: Learner Lab Limitations

**AWS Academy Learner Labs have restrictions** that prevent certain actions:

| Action | Status |
|--------|--------|
| Create SageMaker Domain | ✅ Allowed |
| Create Code Editor Space | ✅ Allowed |
| Run Jupyter Notebooks | ✅ Allowed |
| Train Models | ✅ Allowed |
| Upload to S3 | ✅ Allowed |
| **Create Endpoint** | ❌ **Blocked** |
| **Create EndpointConfig** | ❌ **Blocked** |

**Solution:** Use `demo_training.py` which demonstrates training and saves model artifacts without creating an endpoint.

---

## Prerequisites

- AWS Account with access to Amazon SageMaker (or AWS Academy Learner Lab)
- LabRole IAM role configured
- Project files ready:
  - `cnn_exploration.ipynb`
  - `msl-images/` dataset folder

## Supported Instance Types

| Resource | Supported Types |
|----------|-----------------|
| **Studio/Code Editor** | ml.t3.medium, ml.t3.large, ml.m5.large |
| **Training** | ml.t3.medium, ml.m5.large, ml.p3.2xlarge (GPU) |

> ⚠️ **Note:** For CNN training, `ml.t3.medium` works but may be slow. Use `ml.m5.large` for better performance.

---

## Step 1: Create a SageMaker Domain

1. Navigate to **Amazon SageMaker** in the AWS Console
2. Choose **Domains** → **Create domain**
3. Choose **Set up for organizations** → **Set up**

### Domain Configuration

1. **Domain Details:**
   - Name: `myDomain`
   - Keep **Login through IAM** default
   - Choose **Next**

2. **Roles:**
   - Choose **Use an existing role**
   - Set **Default execution role** to `LabRole`
   - Choose **Next**

3. **Applications:**
   - **SageMaker Studio:** Choose **SageMaker Studio - New**
   - **CodeEditor:** Enable idle shutdown (60 minutes)
   - Choose **Next**

4. **Network:**
   - Choose **VPC Only** or **Public internet access**
   - Select **Default VPC** and at least **two public subnets**
   - Choose **default security group**
   - Choose **Next** → **Submit**

5. Wait **5-8 minutes** for domain creation

---

## Step 2: Create a User Profile

1. In your domain, go to **User profiles** → **Add user**
2. Set **Execution role** to `LabRole`
3. Click **Next** through all steps → **Submit**

---

## Step 3: Create Code Editor Space

1. Go to **Studio** → Select your user profile → **Open Studio**
2. From **Applications**, choose **Code Editor**
3. Click **Create Code Editor space**
4. Name: `cnn-space` → **Create space**
5. Verify instance type is `ml.t3.medium` → **Run space**
6. Click **Open Code Editor** (opens VS Code in browser)

---

## Step 4: Upload Project Files

In Code Editor, upload these files via drag & drop:

```
├── cnn_exploration.ipynb     # Main notebook
├── msl-images/               # Dataset folder
│   ├── calibrated/           # Image files
│   ├── train-calibrated-shuffled.txt
│   ├── val-calibrated-shuffled.txt
│   ├── test-calibrated-shuffled.txt
│   └── msl_synset_words-indexed.txt
└── sagemaker_scripts/        # Deployment scripts
    ├── demo_training.py      # ✅ Use this in Learner Lab
    ├── deploy.py             # Full deployment (needs permissions)
    ├── test_endpoint.py      # Test deployed endpoint
    └── cleanup.py            # Delete endpoint
```

---

## Step 5: Run Training

### Option A: Learner Lab (Recommended)

Open Terminal in Code Editor (`Terminal > New Terminal`) and run:

```bash
cd ~/cnn-architectural-aws-ai
pip install tensorflow numpy matplotlib scikit-learn pillow pandas seaborn --quiet
python sagemaker_scripts/demo_training.py
```

**Expected Output:**
```
🚀 CNN MARS CLASSIFICATION - TRAINING DEMO
======================================================================
📦 Step 1: Initializing SageMaker session...
   ✅ Region: us-east-1
   ✅ Bucket: sagemaker-us-east-1-XXXX

📊 Step 2: Loading Mars Surface Image dataset...
   ✅ Training samples: 4,362
   ✅ Validation samples: 1,191
   ✅ Test samples: 1,138

🏗️ Step 3: Building CNN model...
   ✅ Model architecture created

🎯 Step 4: Training model (5 epochs demo)...
   Epoch 1/5 - acc: 0.15 - val_acc: 0.18
   Epoch 2/5 - acc: 0.22 - val_acc: 0.25
   ...

📈 Step 5: Results Summary
   ✅ Final Test Accuracy: XX.XX%

✅ TRAINING DEMO COMPLETE!
```

### Option B: Full AWS Account (When Permissions Allow)

```bash
# 1. Deploy endpoint (takes 5-10 minutes)
python sagemaker_scripts/deploy.py

# 2. Test the endpoint
python sagemaker_scripts/test_endpoint.py

# 3. CRITICAL: Delete endpoint to stop charges
python sagemaker_scripts/cleanup.py
```

---

## Troubleshooting

### Error: AccessDeniedException on CreateEndpointConfig

```
User is not authorized to perform: sagemaker:CreateEndpointConfig
with an explicit deny in an identity-based policy
```

**Cause:** Learner Lab policy blocks endpoint creation.  
**Solution:** Use `demo_training.py` instead.

### Error: Module Not Found

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install tensorflow --quiet
```

### Error: Out of Memory

**Cause:** Instance too small for CNN training.  
**Solution:** Use `ml.m5.large` instance instead of `ml.t3.medium`.

---

## Budget Tips

| Tip | Description |
|-----|-------------|
| **Stop Spaces** | Stop Code Editor when not in use |
| **Use t3.medium** | Good for setup, switch to larger only for training |
| **Monitor Dashboard** | Check SageMaker dashboard regularly |
| **Idle Shutdown** | Enable 60-min auto-shutdown |

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `demo_training.py` | Safe demo for Learner Labs - trains and evaluates locally |
| `deploy.py` | Full deployment - creates real endpoint |
| `test_endpoint.py` | Sends test images to endpoint |
| `cleanup.py` | Deletes endpoint to stop charges |

See [sagemaker_scripts/README.md](sagemaker_scripts/README.md) for detailed documentation.
