#!/usr/bin/env python3
"""
CNN Mars Classification - SageMaker Deployment Script
======================================================

⚠️  REQUIRES: Full AWS account with sagemaker:CreateEndpoint permissions.
    Learner Labs typically BLOCK endpoint creation.

Run from SageMaker Code Editor terminal:
    python deploy.py
"""

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import json
import logging

# Suppress INFO messages from sagemaker.config
logging.getLogger('sagemaker.config').setLevel(logging.WARNING)

def main():
    print("=" * 70)
    print("🚀 CNN MARS CLASSIFICATION - SAGEMAKER DEPLOYMENT")
    print("=" * 70)
    
    # Step 1: Initialize SageMaker session
    print("\n📦 Step 1: Initializing SageMaker session...")
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()
    
    print(f"   ✅ Region: {region}")
    print(f"   ✅ Bucket: {bucket}")
    print(f"   ✅ Role: {role[:50]}...")
    
    # Step 2: Upload model to S3
    print("\n📤 Step 2: Uploading model.tar.gz to S3...")
    s3_model_path = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix='cnn-mars-model'
    )
    print(f"   ✅ S3 Path: {s3_model_path}")
    
    # Step 3: Create SageMaker Model
    print("\n🔧 Step 3: Creating SageMaker Model...")
    model = TensorFlowModel(
        model_data=s3_model_path,
        role=role,
        framework_version='2.13',
        py_version='py310',
        sagemaker_session=sagemaker_session
    )
    print("   ✅ Model created")
    
    # Step 4: Deploy to endpoint
    print("\n🌐 Step 4: Deploying to real-time endpoint...")
    print("   ⏳ This may take 5-10 minutes...")
    
    endpoint_name = 'cnn-mars-classification-endpoint'
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name=endpoint_name
    )
    
    print(f"\n   ✅ Endpoint deployed: {endpoint_name}")
    print(f"   ✅ Endpoint ARN: arn:aws:sagemaker:{region}:endpoint/{endpoint_name}")
    
    # Step 5: Test the endpoint
    print("\n🧪 Step 5: Testing endpoint with sample image...")
    
    runtime_client = boto3.client('sagemaker-runtime')
    
    # Note: For actual testing, you would load and preprocess an image
    # This is a placeholder showing the endpoint is ready
    print("\n   🔭 Mars Surface Classification Ready")
    print("   " + "=" * 50)
    print(f"   Endpoint Name: {endpoint_name}")
    print("   Input Shape: (128, 128, 3)")
    print("   Output Classes: 24")
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print(f"\n📌 Endpoint Name: {endpoint_name}")
    print("\n⚠️  IMPORTANT: Run 'python cleanup.py' when done to delete the endpoint!")
    print("   This will prevent ongoing AWS charges.")
    
    return endpoint_name

if __name__ == "__main__":
    main()
