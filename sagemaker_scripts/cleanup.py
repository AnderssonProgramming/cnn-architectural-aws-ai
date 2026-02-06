#!/usr/bin/env python3
"""
CNN Mars Classification - Cleanup Script
=========================================

⚠️  Run this IMMEDIATELY after testing to avoid AWS charges!

Run from SageMaker Code Editor terminal:
    python cleanup.py
"""

import boto3

def main():
    print("=" * 70)
    print("🧹 CLEANUP - DELETE SAGEMAKER RESOURCES")
    print("=" * 70)
    
    endpoint_name = 'cnn-mars-classification-endpoint'
    sagemaker_client = boto3.client('sagemaker')
    
    # Delete endpoint
    print(f"\n🗑️  Deleting endpoint: {endpoint_name}...")
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Endpoint deleted")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Delete endpoint configuration
    print(f"\n🗑️  Deleting endpoint configuration...")
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"   ✅ Endpoint configuration deleted")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 70)
    print("\n💰 No more charges will be incurred for this endpoint.")

if __name__ == "__main__":
    main()
