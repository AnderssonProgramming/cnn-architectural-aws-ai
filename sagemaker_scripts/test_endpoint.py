#!/usr/bin/env python3
"""
CNN Mars Classification - Test Endpoint Script
===============================================

⚠️  REQUIRES: Active deployed endpoint.
    Run deploy.py first, or use demo_training.py for local testing.

Run from SageMaker Code Editor terminal:
    python test_endpoint.py
"""

import boto3
import json
import numpy as np
from PIL import Image
import os

# Mars terrain class names
CLASS_NAMES = [
    'arm cover', 'arm', 'calibration target', 'chassis',
    'drill', 'drt front', 'drt side', 'ground',
    'horizon', 'inlet', 'mahli', 'mastcam',
    'observation tray', 'portion box', 'portion tube', 'rear hazcam',
    'scoop', 'sun', 'turret', 'wheel'
]

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image for prediction."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_data, endpoint_name='cnn-mars-classification-endpoint'):
    """Send a prediction request to the deployed endpoint."""
    runtime_client = boto3.client('sagemaker-runtime')
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({"instances": image_data.tolist()})
    )
    
    return json.loads(response['Body'].read().decode())

def main():
    print("=" * 70)
    print("🔭 CNN MARS CLASSIFICATION - ENDPOINT TEST")
    print("=" * 70)
    
    # Define test images (paths relative to project root)
    test_images = [
        "msl-images/calibrated/0025ML0001270000100800C00_DRCL.jpg",
        "msl-images/calibrated/0025MR0001160000100715C00_DRCL.jpg",
        "msl-images/calibrated/0036MH0000290010100072C00_DRCL.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n📷 Testing: {os.path.basename(image_path)}")
        print("-" * 50)
        
        if not os.path.exists(image_path):
            print(f"   ⚠️  Image not found: {image_path}")
            continue
        
        try:
            # Load and preprocess image
            image_data = load_and_preprocess_image(image_path)
            
            # Get prediction
            result = predict_image(image_data)
            predictions = np.array(result['predictions'][0])
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            
            print(f"   🏆 Top 3 Predictions:")
            for i, idx in enumerate(top_3_idx):
                class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                confidence = predictions[idx] * 100
                print(f"      {i+1}. {class_name}: {confidence:.2f}%")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
