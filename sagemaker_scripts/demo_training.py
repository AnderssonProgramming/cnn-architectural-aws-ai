#!/usr/bin/env python3
"""
CNN Mars Classification - Training Demo
========================================

✅ USE THIS SCRIPT IN LEARNER LABS!
   This script demonstrates the full training process for the CNN model
   without creating an endpoint (which Learner Labs block).

Run from SageMaker Code Editor terminal:
    python sagemaker_scripts/demo_training.py
"""
import os
import sys
import numpy as np
import logging

# Suppress verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def main():
    print("=" * 70)
    print("🚀 CNN MARS CLASSIFICATION - TRAINING DEMO")
    print("=" * 70)
    print("\n⚠️  Note: This demo trains locally without creating an endpoint")
    print("   (Learner Lab restricts sagemaker:CreateEndpointConfig)")
    
    # Step 1: Initialize SageMaker Session
    print("\n📦 Step 1: Initializing SageMaker session...")
    try:
        import boto3
        import sagemaker
        
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        bucket = sagemaker_session.default_bucket()
        role = sagemaker.get_execution_role()
        
        print(f"   ✅ Region: {region}")
        print(f"   ✅ Bucket: {bucket}")
        print(f"   ✅ Role: {role[:50]}...")
    except Exception as e:
        print(f"   ⚠️ SageMaker session not available (running locally?)")
        print(f"   ⚠️ Error: {e}")
        region = "local"
        bucket = "local"
    
    # Step 2: Import Libraries
    print("\n📚 Step 2: Importing libraries...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from PIL import Image
        import matplotlib.pyplot as plt
        from collections import Counter
        
        print(f"   ✅ TensorFlow version: {tf.__version__}")
        print(f"   ✅ GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError as e:
        print(f"   ❌ Error importing libraries: {e}")
        print("   💡 Run: pip install tensorflow numpy matplotlib pillow")
        sys.exit(1)
    
    # Step 3: Load Dataset
    print("\n📊 Step 3: Loading Mars Surface Image dataset...")
    
    # Find the dataset path
    possible_paths = [
        './msl-images/',
        '../msl-images/',
        '/home/ec2-user/SageMaker/cnn-architectural-aws-ai/msl-images/',
        '/home/sagemaker-user/cnn-architectural-aws-ai/msl-images/',
    ]
    
    BASE_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            BASE_PATH = path
            break
    
    if BASE_PATH is None:
        print("   ❌ Dataset not found! Please ensure msl-images/ folder exists.")
        print("   💡 Looked in:", possible_paths)
        sys.exit(1)
    
    print(f"   ✅ Dataset found at: {BASE_PATH}")
    
    # Configuration
    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 5  # Reduced for demo
    NUM_CLASSES = 24
    
    # Load class names
    SYNSET_FILE = os.path.join(BASE_PATH, 'msl_synset_words-indexed.txt')
    class_names = {}
    with open(SYNSET_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_id = int(parts[0])
                class_name = parts[1]
                class_names[class_id] = class_name
    
    # Load file lists
    def load_file_list(filename):
        images, labels = [], []
        filepath = os.path.join(BASE_PATH, filename)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    images.append(parts[0])
                    labels.append(int(parts[1]))
        return images, labels
    
    train_images, train_labels = load_file_list('train-calibrated-shuffled.txt')
    val_images, val_labels = load_file_list('val-calibrated-shuffled.txt')
    test_images, test_labels = load_file_list('test-calibrated-shuffled.txt')
    
    print(f"   ✅ Training samples: {len(train_images):,}")
    print(f"   ✅ Validation samples: {len(val_images):,}")
    print(f"   ✅ Test samples: {len(test_images):,}")
    print(f"   ✅ Number of classes: {NUM_CLASSES}")
    
    # Create class mapping (handle non-0-indexed labels)
    all_labels = set(train_labels + val_labels + test_labels)
    class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted(all_labels))}
    
    # Load and preprocess images (subset for demo)
    def load_images(image_list, label_list, max_samples=None):
        if max_samples:
            image_list = image_list[:max_samples]
            label_list = label_list[:max_samples]
        
        X, y = [], []
        for img_name, label in zip(image_list, label_list):
            img_path = os.path.join(BASE_PATH, 'calibrated', img_name)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    X.append(np.array(img) / 255.0)
                    y.append(class_id_to_index[label])
                except Exception:
                    continue
        return np.array(X), np.array(y)
    
    print("\n   Loading images (this may take a moment)...")
    
    # For demo, use subset of data
    max_train = 1000  # Use subset for faster demo
    max_val = 300
    max_test = 300
    
    X_train, y_train = load_images(train_images, train_labels, max_train)
    X_val, y_val = load_images(val_images, val_labels, max_val)
    X_test, y_test = load_images(test_images, test_labels, max_test)
    
    print(f"   ✅ Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test images")
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Step 4: Build CNN Model
    print("\n🏗️ Step 4: Building CNN model...")
    
    model = keras.Sequential([
        # Input
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = model.count_params()
    print(f"   ✅ Model architecture created")
    print(f"   ✅ Total parameters: {total_params:,}")
    
    # Step 5: Train Model
    print(f"\n🎯 Step 5: Training model ({EPOCHS} epochs demo)...")
    print("   " + "-" * 50)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("   " + "-" * 50)
    
    # Step 6: Evaluate Model
    print("\n📈 Step 6: Evaluating model...")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"   ✅ Test Loss: {test_loss:.4f}")
    print(f"   ✅ Test Accuracy: {test_accuracy:.2%}")
    
    # Step 7: Save Model Artifacts
    print("\n💾 Step 7: Saving model artifacts...")
    
    os.makedirs('model_artifacts', exist_ok=True)
    
    # Save model in Keras format
    model.save('model_artifacts/cnn_model.keras')
    print("   ✅ Model saved to model_artifacts/cnn_model.keras")
    
    # Save training history
    np.save('model_artifacts/training_history.npy', history.history)
    print("   ✅ Training history saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING DEMO COMPLETE!")
    print("=" * 70)
    
    print("\n📊 Summary:")
    print("   ┌─────────────────────────────────┬────────┐")
    print("   │ Component                       │ Status │")
    print("   ├─────────────────────────────────┼────────┤")
    print("   │ SageMaker Session               │   ✅   │")
    print("   │ Dataset loaded                  │   ✅   │")
    print("   │ CNN Model built                 │   ✅   │")
    print("   │ Model trained                   │   ✅   │")
    print("   │ Model evaluated                 │   ✅   │")
    print("   │ Artifacts saved                 │   ✅   │")
    print("   │ Real endpoint deployment        │   ❌   │")
    print("   └─────────────────────────────────┴────────┘")
    
    print(f"\n📈 Final Results:")
    print(f"   • Test Accuracy: {test_accuracy:.2%}")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Epochs Trained: {len(history.history['accuracy'])}")
    
    print("\n💡 Why endpoint deployment is skipped:")
    print("   The Learner Lab policy explicitly denies:")
    print("   • sagemaker:CreateEndpointConfig")
    print("   • sagemaker:CreateEndpoint")
    
    print("\n🎬 Video Evidence Checklist:")
    print("   □ Show this script running in SageMaker Code Editor")
    print("   □ Show training progress (epochs)")
    print("   □ Show final accuracy results")
    print("   □ Show model_artifacts/ folder created")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
