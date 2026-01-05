#!/usr/bin/env python3
"""
Test script to verify that model.pkl loads and predicts correctly.
Run this after executing model_usage.ipynb
"""

import pickle
import pandas as pd
import numpy as np

def test_model():
    print("=" * 60)
    print("TESTING MODEL LOADING AND PREDICTION")
    print("=" * 60)
    
    # 1. Load model
    try:
        with open('models/model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ ERROR: models/model.pkl not found!")
        print("   Run model_usage.ipynb first to create the model.")
        return False
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False
    
    # 2. Check model package contents
    required_keys = ['model', 'preprocessor', 'top_10_indices', 'feature_names']
    missing_keys = [key for key in required_keys if key not in model_package]
    
    if missing_keys:
        print(f"❌ ERROR: Missing keys in model package: {missing_keys}")
        return False
    
    print(f"✅ Model type: {model_package.get('model_type', 'Unknown')}")
    print(f"✅ R² score: {model_package.get('r2_score', 'Unknown')}")
    print(f"✅ Top 10 indices: {model_package['top_10_indices']}")
    
    # 3. Test prediction with sample data
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    # Create sample song data
    sample_song = pd.DataFrame({
        'danceability': [0.7],
        'acousticness': [0.2],
        'liveness': [0.1],
        'tempo': [120.0],
        'loudness': [-5.0],
        'speechiness': [0.05],
        'duration_ms': [200000],
        'instrumentalness': [0.0],
        'popularity': [50],
        'dynamic_range': [10.0],
        'valence': [0.6],
        'rhythmic_complexity': [0.5],
        'release_month': [6]
    })
    
    # Apply feature engineering (same as in notebooks)
    # Only valid interaction features (no target leakage, no data leakage)
    sample_song['loudness_tempo'] = sample_song['loudness'] * sample_song['tempo']
    sample_song['danceability_valence'] = sample_song['danceability'] * sample_song['valence']
    sample_song['loudness_danceability'] = sample_song['loudness'] * sample_song['danceability']
    sample_song['tempo_valence'] = sample_song['tempo'] * sample_song['valence']
    
    try:
        # Preprocess
        preprocessor = model_package['preprocessor']
        sample_processed = preprocessor.transform(sample_song)
        sample_top10 = sample_processed[:, model_package['top_10_indices']]
        
        # Predict
        model = model_package['model']
        prediction = model.predict(sample_top10)[0]
        
        print(f"\nSample prediction:")
        print(f"  Predicted Energy: {prediction:.4f}")
        
        # Check range
        if 0 <= prediction <= 1:
            print(f"✅ Prediction in valid range [0, 1]")
        else:
            print(f"⚠️  WARNING: Prediction outside valid range [0, 1]")
            print(f"   Consider clipping predictions")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model()
    exit(0 if success else 1)

