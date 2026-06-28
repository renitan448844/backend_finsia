# ============================================
# BACKEND API UNTUK PREDIKSI POLA KEUANGAN
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json

# ============================================
# INISIALISASI FLASK
# ============================================

app = Flask(__name__)
CORS(app)  # Izinkan akses dari Flutter

# ============================================
# LOAD MODEL
# ============================================

# Path ke model
MODEL_PATH = 'models/kmeans_model.pkl'
SCALER_PATH = 'models/scaler_model.pkl'

# Load model
print("\n" + "=" * 60)
print("🚀 LOADING MODEL...")
print("=" * 60)

try:
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model berhasil di-load!")
    print(f"   K-Means: {kmeans}")
    print(f"   Scaler: {scaler}")
except FileNotFoundError as e:
    print(f"❌ File tidak ditemukan: {e}")
    print("   Pastikan file .pkl ada di folder models/")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ============================================
# LABEL MAPPING
# ============================================

label_mapping = {
    0: 'Hemat',
    1: 'Normal/Stabil',
    2: 'Konsumtif/Boros'
}

print(f"\n📊 Label Mapping: {label_mapping}")

# ============================================
# INSIGHTS PER CLUSTER
# ============================================

INSIGHTS = {
    'Hemat': {
        'emoji': '💪',
        'insight': 'Pengeluaran Anda tergolong HEMAT! Pertahankan pola ini!',
        'recommendation': 'Terus pertahankan kebiasaan baik ini. Anda sudah di jalur yang benar!'
    },
    'Normal/Stabil': {
        'emoji': '😊',
        'insight': 'Pengeluaran Anda tergolong STABIL! Cukup baik.',
        'recommendation': 'Pertahankan keseimbangan ini. Coba alokasikan 10% untuk tabungan.'
    },
    'Konsumtif/Boros': {
        'emoji': '⚠️',
        'insight': 'Pengeluaran Anda tergolong KONSUMTIF! Perlu dievaluasi.',
        'recommendation': 'Coba buat anggaran bulanan dan prioritaskan kebutuhan daripada keinginan.'
    }
}

# ============================================
# FUNGSI PREDIKSI
# ============================================

def predict_cluster(data):
    """Prediksi cluster dari data transaksi bulanan"""
    
    # Validasi input
    required = ['total_pengeluaran', 'jumlah_transaksi', 
                'persen_kebutuhan', 'persen_hiburan', 'persen_belanja']
    
    for key in required:
        if key not in data:
            raise ValueError(f"Field '{key}' tidak ditemukan")
    
    # Konversi ke array
    features = np.array([[
        float(data['total_pengeluaran']),
        float(data['jumlah_transaksi']),
        float(data['persen_kebutuhan']),
        float(data['persen_hiburan']),
        float(data['persen_belanja'])
    ]])
    
    # Scaling
    features_scaled = scaler.transform(features)
    
    # Prediksi cluster
    cluster = kmeans.predict(features_scaled)[0]
    label = label_mapping[cluster]
    
    # Hitung jarak ke centroid
    distances = kmeans.transform(features_scaled)[0]
    
    # Konversi jarak ke probabilitas
    probabilities = 1 / (1 + distances)
    probabilities = probabilities / probabilities.sum()
    
    # Mapping probabilitas
    prob_dict = {}
    for i, label_key in enumerate(label_mapping.values()):
        prob_dict[label_key] = float(probabilities[i])
    
    # Ambil insight
    insight_data = INSIGHTS.get(label, {
        'emoji': '📊',
        'insight': f'Pengeluaran Anda masuk kategori {label}',
        'recommendation': 'Terus pantau pengeluaran Anda.'
    })
    
    return {
        'cluster': int(cluster),
        'label': label,
        'emoji': insight_data['emoji'],
        'insight': insight_data['insight'],
        'recommendation': insight_data['recommendation'],
        'probabilities': prob_dict,
        'features': data
    }

# ============================================
# ENDPOINT API
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'API Prediksi Pola Keuangan',
        'version': '1.0.0',
        'model_loaded': True,
        'endpoints': {
            '/': 'GET - Home',
            '/health': 'GET - Cek status server',
            '/predict': 'POST - Prediksi cluster',
            '/info': 'GET - Info model'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Server is running',
        'model_loaded': kmeans is not None,
        'scaler_loaded': scaler is not None,
        'n_clusters': kmeans.n_clusters if kmeans else None,
        'label_mapping': label_mapping
    })

@app.route('/info', methods=['GET'])
def info():
    """Info model endpoint"""
    return jsonify({
        'status': 'success',
        'data': {
            'n_clusters': kmeans.n_clusters,
            'n_features': kmeans.n_features_in_,
            'labels': label_mapping
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi cluster"""
    
    try:
        # Ambil data dari request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body tidak ditemukan'
            }), 400
        
        # Prediksi
        result = predict_cluster(data)
        
        # Return response
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Validation error: {str(e)}'
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

# ============================================
# ERROR HANDLER
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint tidak ditemukan'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Terjadi kesalahan pada server'
    }), 500

# ============================================
# JALANKAN SERVER
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 STARTING FLASK SERVER")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📊 Health check: http://localhost:5000/health")
    print(f"🔮 Predict endpoint: http://localhost:5000/predict")
    print(f"ℹ️  Info model: http://localhost:5000/info")
    print("=" * 60)
    print("\n📌 KETIK Ctrl+C UNTUK STOP SERVER\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)