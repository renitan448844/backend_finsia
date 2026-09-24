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

# Path ke model (Gunakan absolute path agar Vercel Serverless bisa menemukan file .pkl)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_model.pkl')

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
except Exception as e:
    print(f"❌ Error loading model: {e}")
    kmeans = None
    scaler = None


cluster_remap = {
    2: 0,   # Hemat
    1: 1,   # Stabil
    0: 2    # Konsumtif
}

label_mapping = {
    0: 'Hemat',
    1: 'Stabil',
    2: 'Konsumtif'
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
        'recommendation': 'Pertahankan keseimbangan ini. Coba alokasikan uangmu untuk menabung.'
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
    
    if kmeans is None or scaler is None:
        raise ValueError("Model Machine Learning belum berhasil di-load di server.")

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
            '/predict': 'GET/POST - Prediksi cluster',
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
    if kmeans is None:
        return jsonify({
            'status': 'error',
            'message': 'Model Machine Learning belum berhasil di-load'
        }), 500

    return jsonify({
        'status': 'success',
        'data': {
            'n_clusters': kmeans.n_clusters,
            'n_features': getattr(kmeans, 'n_features_in_', None),
            'labels': label_mapping
        }
    })

@app.route('/predict', methods=['GET', 'POST']) # Mendukung GET untuk tes langsung di Browser Tab
def predict():
    """Endpoint untuk prediksi cluster"""
    try:
        if request.method == 'GET':
            # Mengambil parameter input dari alamat URL (Query String)
            data = {
                'total_pengeluaran': request.args.get('total_pengeluaran', default=0.0, type=float),
                'jumlah_transaksi': request.args.get('jumlah_transaksi', default=0.0, type=float),
                'persen_kebutuhan': request.args.get('persen_kebutuhan', default=0.0, type=float),
                'persen_hiburan': request.args.get('persen_hiburan', default=0.0, type=float),
                'persen_belanja': request.args.get('persen_belanja', default=0.0, type=float)
            }
        else:
            # Mengambil parameter input dari Flutter / JSON Body (POST)
            data = request.get_json()
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'Request body tidak ditemukan'
                }), 400
        
        # Prediksi
        result = predict_cluster(data)
        
        # Kembalikan respon
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
    # Ambil port dari environment (Render/Hostinger)
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "=" * 60)
    print("🚀 STARTING FLASK SERVER")
    print("=" * 60)
    print(f"📍 Server running on port: {port}")
    print(f"📊 Health check: /health")
    print(f"🔮 Predict endpoint: /predict")
    print(f"ℹ️  Info model: /info")
    print("=" * 60)
    print("\n📌 KETIK Ctrl+C UNTUK STOP SERVER\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )