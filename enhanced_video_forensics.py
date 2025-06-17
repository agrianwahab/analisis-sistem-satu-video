# -*- coding: utf-8 -*-
"""
SISTEM DETEKSI FORENSIK KEASLIAN VIDEO - VERSI OPTIMIZED
Fokus pada Kecepatan dan Efisiensi dengan Batch Processing.

Arsitektur Analisis yang Dioptimalkan:
1. Perceptual Hashing (pHash): Deteksi duplikasi (cepat).
2. Structural Similarity (SSIM) & Peta Perbedaan: Analisis struktural (cepat).
3. Optical Flow: Analisis konsistensi gerakan (relatif cepat).
4. VGG16 Deep Features (BATCH PROCESSED): Analisis semantik dengan kecepatan tinggi.

Perbaikan Kunci:
- UNIFIKASI FRAMEWORK: Hanya menggunakan TensorFlow/Keras, menghapus PyTorch untuk efisiensi.
- BATCH PROCESSING: Fitur VGG16 diekstrak untuk SEMUA frame dalam satu panggilan `predict()`, 
  menghasilkan percepatan performa yang masif.
- ALGORITMA EFISIEN: Menggunakan perbandingan histogram (cepat) untuk deteksi scene dan 
  perbandingan kosinus (cepat) untuk konsistensi temporal, menggantikan ResNet/LSTM yang lambat.
- PENGHAPUSAN BOTTLENECK: Menghilangkan analisis noise frame-per-frame yang sangat lambat.
- VISUALISASI CANGGIH: Tetap mempertahankan Papan Bukti Visual 2x2 yang detail.
"""
import os
import cv2
import numpy as np
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import logging
from datetime import datetime

# --- KONFIGURASI ---
# Konfigurasi Logging untuk melacak proses
log_filename = f"forensics_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Import TensorFlow setelah mengecek environment untuk menghindari pesan yang tidak perlu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# --- FUNGSI UTILITAS (Cepat dan Efisien) ---

def mad(data):
    """Menghitung Median Absolute Deviation (MAD), metrik robust untuk variabilitas."""
    if len(data) == 0: return 0
    median = np.median(data)
    diff = np.abs(data - median)
    return np.median(diff)

def detect_scene_transitions(frames, threshold=0.55):
    """Metode deteksi scene change berbasis histogram yang ringan dan cepat."""
    scene_changes = {0}
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    
    for i in tqdm(range(1, len(gray_frames)), desc="Mendeteksi Scene Changes (Histogram)"):
        hist1 = cv2.calcHist([gray_frames[i-1]], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([gray_frames[i]], [0], None, [256], [0,256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        if hist_diff > threshold:
            scene_changes.add(i-1)
    logging.info(f"Ditemukan {len(scene_changes)-1} potensi scene change alami untuk diabaikan.")
    return scene_changes

# --- FUNGSI MODEL (Hanya TensorFlow/Keras) ---

def load_vgg_model():
    """Memuat model VGG16 pre-trained sekali saja."""
    logging.info("Memuat model VGG16 (TensorFlow/Keras)...")
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    logging.info("Model VGG16 berhasil dimuat.")
    return model

# --- FUNGSI UTAMA ANALISIS (Dengan Optimisasi) ---

def analyze_video_optimized(video_path, vgg_model, resize_dim=(320, 240), batch_size=32):
    if not os.path.exists(video_path):
        logging.error(f"Error: File tidak ditemukan di '{video_path}'")
        return None
    
    logging.info(f"MEMULAI ANALISIS OPTIMIZED UNTUK: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        logging.error(f"Error: Tidak dapat membaca frame dari video '{video_path}'.")
        cap.release()
        return None
    
    # --- TAHAP 1: Ekstraksi Frame dan Fitur Dasar (Cepat) ---
    frames_rgb = []
    frame_hashes = []
    
    for _ in tqdm(range(total_frames), desc="Tahap 1: Ekstraksi Frame & pHash"):
        ret, frame = cap.read()
        if not ret: break
        resized_frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
        pil_image = Image.fromarray(frame_rgb)
        frame_hashes.append(imagehash.phash(pil_image))
    cap.release()

    if len(frames_rgb) < 2:
        logging.error("Video terlalu pendek untuk dianalisis (kurang dari 2 frame).")
        return None

    scene_changes = detect_scene_transitions(frames_rgb)

    # --- TAHAP 2 (OPTIMIZED): Ekstraksi Fitur VGG16 secara Batch ---
    # Ini adalah optimisasi paling signifikan.
    logging.info("Mempersiapkan batch untuk VGG16...")
    
    # Siapkan batch gambar yang akan diproses
    image_batch_for_vgg = []
    for frame in tqdm(frames_rgb, desc="Mempersiapkan VGG16 Batch"):
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img_array_expanded = np.expand_dims(img, axis=0)
        preprocessed_img = preprocess_input(img_array_expanded)
        image_batch_for_vgg.append(preprocessed_img)
        
    # Tumpuk semua gambar menjadi satu array numpy besar
    vgg_input_batch = np.vstack(image_batch_for_vgg)
    
    # Jalankan prediksi HANYA SEKALI pada seluruh batch
    logging.info(f"Mengekstrak fitur VGG16 untuk {len(vgg_input_batch)} frame (Batch Size: {batch_size})...")
    vgg_features_list = vgg_model.predict(vgg_input_batch, batch_size=batch_size, verbose=1)
    logging.info("Ekstraksi fitur VGG16 secara batch selesai.")

    # --- TAHAP 3: Kalkulasi Metrik Antar Frame (Sequential) ---
    hamming_dists = []
    ssim_scores = []
    ssim_diff_maps = {} 
    flow_mags = []
    vgg_similarities = []
    
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]

    for i in tqdm(range(1, len(frames_rgb)), desc="Tahap 3: Kalkulasi Metrik Antar Frame"):
        # pHash
        hamming_dists.append(frame_hashes[i-1] - frame_hashes[i])
        
        # SSIM (dengan Peta Perbedaan)
        score, diff = ssim(frames_gray[i-1], frames_gray[i], full=True)
        ssim_scores.append(score)
        ssim_diff_maps[i-1] = diff

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(frames_gray[i-1], frames_gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(np.mean(magnitude))
        
        # VGG16 Cosine Similarity (menggunakan fitur yang sudah diekstrak)
        feat1, feat2 = vgg_features_list[i-1], vgg_features_list[i]
        norm1, norm2 = np.linalg.norm(feat1), np.linalg.norm(feat2)
        cosine_sim = np.dot(feat1, feat2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        vgg_similarities.append(cosine_sim)

    # --- TAHAP 4: Analisis dan Deteksi Anomali ---
    logging.info("Tahap 4: Menganalisis anomali dan menggabungkan skor...")
    ssim_anomaly = 1 - np.array(ssim_scores)
    vgg_anomaly = 1 - np.array(vgg_similarities)
    flow_anomaly = np.array(flow_mags)

    scaler = RobustScaler()
    combined_metrics = np.column_stack([ssim_anomaly, vgg_anomaly, flow_anomaly])
    scaled_metrics = scaler.fit_transform(combined_metrics)
    combined_anomaly_score = np.mean(scaled_metrics, axis=1)
    
    results = {
        'video_file': os.path.basename(video_path),
        'anomalies': {},
        'duplicate_frames': [],
        'total_transitions': len(frames_rgb) - 1
    }
    
    for i, dist in enumerate(hamming_dists):
        if dist <= 1: 
            results['duplicate_frames'].append((i, i + 1))
            
    score_median = np.median(combined_anomaly_score)
    score_mad = mad(combined_anomaly_score)
    anomaly_threshold = score_median + 4.0 * score_mad if score_mad > 1e-5 else score_median + 1.0

    suspicious_indices = np.where(combined_anomaly_score > anomaly_threshold)[0]
    for idx in suspicious_indices:
        if idx not in scene_changes:
            results['anomalies'][idx] = {
                'score': combined_anomaly_score[idx],
                'ssim': ssim_scores[idx],
                'flow': flow_mags[idx],
                'vgg_sim': vgg_similarities[idx]
            }

    # --- TAHAP 5: Pelaporan ---
    logging.info(f"--- Laporan Analisis Forensik Optimized ---")
    logging.info(f"Video: {results['video_file']}")
    logging.info(f"Total Transisi Frame Dianalisis: {results['total_transitions']}")
    
    if results['duplicate_frames']:
        logging.warning(f"DUPLIKASI FRAME TERDETEKSI: {len(results['duplicate_frames'])} kasus ditemukan.")
    else:
        logging.info("Tidak ada indikasi duplikasi frame.")
        
    if not results['anomalies']:
        logging.info("TIDAK DITEMUKAN DISKONTINUITAS SIGNIFIKAN (indikasi penghapusan/penyisipan).")
    else:
        logging.warning(f"POTENSI DISKONTINUITAS (DELETION/INSERTION) TERDETEKSI: {len(results['anomalies'])} titik.")
        sorted_anomalies = sorted(results['anomalies'].items(), key=lambda item: item[1]['score'], reverse=True)
        for idx, details in sorted_anomalies[:5]:
            logging.warning(f"  - Transisi Frame {idx} -> {idx+1} [Skor Anomali Gabungan: {details['score']:.2f}] (SSIM={details['ssim']:.2f}, Flow={details['flow']:.2f}, VGG-Sim={details['vgg_sim']:.2f})")
    
    # --- TAHAP 6: Visualisasi ---
    plot_ultimate_dashboard(results, combined_anomaly_score, ssim_scores, flow_mags, vgg_similarities, frames_rgb, anomaly_threshold, ssim_diff_maps)
    
    return results

def plot_ultimate_dashboard(results, combined_score, ssim_s, flow_s, vgg_s, frames_rgb, threshold, ssim_diff_maps):
    """Membuat dashboard visualisasi canggih dengan Papan Bukti 2x2."""
    fig = plt.figure(figsize=(20, 22))
    gs_main = gridspec.GridSpec(3, 1, height_ratios=[1, 0.8, 2.5], hspace=0.4)
    fig.suptitle(f'Dashboard Analisis Forensik Optimized: {results["video_file"]}', fontsize=24, weight='bold')

    # Bagian 1: Plot Utama Skor Anomali
    ax_main = fig.add_subplot(gs_main[0])
    ax_main.plot(combined_score, label='Skor Anomali Gabungan', color='red', linewidth=2, zorder=2)
    ax_main.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold Anomali Adaptif ({threshold:.2f})', zorder=3)
    anom_indices = list(results['anomalies'].keys())
    if anom_indices:
        anom_scores = [results['anomalies'][i]['score'] for i in anom_indices]
        ax_main.scatter(anom_indices, anom_scores, color='blue', s=100, zorder=4, label='Anomali Terdeteksi')
    ax_main.set_title('HASIL UTAMA: Deteksi Diskontinuitas Temporal', fontsize=18, weight='bold')
    ax_main.set_xlabel('Indeks Transisi Frame')
    ax_main.set_ylabel('Tingkat Anomali (Robust Scaled)')
    ax_main.legend()
    ax_main.grid(True, linestyle=':')

    # Bagian 2: Plot Pilar Analisis
    gs_pillars = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.3)
    ax_ssim, ax_flow, ax_vgg = [fig.add_subplot(gs_pillars[i]) for i in range(3)]
    ax_ssim.plot(ssim_s, 'g-'); ax_ssim.set_title('Pilar 1: SSIM'); ax_ssim.set_ylim(0, 1.05)
    ax_flow.plot(flow_s, 'b-'); ax_flow.set_title('Pilar 2: Optical Flow Mag.')
    ax_vgg.plot(vgg_s, 'm-'); ax_vgg.set_title('Pilar 3: VGG16 Sim.'); ax_vgg.set_ylim(0, 1.05)
    for ax in [ax_ssim, ax_flow, ax_vgg]:
        ax.grid(True, linestyle=':'); ax.set_xlabel('Indeks Transisi')

    # Bagian 3: Papan Bukti Visual 2x2
    gs_evidence = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[2], wspace=0.1, hspace=0.3)
    axes_evidence = [fig.add_subplot(gs_evidence[i, j]) for i in range(2) for j in range(2)]
    fig.text(0.5, gs_main[2].get_position(fig).y1 - 0.01, 'PAPAN BUKTI VISUAL (ANOMALI TERTINGGI)', ha='center', va='bottom', fontsize=16, weight='bold')

    if results['anomalies']:
        most_suspicious_idx = max(results['anomalies'], key=lambda k: results['anomalies'][k]['score'])
        details = results['anomalies'][most_suspicious_idx]
        frame_before = frames_rgb[most_suspicious_idx]
        frame_after = frames_rgb[most_suspicious_idx + 1]

        # Panel 1: Frame SEBELUM
        axes_evidence[0].imshow(frame_before)
        axes_evidence[0].set_title(f'1. Frame SEBELUM Anomali ({most_suspicious_idx})', fontsize=14)
        
        # Panel 2: Frame SETELAH
        axes_evidence[1].imshow(frame_after)
        axes_evidence[1].set_title(f'2. Frame SETELAH Anomali ({most_suspicious_idx + 1})', fontsize=14)
        
        # Panel 3: Peta Perbedaan Struktural (SSIM)
        im = axes_evidence[2].imshow(ssim_diff_maps[most_suspicious_idx], cmap='viridis')
        axes_evidence[2].set_title('3. Peta Perbedaan Struktural (SSIM)', fontsize=14)
        fig.colorbar(im, ax=axes_evidence[2], orientation='horizontal', fraction=0.046, pad=0.08).set_label('Tingkat Perbedaan')

        # Panel 4: Visualisasi Optical Flow
        gray_before = cv2.cvtColor(frame_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(frame_after, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_before, gray_after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(frame_before)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        axes_evidence[3].imshow(flow_viz)
        axes_evidence[3].set_title('4. Visualisasi Aliran Optik (Gerakan)', fontsize=14)

        for ax in axes_evidence: ax.axis('off')

        summary_text = (f"BUKTI DISKONTINUITAS TERTINGGI: Transisi Frame {most_suspicious_idx} -> {most_suspicious_idx + 1}\n"
                        f"Skor Anomali Gabungan: {details['score']:.2f} | SSIM: {details['ssim']:.3f} | Optical Flow: {details['flow']:.2f} | VGG Sim: {details['vgg_sim']:.3f}")
        fig.text(0.5, 0.92, summary_text, ha='center', fontsize=15, 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='black', lw=1, alpha=0.9))
    else:
        fig.text(0.5, 0.25, "Tidak ada anomali signifikan untuk divisualisasikan.", ha='center', va='center', fontsize=18)
        for ax in axes_evidence: ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"dashboard_forensik_optimized_{os.path.splitext(results['video_file'])[0]}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Dashboard analisis visual disimpan sebagai '{output_filename}'")
    plt.show()

# --- BLOK EKSEKUSI UTAMA ---
if __name__ == '__main__':
    # Memuat model VGG16 sekali saja untuk efisiensi
    vgg_model = load_vgg_model()
    
    original_video_path = 'original.mp4'
    # Pastikan nama file ini benar, 'frame_deleteion.mp4' memiliki typo. Seharusnya 'frame_deletion.mp4'
    tampered_video_path = 'frame_deletion.mp4' 

    if not os.path.exists(tampered_video_path):
        # Coba perbaiki typo umum
        if os.path.exists('frame_deleteion.mp4'):
            tampered_video_path = 'frame_deleteion.mp4'
            logging.warning("Menggunakan file 'frame_deleteion.mp4' karena 'frame_deletion.mp4' tidak ditemukan.")

    if not os.path.exists(original_video_path) or not os.path.exists(tampered_video_path):
        logging.critical("Satu atau kedua file video contoh tidak ditemukan. Proses dibatalkan.")
    else:
        # Analisis video asli
        analyze_video_optimized(original_video_path, vgg_model)
        
        # Analisis video yang dimanipulasi
        analyze_video_optimized(tampered_video_path, vgg_model)

    logging.info(f"Analisis selesai. Log lengkap tersimpan di '{log_filename}'")