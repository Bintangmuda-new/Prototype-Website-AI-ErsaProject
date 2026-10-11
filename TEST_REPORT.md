# TEST REPORT — Prototype.Website.AI.ErsaProject (versi Streamlit)

Tanggal: 2026-10-11 · Penguji: subagent porting Gradio→Streamlit

## Ringkasan
Port aplikasi Gradio ke Streamlit selesai. Server Streamlit berjalan dan
merespons HTTP 200 tanpa error. Inferensi model YOLOv8n nyata diuji langsung
pada 5 citra test dataset Corrosion Detection: **5/5 terdeteksi Rust dengan
confidence > 0.5**. Semua angka/metrik berasal dari file nyata
(`models/best.pt`, `info_model.json`); tidak ada data karangan.

## File yang dibuat (~/workspace/skripsi_anoa/website_streamlit/)
- `streamlit_app.py` — aplikasi Streamlit, Bahasa Indonesia, judul
  "Prototype.Website.AI.ErsaProject". Fitur: Deteksi (upload + kamera,
  slider confidence 0.1–0.9 default 0.5, gambar ber-anotasi bbox, tabel temuan,
  rekomendasi, unduh JPEG), Batch (multi-upload, ringkasan per gambar, unduh
  CSV), Riwayat (log sesi + tombol hapus), Tentang (metrik nyata dari
  info_model.json, atribusi CC BY 4.0, disclaimer). Navigasi via sidebar radio.
- `requirements.txt` — streamlit==1.65.0, ultralytics==8.4.175,
  torch==2.14.1+cpu (--extra-index-url PyTorch CPU), torchvision==0.29.1,
  opencv-python-headless==5.0.0.93, numpy==2.5.4, pillow==12.3.0
- `models/best.pt` (6,2 MB, md5 44b570fdb37937645048119a1240c52a — identik
  dengan website/models/best.pt)
- `info_model.json` (salinan, berisi metrik nyata)
- `.streamlit/config.toml` (tema gelap + headless)

## Bukti uji
1. **Server**: `streamlit run streamlit_app.py --server.headless true`
   → `curl http://127.0.0.1:8501/` = **HTTP 200**; log server bersih,
   tanpa error/traceback.
2. **Inferensi nyata** (venv website, model models/best.pt, conf=0.5):
   | Citra test nyata | Temuan |
   |---|---|
   | corrosion_39_...jpg | Rust 92,2% |
   | corrosion_63_...jpg | Rust 81,0% |
   | corrosion_90_...jpg | Rust 86,7% |
   | rusted_airplane_5_...jpg | Rust 69,5% |
   | rusted_car_19_...jpg | Rust 91,0% |
   → 5/5 citra ada temuan Rust dengan confidence > 0.5. **LULUS.**
3. **Logika aplikasi**: fungsi `analisis()` dan `rekomendasi()` identik dengan
   versi Gradio yang sudah lolos smoke test; metrik di tab Tentang dibaca
   langsung dari info_model.json (precision 85,84%, recall 91,43%,
   mAP50 89,70% — angka nyata hasil training).

## Belum terverifikasi
- Interaksi UI end-to-end di browser (klik tombol Deteksi via UI web)
  tidak diuji — subagent ini tidak memiliki akses browser. Disarankan uji
  manual sekali setelah deploy ke Streamlit Community Cloud.
- Instalasi bersih dari requirements.txt di Streamlit Cloud belum dijalankan
  (butuh akun & repo GitHub pemilik). Versi di-pin selaras dengan venv lokal
  yang terbukti jalan.

## Langkah deploy (untuk pemilik)
1. Push folder `website_streamlit/` ke repo GitHub (atau repo baru).
2. Di share.streamlit.io → New app → pilih repo, branch, main file:
   `streamlit_app.py` → Deploy. Gratis, tanpa kartu kredit.
