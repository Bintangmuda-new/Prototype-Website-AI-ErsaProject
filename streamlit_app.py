"""
Prototype.Website.AI.ErsaProject (versi Streamlit)
Sistem Deteksi Korosi pada Komponen Logam Replika Panser Anoa
Berbasis Computer Vision (YOLOv8) — Skripsi Mayor Ersa.

Port dari aplikasi Gradio (website/app.py) agar bisa di-deploy gratis
di Streamlit Community Cloud.

Menjalankan lokal:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy (Streamlit Community Cloud):
    Hubungkan repo GitHub -> pilih file utama: streamlit_app.py

Model: models/best.pt (atau via env MODEL_PATH).
Metrik: info_model.json (angka nyata hasil training).
"""

import csv
import datetime
import io
import json
import os
import tempfile

os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", "/tmp/ultralytics-settings")

import streamlit as st
from PIL import Image

PROJECT_NAME = "Prototype.Website.AI.ErsaProject"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt")
)
INFO_PATH = os.path.join(BASE_DIR, "info_model.json")

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🔍",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Pemuatan model (dengan penanganan error yang jelas)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def muat_model():
    from ultralytics import YOLO

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}. "
            "Taruh file best.pt hasil training ke folder models/."
        )
    mdl = YOLO(MODEL_PATH)
    return mdl


model = None
model_error = ""
try:
    model = muat_model()
except Exception as e:  # noqa: BLE001
    model_error = str(e)

info_model = {}
if os.path.exists(INFO_PATH):
    try:
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            info_model = json.load(f)
    except Exception as e:  # noqa: BLE001
        st.warning(f"info_model.json tidak bisa dibaca: {e}")


# ----------------------------------------------------------------------------
# Utilitas (logika identik dengan versi Gradio)
# ----------------------------------------------------------------------------
def rekomendasi(nama_kelas: str) -> str:
    """Rekomendasi tindak lanjut per kelas temuan."""
    k = nama_kelas.lower()
    if any(w in k for w in ["rust", "karat", "korosi", "corrosion"]):
        return (
            "Amplas area karat hingga bersih, aplikasikan cairan anti-karat "
            "(rust converter), lalu cat ulang dengan cat anti-korosi."
        )
    if any(w in k for w in ["retak", "crack"]):
        return "Periksa kedalaman retak; lakukan pengelasan/penambalan bila struktural."
    if any(w in k for w in ["penyok", "dent"]):
        return "Perbaiki dengan teknik body repair (ketok magic/dempul)."
    if any(w in k for w in ["cat", "paint", "kelupas", "peel"]):
        return "Bersihkan permukaan, dempul bila perlu, lalu cat ulang."
    if any(w in k for w in ["hilang", "missing", "longgar", "loose"]):
        return "Ganti komponen yang hilang; kencangkan baut pengikat."
    return "Lakukan pemeriksaan lanjutan oleh teknisi."


def analisis(gambar_pil, ambang):
    """Jalankan model pada satu gambar PIL.

    Kembalikan (array_rgb_beranotasi, list_temuan). Setiap temuan berisi:
    kelas, confidence, bbox(x1,y1,x2,y2).
    """
    if model is None:
        raise RuntimeError(model_error or "Model belum dimuat.")
    hasil = model.predict(gambar_pil, conf=ambang, verbose=False)[0]
    anotasi = hasil.plot()[:, :, ::-1]  # BGR -> RGB
    temuan = []
    for box in hasil.boxes:
        nama = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        temuan.append(
            {
                "kelas": nama,
                "confidence": conf,
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
        )
    return anotasi, temuan


def waktu_sekarang():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ringkasan_teks(temuan, ambang):
    if not temuan:
        return (
            "✅ **Tidak ditemukan indikasi korosi** "
            f"(ambang confidence {ambang:.0%}). "
            "Coba turunkan ambang bila area karat halus tidak terdeteksi."
        )
    hitung = {}
    for t in temuan:
        hitung[t["kelas"]] = hitung.get(t["kelas"], 0) + 1
    rincian = ", ".join(f"{v}× {k}" for k, v in sorted(hitung.items()))
    return f"⚠️ **Ditemukan {len(temuan)} indikasi korosi**: {rincian}."


def gambar_ke_jpeg_bytes(array_rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array_rgb).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _fmt_metrik(kunci, nilai):
    """Format nilai metrik: persen untuk P/R/mAP/F1, apa adanya untuk lainnya."""
    k = kunci.lower()
    if any(s in k for s in ("precision", "recall", "map", "f1", "presisi")):
        try:
            return f"{float(nilai):.1%}"
        except (TypeError, ValueError):
            return str(nilai)
    return str(nilai)


def teks_tentang():
    b = [f"# {PROJECT_NAME}", ""]
    b.append(
        "Sistem inspeksi visual berbantuan AI untuk mendeteksi **korosi (karat)** "
        "pada komponen logam — diterapkan pada **replika Panser Anoa** — "
        "langsung dari foto, menggunakan model *deep learning* **YOLOv8n**."
    )
    b.append("")
    b.append("## Cara memakai")
    b.append(
        "1. **Deteksi** — unggah satu foto (atau pakai kamera), atur ambang "
        "confidence, tekan **Deteksi Kerusakan**. Hasil: gambar ber-bounding box, "
        "tabel temuan beserta koordinat, rekomendasi tindak lanjut, dan tombol unduh."
    )
    b.append(
        "2. **Batch** — unggah banyak foto sekaligus untuk inspeksi massal; "
        "unduh laporan CSV berisi seluruh temuan."
    )
    b.append("3. **Riwayat** — melihat kembali hasil deteksi pada sesi ini.")
    b.append("")

    b.append("## Info model")
    b.append(f"- Nama model: **{info_model.get('nama', '-')}**")
    metrik = info_model.get("metrik") or {}
    utama = [(k, v) for k, v in metrik.items()
             if k not in ("sumber", "validasi_akhir_post_training")]
    if utama:
        b.append("- Kinerja pada data validasi (hasil training nyata):")
        for k, v in utama:
            b.append(f"  - {k}: **{_fmt_metrik(k, v)}**")
    post = metrik.get("validasi_akhir_post_training") or {}
    if post:
        b.append("- Validasi akhir pasca-training (bobot final, data validasi):")
        for k, v in post.items():
            if k == "sumber":
                continue
            b.append(f"  - {k}: **{_fmt_metrik(k, v)}**")
    if metrik.get("sumber"):
        b.append(f"- Sumber angka: {metrik['sumber']}")
    hp = info_model.get("hiperparameter") or {}
    if hp:
        b.append("- Hiperparameter training:")
        for k, v in hp.items():
            b.append(f"  - {k}: {v}")
    if info_model.get("tanggal_training"):
        b.append(f"- Tanggal training: {info_model['tanggal_training']}")
    if info_model.get("catatan"):
        b.append(f"- Catatan: {info_model['catatan']}")
    b.append("")

    ds = info_model.get("dataset") or {}
    if ds:
        b.append("## Dataset")
        b.append(f"- Nama: **{ds.get('nama', '-')}**")
        b.append(
            f"- Jumlah citra: {ds.get('train', '-')} latih / "
            f"{ds.get('valid', '-')} validasi / {ds.get('test', '-')} uji"
        )
        b.append(f"- Lisensi: **{ds.get('lisensi', '-')}**")
        b.append(f"- Sumber: {ds.get('sumber', '-')}")
        if ds.get("atribusi"):
            b.append(f"- Atribusi: {ds['atribusi']}")
        b.append("")

    b.append("## Disclaimer")
    b.append(
        "Aplikasi ini adalah **alat bantu inspeksi awal**, bukan pengganti "
        "pemeriksaan teknis oleh personel yang berkompeten. Keputusan perawatan "
        "dan perbaikan tetap menjadi tanggung jawab pengguna."
    )
    b.append("")
    b.append(
        "---\nDikembangkan sebagai bagian dari skripsi **Mayor Ersa** — "
        "*Implementasi Artificial Intelligence untuk Deteksi Kerusakan Komponen "
        "Eksternal pada Replika Panser Anoa Berbasis Computer Vision*."
    )
    return "\n".join(b)


# ----------------------------------------------------------------------------
# State sesi
# ----------------------------------------------------------------------------
if "riwayat" not in st.session_state:
    st.session_state["riwayat"] = []

KOLOM_DETEKSI = ["Kelas Temuan", "Confidence", "Koordinat BBox (x1,y1,x2,y2)", "Rekomendasi"]
KOLOM_BATCH = ["File", "Jumlah Temuan", "Kelas Terdeteksi", "Confidence Tertinggi"]
KOLOM_RIWAYAT = ["Waktu", "Sumber", "Jumlah Temuan"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #15251a 0%, #2e5233 55%, #15251a 100%);
                color: #f5f5f0; border-radius: 12px; padding: 24px 18px;
                text-align: center; margin-bottom: 12px; border: 1px solid #3f6b46;">
        <h1 style="margin: 0 0 8px 0; font-size: 1.75em; letter-spacing: 1px;">{PROJECT_NAME}</h1>
        <p style="margin: 0; opacity: 0.88;">Deteksi Korosi (Rust) pada Komponen Logam Replika Panser Anoa
        &nbsp;•&nbsp; YOLOv8 Computer Vision &nbsp;•&nbsp; Skripsi Mayor Ersa</p>
    </div>
    """,
    unsafe_allow_html=True,
)
if model is None:
    st.error(f"⚠️ **Model belum tersedia.** {model_error}")

# ----------------------------------------------------------------------------
# Navigasi
# ----------------------------------------------------------------------------
halaman = st.sidebar.radio(
    "Navigasi",
    ["🔍 Deteksi", "📦 Batch", "🕘 Riwayat", "ℹ️ Tentang"],
)

# ----------------------------------------------------------------------------
# Halaman: Deteksi (satu gambar)
# ----------------------------------------------------------------------------
if halaman == "🔍 Deteksi":
    st.subheader("Deteksi Kerusakan — Satu Foto")
    st.write(
        "Unggah foto komponen logam atau ambil langsung dari kamera, "
        "atur **ambang confidence**, lalu tekan **Deteksi Kerusakan**."
    )
    col_kiri, col_kanan = st.columns(2)
    with col_kiri:
        sumber = st.radio("Sumber foto", ["Unggah foto", "Kamera"], horizontal=True)
        berkas = None
        if sumber == "Unggah foto":
            berkas = st.file_uploader(
                "Foto Komponen", type=["jpg", "jpeg", "png", "webp"]
            )
        else:
            berkas = st.camera_input("Ambil foto dari kamera")
        ambang = st.slider(
            "Ambang confidence (0,1–0,9)", 0.1, 0.9, 0.5, 0.05, format="%.2f"
        )
        tombol = st.button("Deteksi Kerusakan", type="primary")

    if tombol:
        if berkas is None:
            st.warning("Silakan unggah foto atau ambil dari kamera terlebih dahulu.")
        elif model is None:
            st.error(f"❌ Model tidak tersedia: {model_error}")
        else:
            try:
                gambar = Image.open(berkas).convert("RGB")
                with st.spinner("Menganalisis gambar..."):
                    anotasi, temuan = analisis(gambar, ambang)
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Deteksi gagal: {e}")
            else:
                with col_kanan:
                    st.image(anotasi, caption="Hasil Deteksi (Bounding Box)")
                    st.markdown(ringkasan_teks(temuan, ambang))
                    baris = [
                        {
                            "Kelas Temuan": t["kelas"],
                            "Confidence": f"{t['confidence']:.1%}",
                            "Koordinat BBox (x1,y1,x2,y2)":
                                f"({t['x1']}, {t['y1']}, {t['x2']}, {t['y2']})",
                            "Rekomendasi": rekomendasi(t["kelas"]),
                        }
                        for t in temuan
                    ]
                    if baris:
                        st.dataframe(baris, use_container_width=True)
                    st.download_button(
                        "⬇️ Unduh gambar hasil (JPEG)",
                        data=gambar_ke_jpeg_bytes(anotasi),
                        file_name="hasil_deteksi.jpg",
                        mime="image/jpeg",
                    )
                # simpan ke riwayat sesi
                path = os.path.join(
                    tempfile.gettempdir(), f"ersa_{os.urandom(6).hex()}.jpg"
                )
                Image.fromarray(anotasi).save(path, quality=92)
                st.session_state["riwayat"].append(
                    {
                        "waktu": waktu_sekarang(),
                        "file": berkas.name if hasattr(berkas, "name") else "kamera",
                        "jumlah": len(temuan),
                        "hasil": path,
                    }
                )

# ----------------------------------------------------------------------------
# Halaman: Batch (banyak gambar)
# ----------------------------------------------------------------------------
elif halaman == "📦 Batch":
    st.subheader("Inspeksi Batch — Banyak Foto Sekaligus")
    st.write(
        "Inspeksi **banyak foto sekaligus** — cocok untuk pemeriksaan rutin "
        "replika dari berbagai sudut. Hasil ringkasan per gambar dapat "
        "diunduh sebagai laporan CSV."
    )
    files = st.file_uploader(
        "Foto-foto (bisa banyak)", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    ambang_b = st.slider(
        "Ambang confidence (0,1–0,9)", 0.1, 0.9, 0.5, 0.05,
        format="%.2f", key="ambang_batch",
    )
    if st.button("Proses Semua Foto", type="primary"):
        if not files:
            st.warning("Belum ada file. Unggah beberapa foto sekaligus.")
        elif model is None:
            st.error(f"❌ Model tidak tersedia: {model_error}")
        else:
            baris = []
            detail = []
            total = 0
            progres = st.progress(0, text="Memproses foto...")
            for i, fp in enumerate(files):
                try:
                    gambar = Image.open(fp).convert("RGB")
                except Exception:  # noqa: BLE001
                    baris.append(
                        {"File": fp.name, "Jumlah Temuan": "gagal dibaca",
                         "Kelas Terdeteksi": "-", "Confidence Tertinggi": "-"})
                    continue
                try:
                    _, temuan = analisis(gambar, ambang_b)
                except Exception as e:  # noqa: BLE001
                    baris.append(
                        {"File": fp.name, "Jumlah Temuan": f"error: {e}",
                         "Kelas Terdeteksi": "-", "Confidence Tertinggi": "-"})
                    continue
                kelas_unik = sorted({t["kelas"] for t in temuan})
                conf_max = f"{max((t['confidence'] for t in temuan), default=0):.1%}"
                baris.append(
                    {"File": fp.name, "Jumlah Temuan": len(temuan),
                     "Kelas Terdeteksi": ", ".join(kelas_unik) or "-",
                     "Confidence Tertinggi": conf_max})
                total += len(temuan)
                for t in temuan:
                    detail.append(
                        {"file": fp.name, "kelas": t["kelas"],
                         "confidence": round(t["confidence"], 4),
                         "x1": t["x1"], "y1": t["y1"],
                         "x2": t["x2"], "y2": t["y2"]})
                progres.progress((i + 1) / len(files), text=f"Memproses foto {i + 1}/{len(files)}...")
            progres.empty()

            st.markdown(
                f"Diproses **{len(files)} foto** → total **{total} indikasi korosi** "
                "terdeteksi. Unduh laporan CSV untuk dokumentasi inspeksi."
            )
            st.dataframe(baris, use_container_width=True)

            buf = io.StringIO()
            w = csv.DictWriter(
                buf, fieldnames=["file", "kelas", "confidence", "x1", "y1", "x2", "y2"]
            )
            w.writeheader()
            w.writerows(detail)
            st.download_button(
                "⬇️ Unduh laporan CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name="laporan_deteksi.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Halaman: Riwayat
# ----------------------------------------------------------------------------
elif halaman == "🕘 Riwayat":
    st.subheader("Riwayat Deteksi Sesi Ini")
    st.write(
        "Log deteksi pada sesi ini. Riwayat bersifat sementara dan "
        "terhapus saat aplikasi di-restart."
    )
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state["riwayat"] = []
        st.rerun()

    riwayat = st.session_state["riwayat"]
    if not riwayat:
        st.info("Belum ada deteksi pada sesi ini.")
    else:
        st.dataframe(
            [{"Waktu": e["waktu"], "Sumber": e["file"], "Jumlah Temuan": e["jumlah"]}
             for e in riwayat],
            use_container_width=True,
        )
        st.subheader("Galeri Hasil")
        cols = st.columns(3)
        for i, e in enumerate(riwayat):
            with cols[i % 3]:
                if os.path.exists(e["hasil"]):
                    st.image(e["hasil"], caption=f"{e['waktu']} — {e['jumlah']} temuan")

# ----------------------------------------------------------------------------
# Halaman: Tentang
# ----------------------------------------------------------------------------
else:
    st.markdown(teks_tentang())

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align: center; color: #6b7c6b; font-size: 0.85em; margin-top: 14px;">'
    f'{PROJECT_NAME} &nbsp;•&nbsp; Sistem bantu inspeksi korosi berbasis YOLOv8 '
    f'&nbsp;•&nbsp; Skripsi Mayor Ersa</div>',
    unsafe_allow_html=True,
)
