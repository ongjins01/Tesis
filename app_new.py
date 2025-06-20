# ====================== BAGIAN TRAINING DAN SIMPAN MODEL ======================
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st

st.set_page_config(page_title="Deteksi Dini Penyakit Hepatitis", layout="wide")

# 1. Baca dataset
df = pd.read_excel("Data Training Hepatitis.xlsx")

# 2. Encode kolom JK
df["JK"] = df["JK"].map({"P": 0, "L": 1})

# 3. Konversi Ya/Tidak
symptom_cols = df.columns.difference(["Umur", "JK", "Kategori Diagnosis"])
for col in symptom_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "ya" else 0)

# 4. Label encoding untuk target
label_encoder = LabelEncoder()
df["Kategori Diagnosis"] = label_encoder.fit_transform(df["Kategori Diagnosis"])

# 5. Pisahkan X dan y
X = df.drop(columns=["Kategori Diagnosis"]).fillna(0)
y = df["Kategori Diagnosis"]

# 6. SMOTE (duluan, pakai data mentah)
if len(y.unique()) < 2:
    st.warning("Jumlah kelas tidak mencukupi untuk resampling.")
else:
    smote = SMOTE(random_state=42, sampling_strategy='not majority')
    X_resampled, y_resampled = smote.fit_resample(X, y)

# 7. Normalisasi (setelah SMOTE)
scaler = MinMaxScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# 8. Latih model SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_resampled_scaled, y_resampled)

# 9. Evaluasi pada data asli
X_scaled = scaler.transform(X)
y_pred = svm_model.predict(X_scaled)

print("Akurasi:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# 10. Simpan model dan encoder
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "used_columns.pkl")


# ====================== BAGIAN STREAMLIT APP ======================

# Load model dan encoder
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
used_columns = joblib.load("used_columns.pkl")

# Dummy JK encoder dengan transform manual
def jk_encode(val):
    return 1 if val.startswith("L") else 0

model_ready = all([svm_model, label_encoder, scaler, used_columns])

st.title("🩺 Aplikasi Deteksi Dini Hepatitis")

menu = st.sidebar.selectbox("📋 Menu", ["🏠 Beranda", "📈 Diagnosis", "🧪 Uji Dengan Data Baru"])

if menu == "🏠 Beranda":
    st.markdown("Aplikasi ini membantu Anda mengetahui tingkat risiko seseorang terkena **penyakit hepatitis**, "
                "berdasarkan berbagai **faktor risiko**")

    st.markdown("👉 **Catatan penting :** Aplikasi ini **bukan alat diagnosis medis**, namun dapat digunakan sebagai "
                "alat bantu untuk **mendeteksi risiko dini** "
                "agar Anda dapat segera melakukan konsultasi lanjutan ke fasilitas kesehatan.")
    st.markdown("💡 Semakin awal risiko diketahui, semakin besar peluang pencegahan dan penanganan yang tepat.")
    st.markdown("📥 Silakan **unggah data faktor risiko pasien** atau **isi data secara manual** untuk memulai analisis.")

elif menu == "📈 Diagnosis":
    st.header("📈 Diagnosis Hepatitis")

    tab1, tab2 = st.tabs(["📂 Upload File", "✍️ Input Manual"])

    with tab1:
        st.subheader("📂 Prediksi Masal")
        if not model_ready:
            st.warning("❗ Model belum tersedia. Silakan latih ulang model terlebih dahulu.")
        else:
            testing_file = st.file_uploader("Upload File Prediksi Dengan Data Masal (.xlsx)", type=["xlsx"])
            if testing_file:
                try:
                    # Baca file asli untuk mempertahankan label
                    df_test_raw = pd.read_excel(testing_file)
                    df_test = df_test_raw.copy()

                    # Proses kolom JK
                    df_test["JK"] = df_test["JK"].astype(str).str[0].map({"P": 0, "L": 1})

                    # Proses Ya/Tidak
                    for col in used_columns:
                        if col not in ["JK", "Umur"]:
                            df_test[col] = df_test[col].apply(lambda x: 1 if str(x).strip().lower() == "ya" else 0)

                    # Lengkapi kolom dan isi NaN
                    df_test = df_test.reindex(columns=used_columns, fill_value=0).fillna(0)

                    # Normalisasi
                    X_test = scaler.transform(df_test)

                    # ======== Jika ada label (Kategori Diagnosis) ========
                    if "Kategori Diagnosis" in df_test_raw.columns:
                        y_true = label_encoder.transform(df_test_raw["Kategori Diagnosis"])
                        y_pred = svm_model.predict(X_test)

                        # Akurasi
                        acc = accuracy_score(y_true, y_pred) * 100
                        st.success(f"🎯 Akurasi Model SVM: {acc:.2f}%")
                        st.markdown(
                            f"**Total data:** {len(y_true)} | "
                            f"**Benar:** {(y_true == y_pred).sum()} | "
                            f"**Salah:** {(y_true != y_pred).sum()}"
                        )

                        # Confusion Matrix
                        cm = confusion_matrix(y_true, y_pred)
                        df_cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
                        st.subheader("📌 Confusion Matrix")
                        st.dataframe(df_cm)

                    # ======== Jika tidak ada label, tampilkan distribusi prediksi ========
                    else:
                        y_pred = svm_model.predict(X_test)
                        pred_labels = label_encoder.inverse_transform(y_pred)
                        pred_counts = pd.Series(pred_labels).value_counts().reset_index()
                        pred_counts.columns = ["Kategori Diagnosis", "Jumlah Prediksi"]
                        st.info("📊 Data tidak memiliki label asli. Berikut distribusi prediksi:")
                        st.dataframe(pred_counts)

                    # Tampilkan hasil prediksi lengkap
                    df_test_raw["Prediksi"] = label_encoder.inverse_transform(svm_model.predict(X_test))
                    st.subheader("📋 Hasil Prediksi")
                    st.dataframe(df_test_raw)

                except Exception as e:
                    st.error(f"❌ Gagal memproses data: {e}")

    # ====================== TAB 2: Input Manual ======================
    with tab2:
        st.subheader("✍️ Input Manual Pasien")
        st.error("##### 📝 Note :\n"
                 "- **Ikterus:** Kulit dan bagian putih mata menguning\n"
                 "- **Edema/Ascites:** Pembengkakan kaki (edema) atau perut (ascites)")
        if not model_ready:
            st.warning("❗ Model belum tersedia.")
        else:
            data_input = {
                'JK': st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"]),
                'Umur': st.number_input("Umur", min_value=0, max_value=90, value=30)
            }

            gejala_list = [
                'Demam', 'Kelelahan', 'Kehilangan Nafsu Makan', 'Mual dan Muntah', 'Nyeri Perut Kanan Atas',
                'Urin Gelap', 'Feses Pucat', 'Ikterus', 'Gatal', 'Edema/Ascites',
                'Diare/Gangguan Pencernaan', 'Berat Badan Turun', 'Ruam/Nyeri Sendi', 'Menggigil'
            ]

            for g in gejala_list:
                data_input[g] = 1 if st.radio(g, ["Ya", "Tidak"], key=g) == "Ya" else 0

            if st.button("🔍 Prediksi Sekarang"):
                data_input['JK'] = jk_encode(data_input['JK'])

                # Buat input_df lengkap
                input_df = pd.DataFrame([[data_input[col] for col in used_columns]], columns=used_columns).fillna(0)

                # Normalisasi semua kolom sekaligus
                input_scaled = scaler.transform(input_df)

                # Prediksi
                pred = svm_model.predict(input_scaled)[0]
                probas = svm_model.predict_proba(input_scaled)[0]
                hasil = label_encoder.inverse_transform([pred])[0]

                st.success(f"🧾 Prediksi Diagnosis: **{hasil}**")

                # Tampilkan keterangan dan tindakan
                penjelasan = {
                    "Abses Hati": (
                        "Abses hati Abses hati atau abses hepar adalah kantong berisi nanah yang terbentuk di dalam hati. "
                        "Kondisi ini umumnya disebabkan oleh infeksi bakteri dan ameba yang masuk ke hati melalui luka tusuk pada perut, "
                        "atau penyebaran infeksi dari organ pencernaan lain.",
                        "📌 **Tindakan**: Segera konsultasikan ke dokter untuk pemeriksaan lanjutan (USG, CT Scan), dan kemungkinan pemberian antibiotik atau drainase."
                        " Adapun tindakan pengobatan yang dapat dilakukan adalah Minum antibiotik sesuai anjuran dokter"
                        ", Rutin cek kesehatan ke dokter untuk memantau kondisi kesehatan, Selalu cuci tangan pakai sabun sebelum makan"
                        ", Pastikan untuk memasak makanan hingga matang, Hindari kebiasaan jajan sembarangan."
                    ),
                    "Hepatitis Kronis": (
                        "Hepatitis kronis adalah peradangan hati jangka panjang yang bisa disebabkan oleh virus hepatitis B atau C."
                        "Kedua virus ini dapat ditularkan dari orang ke orang melalui kontak seksual atau melalui kontak darah atau "
                        "cairan tubuh lainnya melalui jarum suntik atau transfusi darah. Maka dari itu, sebaiknya hindari melakukan "
                        "hubungan seksual yang tidak aman dan pastikan kebersihan jarum suntik saat akan menggunakannya.",
                        "📌 **Tindakan**: Lakukan tes darah lanjutan serta konsultasi dengan dokter untuk mendapatkan diagnosis dan rencana pengobatan yang tepat."
                        "Untuk mencegah dan mengendalikan hepatitis kronis, lakukan beberapa tindakan seperti "
                        "berhenti minum alkohol, hindari obat-obatan tanpa resep dokter, istirahat yang cukup, "
                        "konsumsi makanan sehat, jangan berbagi alat pribadi, dan lakukan hubungan seks yang aman. "
                        "Tindakan-tindakan ini penting agar hati tetap sehat dan penularan hepatitis bisa dicegah."
                    ),
                    "Infeksi Parasit atau Virus": (
                        "Infeksi ini disebabkan oleh parasit atau virus lain di luar hepatitis, seperti amuba atau virus saluran cerna."
                        "Infeksi parasit terjadi ketika parasit masuk ke dalam tubuh manusia melalui mulut atau kulit. "
                        "Parasit tersebut kemudian berkembang dan menginfeksi organ tubuh tertentu.",
                        "📌 **Tindakan**: Jika Anda mengalami gejala infeksi, yang tidak membaik setelah 3 hari, segera periksakan diri ke Dokter Umum, "
                        "Pemeriksaan laboratorium lanjutan mungkin diperlukan. Perawatan tergantung penyebab spesifik—antiparasit atau antivirus juga diperlukan."
                    ),

                    "Hepatitis Akut": (
                        "Hepatitis akut adalah peradangan hati yang muncul secara tiba-tiba, umumnya karena infeksi virus, obat, atau zat toksik. "
                        "Hepatitis akut bisa disebabkan oleh beberapa hal, tetapi penyakit ini lebih sering terjadi akibat infeksi virus hepatitis A, B, C, D, dan E. "
                        "Pada kasus yang tengah marak sekarang ini, ada dugaan jika adenovirus tipe 41 dan virus corona (SARS-CoV-2) juga bisa menyebabkan hepatitis akut. "
                        "Namun, dugaan tersebut masih membutuhkan bukti dan penelitian lebih lanjut.",
                    "📌 **Tindakan**: "
                        "Istirahat total, pemantauan fungsi hati, hindari obat sembarangan, dan segera periksa ke dokter. "
                        "Berikut adalah beberapa tips terhindar dari hepatitis akut, yang dapat dilakukan bersama, "
                        "diantaranya adalah Menerapkan protokol kesehatan, terutama menggunakan masker dan mencuci tangan sebelum dan sesudah melakukan aktivitas, "
                        "Memastikan makanan yang dikonsumsi dalam keadaan matang dan bersih, "
                        "Menghindari kontak dengan orang yang sakit, "
                        "Kurangi mobilitas, "
                        "Tidak bergantian alat makan dengan orang lain. "
                    )
                }

                if hasil in penjelasan:
                    deskripsi, tindakan = penjelasan[hasil]
                    st.markdown("### 🩺 Keterangan Medis")
                    st.info(deskripsi)
                    st.markdown(tindakan)

# ==========================
# IMPORT DATA TRAINING BARU
# ==========================
elif menu == "🧪 Uji Dengan Data Baru":
    st.header("📥 Unggah Data Training (.xlsx)")
    training_file = st.file_uploader("Upload file Excel untuk Training", type=["xlsx"])

    if training_file:
        try:
            df = pd.read_excel(training_file)
            st.write("Preview Data:", df.head())

            df["JK"] = df["JK"].astype(str).str[0]
            jk_encoder = LabelEncoder()
            df["JK"] = jk_encoder.fit_transform(df["JK"])

            label_encoder = LabelEncoder()
            df["label"] = label_encoder.fit_transform(df["Kategori Diagnosis"])

            # ⬅️ Menampilkan jumlah kategori diagnosis
            st.markdown("### 📊 Jumlah Kategori Diagnosis")
            st.dataframe(df["Kategori Diagnosis"].value_counts().reset_index().rename(columns={
                "index": "Kategori Diagnosis", "Kategori Diagnosis": "Jumlah"
            }))

            # Pra-pemrosesan fitur
            used_columns = ['JK', 'Umur'] + [col for col in df.columns if col not in ['Kategori Diagnosis', 'label', 'JK', 'Umur']]
            for col in used_columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "ya" else 0)

            scaler = MinMaxScaler()
            df[['Umur']] = scaler.fit_transform(df[['Umur']])

            X_train = df[used_columns].fillna(0)
            y_train = df["label"]

            use_smote = st.checkbox("Gunakan SMOTE untuk penyeimbangan data", value=False)
            if use_smote:
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.info("✅ SMOTE berhasil diterapkan.")

            model = SVC(kernel="linear", probability=True)
            model.fit(X_train, y_train)

            # ===== Akurasi Training =====
            y_pred_train = model.predict(X_train)
            acc_train = accuracy_score(y_train, y_pred_train) * 100
            st.info(f"🎯 Akurasi Model di Data Training: **{acc_train:.2f}%**")
            st.markdown("---")

            joblib.dump(model, "svm_model.pkl")
            joblib.dump(jk_encoder, "jk_encoder.pkl")
            joblib.dump(label_encoder, "label_encoder.pkl")
            joblib.dump(used_columns, "used_columns.pkl")
            joblib.dump(scaler, "scaler.pkl")

            st.success("✅ Model berhasil dilatih dan disimpan.")
            st.markdown("---")

            evaluasi_mode = st.radio("🧪 Prediksi Dengan Data Baru:", ["📂 Upload Data Masal", "✍️ Input Manual"], horizontal=True)

            if evaluasi_mode == "📂 Upload Data Masal":
                testing_file = st.file_uploader("Unggah File (.xlsx)", type=["xlsx"],
                                                key="uji_testing_dari_training")
                if testing_file:
                    try:
                        df_test = pd.read_excel(testing_file)

                        # Tambahkan kolom yang hilang agar sesuai dengan used_columns
                        for col in used_columns:
                            if col not in df_test.columns:
                                df_test[col] = "Tidak"

                        # Proses JK
                        df_test["JK_model"] = df_test["JK"].astype(str).str[0]
                        df_test["JK_model"] = jk_encoder.transform(df_test["JK_model"])

                        # Proses Umur
                        df_test["Umur_model"] = scaler.transform(df_test[["Umur"]])

                        # Proses kolom gejala → buat salinan kolom gejala_model
                        gejala_cols = [col for col in used_columns if col not in ["JK", "Umur"]]
                        X_gejala = df_test[gejala_cols].applymap(lambda x: 1 if str(x).strip().lower() == "ya" else 0)

                        # Siapkan X_test final untuk model (JK_model, Umur_model, Gejala)
                        X_test = pd.concat([
                            df_test[["JK_model", "Umur_model"]].rename(
                                columns={"JK_model": "JK", "Umur_model": "Umur"}),
                            X_gejala
                        ], axis=1)

                        X_test = X_test[used_columns].fillna(0)

                        # Prediksi
                        y_pred = model.predict(X_test)
                        df_test["Hasil Prediksi"] = label_encoder.inverse_transform(y_pred)

                        # 👉 Drop kolom JK_model dan Umur_model agar tidak tampil di output
                        df_test = df_test.drop(columns=["JK_model", "Umur_model"])

                        # Tampilkan df_test asli (tetap "Ya"/"Tidak", Umur asli, JK asli)
                        st.subheader("📄 Hasil Prediksi")
                        st.dataframe(df_test)

                        # Distribusi
                        st.markdown("### 📊 Distribusi Hasil Prediksi")
                        st.dataframe(
                            df_test["Hasil Prediksi"].value_counts().reset_index().rename(columns={
                                "index": "Kategori Diagnosis", "Hasil Prediksi": "Jumlah"
                            })
                        )


                    except Exception as e:
                        st.error(f"❌ Gagal memproses data testing: {e}")


            elif evaluasi_mode == "✍️ Input Manual":
                st.subheader("✍️ Input Manual Pasien")
                st.error("##### 📝 Note :\n"
                            "- **Ikterus:** Kulit dan bagian putih mata menguning\n"
                            "- **Edema/Ascites:** Pembengkakan kaki (edema) atau perut (ascites)")

                manual_input = {}
                manual_input['JK'] = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key="jk_manual")
                manual_input['Umur'] = st.number_input("Umur", min_value=0, max_value=90, value=30, key="umur_manual")

                def yn(label):
                    return 1 if st.radio(label, ["Ya", "Tidak"], key=label) == "Ya" else 0

                gejala_list = [
                    'Demam', 'Kelelahan', 'Kehilangan Nafsu Makan', 'Mual dan Muntah', 'Nyeri Perut Kanan Atas',
                    'Urin Gelap', 'Feses Pucat', 'Ikterus', 'Gatal', 'Edema/Ascites',
                    'Diare/Gangguan Pencernaan', 'Berat Badan Turun', 'Ruam/Nyeri Sendi', 'Menggigil'
                ]
                for g in gejala_list:
                    manual_input[g] = yn(g)

                if st.button("🔍 Prediksi Sekarang"):
                    try:
                        manual_input["JK"] = jk_encoder.transform([manual_input["JK"][0]])[0]
                        manual_input["Umur"] = scaler.transform([[manual_input["Umur"]]])[0][0]
                        input_df = pd.DataFrame([[manual_input[col] for col in used_columns]], columns=used_columns).fillna(0)

                        pred = model.predict(input_df)[0]
                        probas = model.predict_proba(input_df)[0]
                        hasil = label_encoder.inverse_transform([pred])[0]

                        st.success(f"🧾 Prediksi Diagnosis: **{hasil}**")

                        # Tampilkan keterangan dan tindakan
                        penjelasan = {
                            "Abses Hati": (
                                "Abses hati Abses hati atau abses hepar adalah kantong berisi nanah yang terbentuk di dalam hati. "
                                "Kondisi ini umumnya disebabkan oleh infeksi bakteri dan ameba yang masuk ke hati melalui luka tusuk pada perut, "
                                "atau penyebaran infeksi dari organ pencernaan lain.",
                                "📌 **Tindakan**: Segera konsultasikan ke dokter untuk pemeriksaan lanjutan (USG, CT Scan), dan kemungkinan pemberian antibiotik atau drainase."
                                " Adapun tindakan pengobatan yang dapat dilakukan adalah Minum antibiotik sesuai anjuran dokter"
                                ", Rutin cek kesehatan ke dokter untuk memantau kondisi kesehatan, Selalu cuci tangan pakai sabun sebelum makan"
                                ", Pastikan untuk memasak makanan hingga matang, Hindari kebiasaan jajan sembarangan."
                            ),
                            "Hepatitis Kronis": (
                                "Hepatitis kronis adalah peradangan hati jangka panjang yang bisa disebabkan oleh virus hepatitis B atau C."
                                "Kedua virus ini dapat ditularkan dari orang ke orang melalui kontak seksual atau melalui kontak darah atau "
                                "cairan tubuh lainnya melalui jarum suntik atau transfusi darah. Maka dari itu, sebaiknya hindari melakukan "
                                "hubungan seksual yang tidak aman dan pastikan kebersihan jarum suntik saat akan menggunakannya.",
                                "📌 **Tindakan**: Lakukan tes darah lanjutan serta konsultasi dengan dokter untuk mendapatkan diagnosis dan rencana pengobatan yang tepat."
                                "Untuk mencegah dan mengendalikan hepatitis kronis, lakukan beberapa tindakan seperti "
                                "berhenti minum alkohol, hindari obat-obatan tanpa resep dokter, istirahat yang cukup, "
                                "konsumsi makanan sehat, jangan berbagi alat pribadi, dan lakukan hubungan seks yang aman. "
                                "Tindakan-tindakan ini penting agar hati tetap sehat dan penularan hepatitis bisa dicegah."
                            ),
                            "Infeksi Parasit atau Virus": (
                                "Infeksi ini disebabkan oleh parasit atau virus lain di luar hepatitis, seperti amuba atau virus saluran cerna."
                                "Infeksi parasit terjadi ketika parasit masuk ke dalam tubuh manusia melalui mulut atau kulit. "
                                "Parasit tersebut kemudian berkembang dan menginfeksi organ tubuh tertentu.",
                                "📌 **Tindakan**: Jika Anda mengalami gejala infeksi, yang tidak membaik setelah 3 hari, segera periksakan diri ke Dokter Umum, "
                                "Pemeriksaan laboratorium lanjutan mungkin diperlukan. Perawatan tergantung penyebab spesifik—antiparasit atau antivirus juga diperlukan."
                            ),

                            "Hepatitis Akut": (
                                "Hepatitis akut adalah peradangan hati yang muncul secara tiba-tiba, umumnya karena infeksi virus, obat, atau zat toksik. "
                                "Hepatitis akut bisa disebabkan oleh beberapa hal, tetapi penyakit ini lebih sering terjadi akibat infeksi virus hepatitis A, B, C, D, dan E. "
                                "Pada kasus yang tengah marak sekarang ini, ada dugaan jika adenovirus tipe 41 dan virus corona (SARS-CoV-2) juga bisa menyebabkan hepatitis akut. "
                                "Namun, dugaan tersebut masih membutuhkan bukti dan penelitian lebih lanjut.",
                                "📌 **Tindakan**: Istirahat total, pemantauan fungsi hati, hindari obat sembarangan, dan segera periksa ke dokter. "
                                "Berikut adalah beberapa tips terhindar dari hepatitis akut, yang dapat dilakukan bersama, "
                                "diantaranya adalah Menerapkan protokol kesehatan, terutama menggunakan masker dan mencuci tangan sebelum dan sesudah melakukan aktivitas, "
                                "Memastikan makanan yang dikonsumsi dalam keadaan matang dan bersih, "
                                "Menghindari kontak dengan orang yang sakit, "
                                "Kurangi mobilitas, "
                                "Tidak bergantian alat makan dengan orang lain. "
                            )
                        }

                        if hasil in penjelasan:
                            deskripsi, tindakan = penjelasan[hasil]
                            st.markdown("### 🩺 Keterangan Medis")
                            st.info(deskripsi)
                            st.markdown(tindakan)

                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat prediksi manual: {e}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data training: {e}")
