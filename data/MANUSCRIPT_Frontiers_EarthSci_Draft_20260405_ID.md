# Kerangka Kerja Dua-Tahap Sadar-Saturasi untuk PTW Adaptif Berbasis Intensitas: Memprioritaskan Keselamatan Operasional dan Robustness Respons Situs di Wilayah Jawa-Sumatra

**Target Jurnal:** *Frontiers in Earth Science* — Bagian: Solid Earth Geophysics  
**Jenis Artikel:** Original Research Article  
**Jumlah Kata:** ~9,800 | **Gambar:** 15 | **Tabel:** 8 (utama) + 2 (tambahan) = 10 total  
**Pembaruan Terakhir:** 10 April 2026 — Eksperimen V_S30 yang diperluas (338 stasiun) terintegrasi

---

## Penulis

**Hanif Andi Nugraha**¹\* (ORCID: 0009-0007-9975-1566), **Dede Djuhana**¹ (ORCID: 0000-0002-2025-0782), **Adhi Harmoko Saputro**¹ (ORCID: 0000-0001-6651-0669), **Sigit Pramono**² (ORCID: 0009-0000-5684-282X)

¹ Departemen Fisika, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Indonesia, Depok, Jawa Barat, Indonesia
² Badan Meteorologi, Klimatologi, dan Geofisika Republik Indonesia (BMKG), Jakarta, Indonesia

\*Korespondensi: hanif.andi@ui.ac.id

---

Dataset berisi 34.033 jejak akselerometrik tiga-komponen dari 329 gempa bumi (M 4.0–6.9) yang tercatat di 388 stasiun di seluruh zona subduksi Sunda, khususnya berfokus pada wilayah studi Jawa-Sumatra (95°–115°E, 11°S–6°N). Kami mempresentasikan kerangka kerja Dua-Tahap Sadar-Saturasi yang secara adaptif memilih PTW melalui paradigma *Dikotomi Fitur* yang inovatif — secara sistematis memanfaatkan perilaku fisik yang berbeda dari ukuran intensitas gelombang P yang menjenuhkan dan tidak menjenuhkan (IMs). Tahap 1 menggunakan pengklasifikasi XGBoost "Intensity Gate" yang menggunakan terutama parameter menjenuhkan τ_c (periode predominan, 51% feature importance) untuk merutekan jejak seismik ke dalam tiga kelas intensitas operasional (*Lemah*, *Terasa*, dan *Merusak*), sementara Tahap 2 memanfaatkan Cumulative Velocity Absolute Displacement (CVAD) non-menjenuhkan dengan importance 89,5% untuk memprediksi spektral akselerasi Sa(T) di 103 periode (T = 0,1–5,0 s). Sebuah penduga jarak episentral on-site yang dilatih pada 30 fitur gelombang P (Zhang et al., 2024) memenuhi tabel pencarian **Intensitas × Jarak** yang dibatasi oleh golden time kedatangan gelombang S, memastikan bahwa PTW yang dipilih tidak pernah mengorbankan kelayakan peringatan. Menggunakan cross-validasi 5-fold yang dikelompokkan menurut kejadian pada dataset wilayah studi, pipeline mencapai R² gabungan = **0.7309** dan RMSE = **0.5181** log₁₀ m/s² untuk deployment operasional penuh — di mana kelas intensitas dan jarak episentral keduanya diprediksi murni dari sinyal gelombang P awal 3 detik di stasiun tunggal. Hasil ini menunjukkan bahwa kerangka kerja adaptif mempertahankan akurasi tingkat engineering sambil memprioritaskan keselamatan operasional melalui peningkatan recall merusak sebesar 55× dibandingkan dengan model berbasis sumber standar. Dikotomi Fitur muncul secara alami dari fisika gelombang P dan menawarkan paradigma desain baru untuk EEWS generasi berikutnya.

**Kata Kunci:** earthquake early warning, spectral acceleration, adaptive time window, XGBoost, feature dichotomy, Sunda subduction zone, single-station on-site, golden time constraint

---

## 1. Pendahuluan

### 1.1 Sistem Peringatan Dini Gempa Bumi

Kepulauan Indonesia terletak di atas salah satu wilayah paling aktif secara seismik di Bumi, di mana lempeng Indo-Australia menunjam di bawah lempeng Eurasia sepanjang megathrust Sunda 5.500 km dengan kecepatan 50–70 mm/tahun (Simons et al., 2007; Natawidjaja et al., 2006). Kondisi tektonik ini telah menghasilkan gempa bumi yang menghancurkan termasuk Sumatra-Andaman M 9,1 tahun 2004, Yogyakarta M 6,3 tahun 2006, dan peristiwa Palu M 7,5 tahun 2018. Sistem Peringatan Dini Gempa Bumi (EEWS) memanfaatkan perbedaan waktu tempuh antara gelombang P cepat (V_P ≈ 6,5 km/s) dan gelombang S yang lebih lambat namun lebih merusak (V_S ≈ 3,5 km/s) untuk mengeluarkan peringatan beberapa detik hingga puluhan detik sebelum getaran kuat tiba (Allen and Melgar, 2019; Cremen and Galasso, 2020).

EEWS nasional Indonesia, InaTEWS, dioperasikan oleh Badan Meteorologi, Klimatologi, dan Geofisika (BMKG), saat ini menggunakan pendekatan berbasis jaringan yang memerlukan multiple station triggers sebelum mengeluarkan peringatan (Kopp et al., 2008; Münchmeyer et al., 2021). Metode on-site stasiun tunggal menawarkan pendekatan komplemen yang dapat memberikan estimasi awal lebih cepat, khususnya untuk kejadian near-field di mana triangulasi jaringan memperkenalkan delay kritis (Nakamura, 1988; Wu and Kanamori, 2005a). Dalam beberapa tahun terakhir, implementasi EEWS regional telah berkembang pesat secara global, mencerminkan lingkungan tektonik dan kendala teknologi yang berbeda. Ini termasuk peningkatan nasional komprehensif di Jepang (Kodera et al., 2021), jaringan observasi lepas pantai (Dhakal and Kunugi, 2021), dan sistem operasional yang terus berkembang di Israel (Nof et al., 2021), Swiss (Massin et al., 2021), Italia (Ladina et al., 2021), dan Yunani (Bracale et al., 2021). Pengembangan paralel terjadi di zona aktif seperti Kosta Rika (Porras et al., 2021), Iberia selatan (Carranza et al., 2021), British Columbia (Schlesinger et al., 2021), dan Sichuan, Cina (Peng et al., 2021). Studi komparatif yang mengevaluasi algoritma regional ini mengkonfirmasi kesulitan inheren dalam menyeimbangkan kecepatan dan akurasi di berbagai konfigurasi geografis (Zuccolo et al., 2021).

Di daerah dengan cakupan stasiun yang jarang atau seismisitas lebih rendah, membangun jaringan padat yang kokoh memerlukan investasi modal yang substansial, menjadikan algoritma stasiun tunggal yang dioptimalkan atau jaringan telemetri crowdsource sebagai strategi alternatif yang sangat berharga (Parolai et al., 2017; Finazzi, 2020; Velazquez et al., 2020; Ahn et al., 2023a). Namun, pendekatan stasiun tunggal secara historis telah dibatasi oleh akurasi mereka yang lebih rendah dibandingkan solusi jaringan.

### 1.2 Dari PGA ke Spektral Akselerasi

EEWS tradisional memprediksi parameter motion tanah skalar seperti Peak Ground Acceleration (PGA) atau Peak Ground Velocity (PGV) (Wu and Kanamori, 2005b; Zollo et al., 2010). Meskipun PGA berkorelasi dengan intensitas getaran yang dirasakan, ia secara buruk merepresentasikan potensi kerusakan bergantung-frekuensi gempa bumi pada struktur (Bommer and Alarcon, 2006). Earthquake engineering berbasis kinerja modern memerlukan spektral akselerasi Sa(T) pada periode alami struktur: gedung rendah merespons spektral akselerasi periode pendek Sa (T ≈ 0,1–0,5 s), sedangkan gedung tinggi dan jembatan diatur oleh Sa periode panjang (T ≈ 1–5 s) (Abrahamson et al., 2014).

Ekspanur machine learning dalam EEWS sangat pesat; misalnya, arsitektur konvolusi deep learning modern telah secara dramatis meningkatkan estimasi magnitude (Zhu et al., 2021), dan jaringan Long Short-Term Memory (LSTM) berhasil diterapkan secara ketat pada prediksi PGA awal (Hsu and Pratomo, 2022). Kemajuan bahkan telah memungkinkan pemetaan langsung fitur peringatan awal ke prediksi structural drift menggunakan machine learning regressors (Iaccarino et al., 2021).
Lara et al. (2023) mendemonstrasikan bahwa machine learning dapat memprediksi Sa(T) pada kurang lebih 20 periode spektral dari fitur gelombang P, mencapai R² = 0,88–0,92 menggunakan data jaringan multi-stasiun. Namun, pendekatan mereka memerlukan observasi stasiun ganda dan tidak mengatasi masalah optimasi PTW. Memperluas EEWS untuk memprediksi Sa(T) di seluruh rentang frekuensi komprehensif (100+ periode) dari stasiun tunggal tetap menjadi tantangan terbuka.

### 1.3 Trade-off P-Wave Time Window

P-wave time window (PTW) — durasi sinyal gelombang P yang dianalisis sebelum mengeluarkan prediksi — mempresentasikan trade-off fundamental antara akurasi dan timeliness. Jendela yang lebih panjang mengumpulkan informasi seismik lebih banyak, khususnya konten frekuensi rendah yang berkorelasi dengan magnitude gempa bumi, menghasilkan prediksi yang lebih akurat (Lancieri and Zollo, 2008; Colombelli et al., 2015). Jendela yang lebih pendek memberikan peringatan lebih awal tetapi mengorbankan kualitas prediksi. EEWS yang ada secara keseluruhan menerapkan nilai PTW tetap (biasanya 3–4 s), menerapkan durasi observasi yang sama terlepas dari magnitude gempa bumi, jarak, atau lead time yang tersedia (Böse et al., 2009; Festa et al., 2008).

Pendekatan one-size-fits-all ini secara fundamental suboptimal. Gempa bumi kecil di dekat menghasilkan impulse gelombang P durasi pendek yang informasi diagnostiknya sepenuhnya tertangkap dalam 2–3 s, sementara gempa bumi besar jauh menghasilkan train gelombang P yang diperpanjang yang karakteristik frekuensi rendahnya hanya muncul setelah 6–10 s (Yamada and Mori, 2009). PTW adaptif yang mempertimbangkan karakteristik kejadian dapat secara bersamaan meningkatkan akurasi untuk kejadian besar dan mempertahankan timeliness untuk kejadian kecil. Penerapan fundamental dari data gelombang P awal untuk estimasi motion tanah akurat telah divalidasi secara kokoh di dataset besar (Tsuno, 2021; Wang and Zhao, 2021), sementara algoritma advanced telah mulai bertransisi menuju prediksi real-time berbasis wavefield sepenuhnya kontinyu (Hoshiba, 2021; Li et al., 2021).

### 1.4 Masalah Saturasi Fitur

Keterbatasan P-wave-based EEWS yang kritis namun kurang dieksplorasi adalah *perilaku saturasi* dari ukuran intensitas (IMs). Periode predominan τ_c (Kanamori, 2005) dan peak displacement P_d menunjukkan ceiling effects untuk gempa bumi besar (M > 6,0–6,5), di mana nilainya plateau daripada terus scaling dengan magnitude (Brown et al., 2009; Lancieri and Zollo, 2008). Saturasi ini muncul dari rupture duration terbatas: untuk gempa bumi besar, impulse gelombang P awal menangkap hanya onset rupture, dan τ_c konvergen ke nilai maksimum terlepas dari magnitude total.

Sebaliknya, parameter kumulatif seperti Cumulative Absolute Velocity (CAV; Reed and Kassawara, 1990) dan Cumulative Velocity Absolute Displacement (CVAD) terus berkembang dengan magnitude dan waktu observasi tanpa saturasi teoritis. CVAD, didefinisikan sebagai time integral dari produk velocity–displacement absolut, meningkat secara monoton saat jendela gelombang P berkembang, membuatnya sensitif terhadap ukuran sumber dan efek propagasi gelombang.

Kami menghipotesiskan bahwa *dikotomi saturasi* ini dapat secara sistematis dimanfaatkan: parameter menjenuhkan yang ceiling-nya menciptakan boundary kategori alami cocok untuk klasifikasi diskrit, sementara parameter non-menjenuhkan yang mempertahankan continuous scaling ideal untuk tugas regresi. Paradigma ini belum pernah dieksplorasi dalam literatur EEWS sebelumnya.

### 1.5 Seleksi Jendela Sadar-Jarak

PTW optimal tambahan bergantung pada jarak source-to-site melalui *golden time* constraint — perbedaan travel-time S−P yang mendefinisikan maximum allowable observation window sebelum onset getaran. Stasiun near-field (D < 50 km) memiliki golden time sekecil 5 s, sangat membatasi pilihan PTW. Stasiun far-field (D > 300 km) mungkin memiliki golden times melebihi 60 s, memberikan budget ample untuk extended observation (Zhang et al., 2024). Kemajuan terkini dalam estimasi jarak episentral on-site dari fitur gelombang P (Zhang et al., 2024) memungkinkan seleksi PTW real-time sadar-jarak tanpa memerlukan network-based source location.

### 1.6 Tujuan Studi

Studi ini memperkenalkan kerangka kerja Saturation-Aware Dual-Stage Adaptive PTW dengan tiga kontribusi utama:

1. **Paradigma Dikotomi Fitur.** Eksploitasi sistematis pertama dari sifat saturasi IM gelombang P dalam desain EEWS: parameter menjenuhkan (τ_c) untuk routing klasifikasi intensitas, dan parameter non-menjenuhkan (CVAD) untuk regresi Sa(T).

2. **PTW Adaptif Sadar-Jarak.** Tabel pencarian Intensitas × Jarak dengan golden time constraint yang secara dinamis memilih PTW optimal per trace, memaksimalkan akumulasi informasi sambil menjamin kelayakan peringatan.

3. **Prediksi Spektral Stasiun-Tunggal Komprehensif.** Prediksi end-to-end Sa(T) di 103 periode spektral (T = 0,1–5,0 s) dari stasiun tunggal menggunakan hanya sinyal gelombang P awal 3 detik, divalidasi dengan cross-validasi event-grouped ketat pada 34.033 jejak dari zona subduksi Sunda.

---
## 2. Data dan Metode

### 2.1 Area Penelitian dan Dataset

Zona subduksi Sunda meluas lebih dari 5.500 km dari Myanmar melalui Sumatra, Jawa, dan Kepulauan Sunda Kecil, mengakomodasi konvergensi 50–70 mm/tahun antara lempeng Indo-Australia dan Eurasian (Simons et al., 2007). Dataset kami terdiri dari bentuk gelombang akselerometrik tiga komponen (HN*) yang tercatat di seluruh wilayah ini, dikompilasi ke dalam dataset Sunda v2 (Geomean) yang kompatibel dengan SeisBench (Woollam et al., 2022).

Dataset berisi 34.033 jejak akselerometrik tiga komponen dari 329 gempa bumi (M 4.0–6.9) yang tercatat di 388 stasiun jaringan akselerograf BMKG (jaringan IA) dalam area penelitian yang ditentukan (95°–115°E, 11°S–6°N). Semua data bentuk gelombang diperoleh langsung dari infrastruktur perekaman operasional BMKG, mencerminkan gerakan tanah nyata yang teramati di seluruh periode akuisisi data. Jarak episentrum berkisar dari 0 hingga 1.205 km dengan median 384 km, mencerminkan distribusi stasiun yang jarang di seluruh kepulauan Indonesia dan lokasi terutama lepas pantai dari sismisitas zona subduksi. Jarak median yang besar berarti waktu emas yang substansial (median ≈ **51 s**), memberikan anggaran yang cukup untuk eksplorasi PTW adaptif. Filter area penelitian memastikan konsistensi geografis dengan mengecualikan 281 stasiun tanpa koordinat, 80 stasiun di timur 115°E (Bali, NTT, Sulawesi), dan 22 kejadian dengan koordinat katalog yang keliru.

**Table 1.** Ringkasan dataset dan distribusi kelas intensitas.

| Parameter | Value |
|---|---|
| Total traces | 34,033 |
| Unique earthquakes | 329 |
| Unique stations (IA-BMKG) | 388 |
| Study area | 95°–115°E, 11°S–6°N |
| Data source | BMKG operational accelerograph network |
| Magnitude range | M 4.0–6.9 |
| Distance range | 0–1,205 km (median 384 km) |
| *Weak* class (PGA < 3.0 Gal) | 32,001 traces (94.0%) |
| *Felt* (3.0 ≤ PGA < 62.0 Gal) | 1,931 traces (5.7%) |
| *Damaging* (PGA ≥ 62.0 Gal) | 101 traces (0.3%) |
| PTW windows | 2, 3, 4, 6, 8 s |
| Sa(T) target periods | 103 (T = 0.1–5.0 s) |

Ketidakseimbangan parah dalam kelas *Damaging* (hanya 101 jejak, 0,3%) mencerminkan kelangkaan yang melekat pada gerakan tanah intens, namun ini adalah peristiwa yang paling kritis untuk EEWS. Ketidakseimbangan ini semakin diperkuat oleh pembatasan area penelitian: 74,5% dari semua jejak yang tercatat berasal dari jalur far-field (>200 km), di mana gelombang seismik telah melemah secara substansial sebelum mencapai stasiun perekaman. Median PGA sebesar 0,13 Gal (jauh di bawah ambang batas persepsi manusia ~3 Gal) mengonfirmasi bahwa dataset didominasi oleh getaran lemah — representasi setia dari realitas operasional yang dihadapi jaringan akselerograf BMKG di wilayah Jawa-Sumatra.

Distribusi spasial dari 329 gempa bumi dan 388 stasiun akselerograf dalam area penelitian ditunjukkan pada Figure 1. Kedalaman kejadian diklasifikasikan mengikuti konvensi standar BMKG: dangkal (≤60 km, merah), menengah (60–300 km, kuning), dan dalam (>300 km, hijau). Basemap ditampilkan menggunakan layanan ubin Rupabumi Indonesia yang disediakan oleh Badan Informasi Geospasial (BIG). Cakupan ini memastikan bahwa pipeline terpajan pada berbagai efek jalur dan kondisi situs yang karakteristik dari busur Sunda Indonesia, sambil mempertahankan konsistensi geografis dengan membatasi semua data ke wilayah penelitian Jawa-Sumatra.

![Figure 1. Seismicity map of the Java-Sumatra study area overlaid on the BIG Rupabumi Indonesia basemap. Colored circles represent 329 earthquakes (M 4.0–6.9) scaled by magnitude and colored by depth following BMKG convention: shallow ≤60 km (red), intermediate 60–300 km (yellow), deep >300 km (green). Black triangles denote 388 IA-BMKG accelerograph stations. The red rectangle delineates the study area boundary (95°–115°E, 11°S–6°N). The dashed red line indicates the approximate trace of the Sunda Trench. Data period: 2020–2025.](figures/fig01.png)

Figure 2 mengilustrasikan karakteristik numerik dataset, termasuk distribusi magnitudo–intensitas, distribusi jarak hiposentral, dan pencar bersama mereka.

![Figure 2. Dataset overview (Java-Sumatra Filtered). (a) Magnitude distribution across filtered classes. (b) Hypocentral distance distribution with median focus on the regional study area.](figures/fig02.png)

### 2.2 Ekstraksi Fitur

Untuk setiap jejak pada setiap PTW, kami mengekstrak 42 fitur yang terdiri dari dua set komplementer:

**12 Intensity Measures (IMs).** Parameter gelombang P ini mencakup spektrum saturasi (Table 2). Periode dominan τ_c dihitung sebagai:

$$\tau_c = \frac{2\pi}{\sqrt{r}} \quad \text{where} \quad r = \frac{\int_0^{\text{PTW}} \dot{u}^2(t) \, dt}{\int_0^{\text{PTW}} u^2(t) \, dt}$$

di mana u(t) adalah perpindahan vertikal dan u̇(t) adalah turunan waktunya (Kanamori, 2005). τ_c jenuh untuk M > 6,5 karena efek durasi ruptur terbatas tetapi memberikan diskriminasi yang sangat baik antar kategori magnitudo yang lebih kecil.

Cumulative Velocity Absolute Displacement (CVAD) didefinisikan sebagai:

$$\text{CVAD} = \int_0^{\text{PTW}} |v(t)| \cdot |d(t)| \, dt$$

di mana v(t) dan d(t) masing-masing adalah bentuk gelombang kecepatan dan perpindahan. CVAD meningkat secara monoton dengan PTW dan tidak menunjukkan saturasi teoretis, menjadikannya ideal untuk regresi berkelanjutan.

**Table 2.** Dua belas ukuran intensitas dengan klasifikasi saturasi.

| Feature | Description | Saturation |
|---|---|---|
| τ_c (bandpass) | Predominant period (0.075–3 Hz BP) | **Saturating** |
| τ_c (raw) | Predominant period (unfiltered) | Saturating |
| P_d (3-comp) | Peak displacement (geometric mean) | Saturating |
| T_p | Predominant half-period | Saturating |
| P_v (3-comp) | Peak velocity | Partially saturating |
| T_va | Velocity-to-acceleration time ratio | Mixed |
| P_a (3-comp) | Peak acceleration | Non-saturating |
| IV² | Integral velocity squared | Non-saturating |
| CAV | Cumulative absolute velocity | Non-saturating |
| PIV | P-wave integral velocity (log) | Non-saturating |
| **CVAD** | **Cum. velocity abs. displacement** | **Non-saturating** |
| CVAV | Cum. velocity abs. velocity | Non-saturating |

**30 Zhang Distance Features.** Mengikuti kerangka kerja estimasi jarak stasiun tunggal dari Zhang et al. (2024), kami mengekstrak 30 fitur yang diorganisir ke dalam tiga grup: 12 temporal (amplitudo puncak pada jejak vertikal dan 3-komponen, parameter bentuk envelope A dan B, akselerasi dan kecepatan RMS, kurtosis, skewness, τ_c, frekuensi dominan), 12 spektral (rasio spektral tinggi-rendah dan tinggi-menengah, frekuensi rata-rata, bandwidth, sentroid spektral, rolloff spektral, CAV selama 3 s, intensitas Arias selama 3 s, rasio energi antara setengah pertama dan kedua serta antara derau pra-kejadian dan sinyal), dan 6 koefisien kepstral (c₁–c₅ dan tingkat zero-crossing).

**Preprocessing.** Semua matriks fitur menjalani lima langkah pembersihan berurutan: (i) konversi nilai infinite menjadi NaN, (ii) Winsorization per-kolom pada persentil 0,5 dan 99,5 untuk menekan outlier, (iii) clipping keras ke [−10⁶, 10⁶], (iv) imputasi median per-kolom untuk nilai NaN yang tersisa, dan (v) konversi keamanan akhir dari NaN/Inf residual menjadi nol.

### 2.3 Arsitektur Dual-Stage

Pipeline (Figure 3) terdiri dari tiga komponen berurutan: **pengklasifikasi intensitas** (Stage 1), estimator jarak episentrum (Stage 1.5), dan regressor akselerasi spektral (Stage 2). Semua komponen menggunakan XGBoost (Chen dan Guestrin, 2016), dipilih karena efisiensi komputasinya, regularisasi bawaan, penanganan nilai hilang asli, dan kinerja yang terbukti kuat dalam aplikasi seismologi (Münchmeyer et al., 2021; Lara et al., 2023).

![Figure 3. Schematic of the Saturation-Aware Dual-Stage Framework for Intensity-Driven Adaptive (IDA) PTW. Upon P-wave detection, Stage 1 (Intensity Gate) classifies the trace into *Weak*, *Felt*, or *Damaging* classes using τ_c-dominant features. This intensity-driven routing, combined with predicted distance, selects the optimal PTW constrained by the golden time. Stage 2 regression then predicts Sa(T) using CVAD-dominant features.](figures/fig03.png)

#### 2.3.1 Stage 1: Pengklasifikasi Intensity Gate

Stage 1 mengklasifikasikan setiap jejak menjadi salah satu dari tiga kelas intensitas operasional berdasarkan 42 fitur yang diekstrak pada PTW = 2 s — jendela paling awal yang tersedia — menggunakan pengklasifikasi XGBoost dengan objektif `multi:softprob`. Tiga kelas (*Weak*, *Felt*, dan *Damaging*) didefinisikan oleh ambang batas PGA mengikuti skema intensitas ShakeMap BMKG 2023: *Weak* (PGA < 3,0 Gal, MMI ≤ II), *Felt* (3,0 ≤ PGA < 62,0 Gal, MMI III–V), dan *Damaging* (PGA ≥ 62,0 Gal, MMI ≥ VI). Meskipun kelas intensitas ini berkorelasi dengan magnitudo gempa bumi, kerangka kerja IDA memprioritaskan kekuatan sinyal yang teramati di situs stasiun untuk memastikan ketahanan terhadap efek situs lokal.

Regularisasi kuat diterapkan untuk mencegah overfitting pada kelas *Damaging* yang parah tidak seimbang: max_depth = 5 membatasi kompleksitas pohon, min_child_weight = 30 memerlukan dukungan sampel yang substansial, dan reg_lambda = 3,0 menerapkan penalisasi L2 agresif. Detail hyperparameter lengkap disediakan dalam Supplementary Table S1.

#### 2.3.2 Stage 1.5: Estimator Jarak Episentrum

Untuk memungkinkan operasi yang sepenuhnya otonom tanpa informasi katalog, regressor jarak dilatih pada 30 fitur Zhang et al. (2024) yang diekstrak pada PTW = 2 s menggunakan regressor XGBoost (max_depth = 7, learning_rate = 0,05, min_child_weight = 10). Jarak yang diestimasi D̂ berfungsi sebagai input ke mekanisme seleksi PTW, bukan sebagai produk prediksi mandiri. Peran fungsional ini berarti bahwa **akurasi rute lebih penting daripada akurasi jarak** — perbedaan yang dikonfirmasi oleh kinerja E2E yang hampir identik antara rute jarak prediksi dan kebenaran dasar (Section 3.4).

#### 2.3.3 Seleksi PTW Sadar Jarak dengan Batasan Waktu Emas

Kelas intensitas prediksi (Weak, Felt, Damaging) dipetakan ke tingkat guncangan representatif yang mana, dikombinasikan dengan jarak prediksi D̂, mengindeks tabel pencarian **Intensitas–Jarak** 6 × 6 (Table S2). PTW yang dipilih kemudian dibandingkan terhadap waktu emas gelombang-S untuk jarak prediksi untuk memastikan kelayakan peringatan:

$$\text{PTW}_{\text{selected}} \leq t_{\text{golden}}(\hat{D}) - t_{\text{warn}}$$

di mana waktu emas adalah perbedaan waktu tempuh gelombang S−P:

$$t_{\text{golden}}(\hat{D}) = \frac{\hat{D}}{V_S} - \frac{\hat{D}}{V_P}$$

dengan V_P = 6,5 km/s dan V_S = 3,5 km/s (representatif untuk kerak Indonesia), dan t_warn = 3 s adalah buffer peringatan minimum yang disediakan untuk penyebaran peringatan dan tindakan perlindungan (Minson et al., 2018). Figure 9 mengilustrasikan waktu emas sebagai fungsi jarak dan wilayah kelayakan PTW.

![Figure 9. Golden time constraint and PTW feasibility. The blue curve shows the S−P travel-time difference (golden time) as a function of epicentral distance. The green dashed line and shaded region indicate the maximum feasible PTW (golden time minus 3 s warning buffer). At the dataset median distance of 384 km, the golden time is ~51 s, providing ample budget for adaptive PTW selection up to 8 s. Near-field events (D < 22 km) have golden times shorter than the minimum useful PTW of 2 s.](figures/fig09.png)

#### 2.3.4 Stage 2: Regressor Akselerasi Spektral

Untuk setiap kombinasi periode spektral (103 periode, T = 0,1–5,0 s) dan PTW (5 jendela: 2, 3, 4, 6, 8 s), regressor XGBoost individual memprediksi log₁₀ Sa(T) dalam satuan m/s², menghasilkan 515 model. Hyperparameter dioptimalkan melalui pencarian Bayesian Optuna 100-trial (Akiba et al., 2019) pada subset representatif, tiba di max_depth = 9, learning_rate = 0,011, subsample = 0,67, dan reg_lambda = 0,68 (Supplementary Table S2). Transformasi log₁₀ menstabilkan varians dan melinierkan hubungan magnitudo–gerakan tanah.

### 2.4 Anti-Overfitting: Median Iteration Selection

Kontribusi metodologi kunci adalah strategi *median-based iteration selection*. Ketika melatih model produksi akhir pada 100% data yang tersedia, tidak ada set validasi holdout yang ada untuk sinyal kapan harus berhenti boosting. Praktik standar baik menggunakan semua n_estimators (berisiko overfit) atau meninggalkan model full-data sepenuhnya.

Pendekatan kami menyelesaikan ini dengan: (1) melakukan validasi silang 5-fold GroupKFold (Hastie et al., 2009), dengan grup didefinisikan oleh ID kejadian gempa bumi sehingga tidak ada kejadian seismik yang berkontribusi jejak ke split pelatihan dan validasi; (2) merekam best_iteration early-stopping untuk setiap fold; (3) menghitung n_optimal = median(best_iteration₁, ..., best_iteration₅); dan (4) melatih model produksi pada semua data dengan n_estimators = n_optimal. Median dipilih daripada mean untuk memberikan ketahanan terhadap fold outlier. Ini memastikan kompleksitas model akhir diatur oleh perilaku konvergensi yang representatif secara statistik daripada split holdout tunggal yang sewenang-wenang.

### 2.5 Metodologi Evaluasi

Kinerja end-to-end (E2E) dievaluasi dengan mensimulasikan alur operasional penuh: Stage 1 mengklasifikasikan (fitur PTW = 2 s), Stage 1.5 memperkirakan jarak (PTW = 2 s), tabel pencarian menetapkan PTW, dan model Stage 2 yang sesuai memprediksi log₁₀ Sa(T). Komposit R² dan RMSE dihitung dengan menggabungkan semua prediksi di 103 periode secara bersamaan, mengikuti kerangka kerja evaluasi komposit yang digunakan dalam perbandingan Ground Motion Model (Abrahamson et al., 2014). Tujuh strategi dibandingkan: lima baseline PTW tetap (2, 3, 4, 6, 8 s), oracle sadar jarak (jarak katalog kebenaran dasar), dan strategi IDA-PTW operasional sepenuhnya menggunakan jarak prediksi. Kami menyebutnya protokol "**Honest Evaluation**", karena secara ketat melarang penggunaan informasi katalog kebenaran dasar apa pun (magnitudo atau jarak) selama inferensi, sehingga mencerminkan kondisi lapangan aktual dari deployment on-site otonom.

---

## 3. Hasil

### 3.1 Stage 1: Kinerja Klasifikasi dan Dikotomi Fitur

Pengklasifikasi intensitas (Stage 1) mencapai akurasi out-of-fold 93,01% di seluruh validasi silang 5-fold yang dikelompokkan berdasarkan peristiwa (Table 3). Kesalahan klasifikasi terutama melibatkan jejak dengan intensitas tingkat batas, di mana kesalahan rute (misalnya *Weak* ditugaskan ke *Felt*) dikurangi oleh envelope kinerja yang tumpang tindih dari jendela PTW yang berdekatan. Penting untuk diperhatikan, pengklasifikasi mempertahankan **Damaging Recall sebesar 72,3%**, berhasil memperluas jendela pengamatan untuk mayoritas jejak intens.

Analisis pentingnya fitur mengungkapkan *Dikotomi Fitur* yang mencolok antara dua tahap pipeline (Figure 4). Pada Stage 1, parameter saturasi τ_c menyumbang 51% dari total pentingnya — efek ceilnya pada M > 6,5 menciptakan batas keputusan alami yang dimanfaatkan pengklasifikasi untuk kategorisasi intensitas. Pada Stage 2, CVAD yang tidak jenuh mendominasi dengan pentingnya 89,5%, karena penskalaan monoton dan tak terbatasnya memberikan diskriminasi berkelanjutan di seluruh rentang dinamis Sa(T) penuh. Partisi ini muncul secara organik dari proses seleksi fitur XGBoost tanpa rekayasa fitur manual.

**Table 3.** Kinerja Stage 1 Intensity Gate (out-of-fold).

| Metric | Weak | Felt | Damaging | Aggregate |
|---|---|---|---|---|
| **Precision** | 97.2% | 41.6% | 71.6% | — |
| **Recall** | 95.4% | 53.9% | **72.3%** | — |
| **F1-Score** | 96.3% | 47.0% | 71.9% | — |
| **Overall Accuracy** | — | — | — | **93.01%** |
| **Critical Miss Rate** | — | — | — | **8.9%** |

![Figure 4. Feature Dichotomy for the IDA Scenario. (a) Stage 1 Intensity Classifier: saturating features dominate (τ_c). (b) Stage 2 Regressor: non-saturating features dominate (CVAD). This dichotomy is preserved in the regional Java-Sumatra dataset.](figures/fig04.png)

### 3.2 Stage 2: Kinerja Regresi Spektral

Semua 515 regressor Sa(T) melatih dengan sukses tanpa kegagalan. Table 4 merangkum metrik agregat per-PTW, dan Figure 5 memvisualisasikan distribusi R². R² meningkat secara monoton dari 0,571 (PTW = 2 s) hingga 0,589 (PTW = 8 s), mengonfirmasi bahwa jendela pengamatan yang lebih lama memberikan prediksi spektral yang semakin baik. RMSE mean yang sesuai menurun dari 0,632 menjadi 0,619 log₁₀ m/s². Nilai R² model individual berkisar dari 0,481 (Sa periode pendek pada PTW = 2 s) hingga 0,644 (Sa periode panjang pada PTW = 8 s).

**Table 4.** Kinerja Stage 2 OOF menurut jendela PTW (teragregasi di atas 103 periode spektral).

| PTW (s) | R² Mean | R² Median | R² Min | R² Max | RMSE Mean |
|---|---|---|---|---|---|
| 2 | 0.571 | 0.586 | 0.481 | 0.623 | 0.632 |
| 3 | 0.576 | 0.590 | 0.484 | 0.628 | 0.629 |
| 4 | 0.578 | 0.592 | 0.486 | 0.630 | 0.627 |
| 6 | 0.585 | 0.600 | 0.490 | 0.640 | 0.622 |
| **8** | **0.589** | **0.604** | 0.494 | **0.644** | **0.619** |

Tren monoton ini memiliki interpretasi fisik yang jelas: jendela PTW yang lebih lama menangkap konten gelombang P frekuensi rendah (f < 1 Hz) yang membawa informasi tentang momen seismik dan dengan demikian berkorelasi dengan Sa periode panjang. Keuntungan R² inkremental berkurang dengan setiap ekstensi jendela (Δ = +0,005 dari 2→3 s versus Δ = +0,004 dari 6→8 s), menunjukkan pengembalian yang berkurang di luar PTW ≈ 8 s untuk set fitur yang digunakan.

![Figure 5. Distribution of OOF R² across 103 spectral periods. The boxplot shows the model accuracy distribution for the filtered IDA-PTW operational scenario.](figures/fig05.png)

### 3.3 Evaluasi Pipeline End-to-End

Table 5 menyajikan metrik E2E komposit di seluruh tujuh strategi, dan Figure 6 memvisualisasikan perbandingan. Baseline PTW tetap mencapai R² = 0,7454–0,7584, dengan jendela yang lebih lama menghasilkan akurasi yang lebih baik secara monoton. Strategi operasional IDA-PTW mencapai R² komposit = 0,7309 — kira-kira 1,5–2,8% lebih rendah daripada baseline tetap — mewakili trade-off akurasi-keselamatan yang disengaja. Oracle IDA-PTW (jarak kebenaran dasar) mencapai R² = 0,7311, secara efektif identik dengan hasil operasional, menunjukkan ketahanan ekstrem terhadap kesalahan estimasi jarak. Fixed-8s memberikan akurasi maksimal tetapi tidak layak secara operasional untuk peristiwa near-field karena pelanggaran waktu emas. IDA-PTW malah mengompresi secara dinamis jendela pengamatan untuk peristiwa berisiko tinggi near-field sambil mempertahankan akurasi untuk guncangan intensitas lebih rendah atau jauh. Penting untuk diperhatikan, kalibrasi ini dicapai di bawah kerangka kerja "Honest Evaluation" di mana semua sinyal rute (kelas intensitas dan jarak) diprediksi secara otonom dari hanya 3 detik data gelombang P awal.

**Table 5.** Metrik komposit end-to-end di atas 103 periode Sa(T).

| Strategy | Composite R² | RMSE (log₁₀ m/s²) | ΔR² vs Fixed-3s |
|:---:|:---:|:---:|:---:|
| Fixed PTW = 2 s | 0.7454 | 0.5154 | -0.0004 |
| **Fixed PTW = 3 s (Baseline)** | **0.7458** | **0.5151** | — |
| Fixed PTW = 4 s | 0.7495 | 0.5113 | +0.0037 |
| Fixed PTW = 6 s | 0.7517 | 0.5090 | +0.0059 |
| Fixed PTW = 8 s | 0.7584 | 0.5021 | +0.0126 |
| IDA-PTW + Distance (Ground Truth) | 0.7311 | 0.5179 | -0.0147 |
| **IDA-PTW Operational (Predicted)** | **0.7309** | **0.5181** | **-0.0149** |

Penting untuk diperhatikan, nilai R² komposit (0,73–0,76) secara substansial melebihi nilai R² OOF per-model (0,57–0,59) dari Table 4. Ketidaksesuaian yang tampak ini timbul karena R² komposit menggabungkan prediksi di seluruh 103 periode — termasuk nilai Sa periode panjang yang berkorelasi kuat di mana konsistensi antar periode meningkatkan statistik agregat.

![Figure 6. End-to-end composite performance comparison (IDA scenario). The IDA-PTW Operational strategy (orange) achieves competitive R² while prioritizing safety and lead time.](figures/fig06.png)

### 3.4 Rute Operasional dan Ketahanan Jarak

Kinerja sangat tahan terhadap kesalahan estimasi jarak: sistem berfungsi hampir sebaik ketika menggunakan estimasi jaraknya sendiri yang tidak sempurna (R² = 0,7309) seperti halnya ketika menggunakan jarak katalog sempurna (R² = 0,7311), perbedaan hanya ΔR² = 0,0002. Ketahanan ini berasal dari fakta bahwa rute sadar jarak terutama berfungsi untuk memberlakukan batasan keselamatan waktu emas daripada mengoptimalkan untuk keuntungan R² marjinal. Seperti ditunjukkan pada Figure 7 dan Table 6, rute operasional memusatkan 99,9% jejak pada PTW = 3–4 s, distribusi konservatif yang memprioritaskan ketepatan waktu peringatan. Rute jarak prediksi menugaskan lebih sedikit jejak ke jendela yang lebih lama (PTW ≥ 6 s) dibandingkan rute kebenaran dasar (0,07% versus 17,06%), menunjukkan bahwa kesalahan estimasi jarak cenderung mengompresi distribusi PTW menuju jendela yang lebih pendek — mode kegagalan sisi aman yang mempertahankan lead time dengan biaya akurasi yang dapat diabaikan.

**Table 6.** Distribusi durasi PTW yang ditetapkan untuk 34.033 jejak. Kolom "Intensity-Only" mencerminkan penetapan PTW berdasarkan hanya kelas intensitas IDA tanpa koreksi jarak (Weak → 3 s, Felt → 4 s, Damaging → 6 s), mewakili mode rute latensi minimum.

| PTW (s) | Intensity-Only (No Distance) | Ground Truth (GT Distance) | Operational (Pred. Distance) |
|:---:|:---:|:---:|:---:|
| 2 | 0 (0.0%) | 10 (0.03%) | 13 (0.04%) |
| 3 | 32,001 (94.03%) | 18,242 (53.60%) | 24,020 (70.58%) |
| 4 | 1,931 (5.67%) | 9,975 (29.31%) | 9,974 (29.31%) |
| 6 | 101 (0.30%) | 5,800 (17.04%) | 26 (0.07%) |
| 8 | 0 (0.0%) | 6 (0.02%) | 0 (0.0%) |

Karena akurasi Stage 2 secara monoton terhubung ke durasi jendela, bias rute "konservatif" ini secara efektif mengoptimalkan sistem untuk akurasi daripada ketepatan waktu, dengan syarat anggaran waktu emas memungkinkan.

![Figure 7. Operational Routing Distribution. Highlights the probability of a trace being assigned to specific PTWs based on the IDA Intensity Gate and golden time constraints.](figures/fig07.png)

### 3.5 Kinerja Tergantung Periode

R² di seluruh 103 periode spektral berkisar dari 0,544 (Sa periode pendek, T ≈ 0,3 s) hingga 0,709 (T = 5,0 s), dengan mean 0,646 dan median 0,663 (Figure 8). Peningkatan monoton dalam R² dengan panjang periode mencerminkan kopling fisik antara parameter gelombang P dan Sa(T), dan penurunan pada periode pendek (T = 0,3–0,6 s, R² ≈ 0,54) dapat dikaitkan dengan tiga faktor gabungan. Pertama, Sa periode pendek terutama dikendalikan oleh kondisi situs lokal — V_S30, resonansi cekungan dangkal, dan kontras impedansi permukaan dekat (Seyhan dan Stewart, 2014; Borcherdt, 1994) — yang bervariasi secara substansial antar stasiun tetapi tidak diwakili dalam set input fitur gelombang P 42 kami. Ketiadaan parameter karakterisasi situs merupakan sumber utama variabilitas aleatori pada periode ini. Kedua, energi seismik frekuensi tinggi (f > 2 Hz) yang mengendalikan Sa periode pendek mengalami atenuasi anelastik yang kuat di sepanjang jalur penyebaran melalui kerak subduksi Sunda yang heterogen, dengan nilai Q yang bervariasi di sepanjang jalur sinar yang berbeda, semakin meningkatkan ketidakpastian prediksi pada periode pendek. Ketiga, jendela gelombang P 3 detik awal secara fundamental menangkap konten frekuensi rendah yang terkait dengan momen seismik (Kanamori, 2005; Lancieri dan Zollo, 2008), yang berkorelasi kuat dengan Sa periode panjang (T > 1 s) tetapi memberikan informasi terbatas tentang detail ruptur frekuensi tinggi (stress drop, directivity) yang mempengaruhi amplitud spektral periode pendek. Ketiga faktor ini dikomplikasikan oleh mekanisme keempat: distribusi energi spektral dalam band energi seismik dominan (2–10 Hz, sesuai dengan T = 0,1–0,5 s) sangat bervariasi di seluruh kejadian dan stasiun. Gempa bumi dengan parameter sumber serupa dapat menampilkan puncak energi pada frekuensi berbeda dalam band ini karena karakteristik ruptur stokastik, dan variabilitas intra-band ini semakin diperkuat oleh respons yang bergantung pada frekuensi spesifik situs — stasiun di tanah lunak dengan periode situs dominan (T_dom) di dekat 0,3 s secara selektif mengamplifikasi Sa pada periode tersebut sementara situs batuan tidak (Borcherdt, 1994). Karena V_S30 maupun T_dom tidak disertakan sebagai fitur input, variabilitas spektral sumber-situs gabungan dalam band fokus energi menjadi derau yang tidak dapat dikurangi dari perspektif model. Dasar fisik untuk kinerja tergantung periode ini mapan dalam teori gerakan tanah. Akselerasi spektral pada periode apa pun dihasilkan dari efek gabungan sumber, jalur, dan situs: Sa(T) = f(Source) × f(Path) × f(Site) (Kramer, 1996). Pada periode panjang (T > 1 s, f < 1 Hz), Sa didominasi oleh momen seismik — properti sumber yang stabil yang 42 fitur gelombang P kami tangkap secara efektif, karena gelombang frekuensi rendah merambat dengan atenuasi minimal dan tidak sensitif terhadap lapisan tanah dangkal (panjang gelombang >> ketebalan sedimen). Pada periode pendek (T < 0,5 s, f > 2 Hz), keseimbangan bergeser secara dramatis: (a) respons situs menjadi faktor dominan, karena lapisan sedimen dangkal dengan frekuensi resonan f₀ = V_S / 4H secara selektif mengamplifikasi Sa pada periode tertentu tergantung V_S30 lokal dan ketebalan lapisan (Borcherdt, 1994; Kramer, 1996); (b) parameter atenuasi frekuensi tinggi kappa (κ), yang meluruh amplitud spektral secara eksponensial sebagai A(f) = A₀·exp(−πκf), bervariasi 0,01–0,08 s antar stasiun (Anderson dan Hough, 1984); dan (c) variabilitas tingkat sumber meningkat karena stress drop — kontrol utama radiasi frekuensi tinggi — bersifat stokastik dan dapat bervariasi 3–10× antara kejadian magnitudo serupa (Boore, 2003). Karena pipeline kami tidak memiliki akses ke V_S30, κ, atau stress drop, ketiga sumber variabilitas ini tidak dapat dikurangi dari perspektif model. Kinerja yang dihasilkan bervariasi secara sistematis di seluruh tiga pita spektral: periode pendek (T ≤ 0,6 s, mean R² = 0,56) di mana efek situs dan jalur mendominasi, periode menengah (T = 0,6–2,0 s, mean R² = 0,60) mewakili zona transisi di mana momen seismik mulai berkontribusi bersama dengan pengaruh situs yang berkurang, dan periode panjang (T > 2,0 s, mean R² = 0,68) di mana momen seismik sepenuhnya mendominasi dan fitur gelombang P kami paling informatif. Kesenjangan R² ~0,12 antara pita periode pendek dan panjang konsisten dengan sigma aleatori yang lebih tinggi yang dilaporkan pada periode pendek dalam GMPE NGA-West2 (Abrahamson et al., 2014).

![Figure 8. Per-period Prediction Accuracy for the IDA-PTW Pipeline. R² and RMSE across 103 periods for the Java-Sumatra filtered dataset.](figures/fig08.png)

### 3.6 Kinerja Keselamatan dan Ketahanan Respons Situs

#### 3.6.1 Keselamatan Operasional dan Analisis Critical Miss

Dalam konteks Peringatan Dini Gempa Bumi (EEWS), tujuan utama adalah memaksimalkan lead time untuk gerakan tanah yang berpotensi merusak sambil meminimalkan "critical misses"—skenario di mana guncangan intensitas tinggi terjadi tanpa jendela peringatan yang cukup. Untuk mengevaluasi hal ini, kami membandingkan strategi **Intensity-Driven Adaptive (IDA) PTW** terhadap baseline **Source-Aware (berbasis Magnitudo)** menggunakan dataset area penelitian yang disaring secara geografis (34.033 jejak).

Hasil menunjukkan kontras yang mencolok dalam keselamatan operasional. Strategi IDA-PTW mencapai **Damaging Recall sebesar 72,3%**, berhasil mengidentifikasi dan memperluas jendela pemrosesan untuk mayoritas besar peristiwa intensitas tinggi. Sebaliknya, model Source-Aware mencapai **recall 1,3%** yang dapat diabaikan untuk intensitas merusak. Ketidaksesuaian ini menunjukkan bahwa rute sentrik magnitudo secara fundamental "buta" terhadap guncangan intensitas tinggi ketika itu didorong oleh faktor selain energi sumber.

**Table 7.** Metrik keselamatan untuk rute IDA vs. Source-Aware pada dataset Sunda Arc yang disaring.

| Strategy | Recall (Damaging Class) | Critical Miss Rate (Damaging → Weak) |
| :--- | :---: | :---: |
| **IDA-PTW (Proposed)** | **72.3%** | **8.9%** |
| **Source-Aware (Baseline)** | **1.3%** | **~98%** |

#### 3.6.2 Ketahanan terhadap Efek Situs Lokal

Kegagalan pendekatan Source-Aware dalam mendeteksi peristiwa merusak menyoroti realitas fisik yang kritis dari lingkungan seismik Indonesia: prevalensi efek situs terlokalisasi. Dari 101 jejak yang diklasifikasikan sebagai **Damaging** (PGA ≥ 62,0 Gal) dalam area penelitian (Table 1), sebagian besar berasal dari gempa bumi dengan magnitudo sedang (M < 5,0).

Dari perspektif inversi sumber, peristiwa Magnitudo 4,5 biasanya dianggap tidak merusak dan karenanya dirute oleh logika Source-Aware ke jendela gelombang P standar 3 detik. Namun, karena dinamika kerak dangkal atau kondisi tanah yang tidak terkonsolidasi (efek cekungan), peristiwa "minor" ini dapat menghasilkan akselerasi puncak tanah yang merusak di situs stasiun tertentu. Dengan memanfaatkan **akselerograf**—yang menangkap getaran onsite frekuensi tinggi—dan menerapkan logika **Intensity-Driven (IDA)**, sistem yang diusulkan merespons langsung terhadap **intensitas yang teramati** di sensor. Ini membuat IDA-PTW secara inheren lebih tahan terhadap amplifikasi situs lokal, memastikan bahwa bahkan peristiwa magnitudo sedang yang memicu gerakan tanah merusak diberikan waktu pemrosesan yang diperpanjang (hingga 6–8 s tergantung jarak) untuk prediksi akselerasi spektral yang akurat.

### 3.7 Metrik Kinerja Komprehensif (Kerangka Kerja Dai et al., 2024)

Untuk memfasilitasi perbandingan langsung dengan studi prediksi Sa(T) sebelumnya, kami mengevaluasi pipeline IDA-PTW operasional penuh menggunakan kerangka kerja metrik komprehensif yang diusulkan oleh Dai et al. (2024), yang menguraikan kinerja model menjadi enam dimensi diagnostik komplementer. Semua metrik dihitung pada prediksi operasional end-to-end (kelas intensitas prediksi dan jarak prediksi) menggunakan validasi silang 5-fold yang dikelompokkan berdasarkan peristiwa, ketat mengikuti protokol Honest Evaluation.

#### 3.7.1 Dekomposisi Variabilitas (τ, φ, σ)

Mengikuti Al Atik et al. (2010), kami menguraikan variabilitas aleatori total (σ) dari residual prediksi menjadi komponen inter-event (τ) dan intra-event (φ). Hasilnya menghasilkan σ_total ≈ 0,752 satuan log₁₀, dengan komponen intra-event φ mendominasi pada sekitar 70% dari total varians. Dominasi φ daripada τ ini mengungkapkan bahwa variabilitas situs-ke-situs dan bergantung-jalur merupakan sumber utama ketidakpastian prediksi — bukan perbedaan sumber gempa bumi. Dibandingkan dengan GMPE konvensional seperti NGA-West2 (Abrahamson et al., 2014; Chiou dan Youngs, 2014), yang melaporkan σ_total ≈ 0,65–0,75, model IDA-PTW mencapai tingkat variabilitas yang sebanding meskipun beroperasi di bawah batasan EEWS stasiun tunggal yang parah dengan hanya 2–8 detik data gelombang P dan tidak ada parameter karakterisasi situs. Dominasi φ yang jelas langsung menyiratkan bahwa pengayaan data Vs30 adalah jalur perbaikan paling berdampak untuk mengurangi ketidakpastian prediksi total (Figure 10).

#### 3.7.2 Metrik Akurasi Tergantung Periode (R², RMSE, Bias)

R² meningkat secara monoton dari sekitar 0,30 pada periode struktural pendek (T ≈ 0,1–0,3 s) menjadi sekitar 0,49 pada periode panjang (T ≈ 4–5 s), sementara RMSE menurun secara bersesuaian. Tren ini konsisten secara fisik dengan teori gelombang seismik: konten frekuensi rendah yang mendominasi Sa periode panjang lebih baik dipertahankan dan direkonstruksi dari kedatangan gelombang P awal. Secara kritis, mean bias tetap di dekat nol (dalam ±0,05 satuan log₁₀) di seluruh 103 periode, mengonfirmasi bahwa model tidak menunjukkan kecenderungan overprediksi atau underprediksi sistematis — persyaratan fundamental untuk deployment EEWS operasional. Pola ini selaras dengan temuan yang mapan dari studi EEWS menggunakan metodologi berbeda (Kodera et al., 2018; Böse et al., 2009), memvalidasi dasar fisik dari pendekatan berbasis XGBoost (Figure 11).

#### 3.7.3 Akurasi Dalam-Faktor

Analisis within-factor mengungkapkan bahwa 83,3% prediksi jatuh dalam ±1,0 satuan log₁₀, 54,4% dalam ±0,5 log₁₀, dan 30,8% dalam margin terketat ±0,3 log₁₀. Akurasi secara khusus lebih tinggi pada periode yang lebih lama (T > 1,0 s), yang sangat signifikan untuk aplikasi EEWS karena akselerasi spektral periode panjang berkorelasi paling kuat dengan kerusakan struktural di gedung-gedung menengah hingga tinggi — target utama untuk tindakan perlindungan. Dalam istilah praktis, untuk bangunan dengan periode fundamental T = 2,0 s, model memprediksi Sa dalam faktor sekitar 3 untuk lebih dari setengah kasus, memungkinkan penilaian kerusakan cepat tingkat rekayasa yang bermakna dalam hitungan detik setelah kedatangan gelombang P (Figure 12).

#### 3.7.4 Distribusi R² Magnitudo-Periode

Peta panas magnitudo-periode mengungkapkan batasan saturasi yang melekat pada semua pendekatan EEWS stasiun tunggal. Model berkinerja sangat baik untuk gempa bumi sedang (M 4,0–6,0), di mana R² secara konsisten melebihi 0,5 di seluruh semua periode struktural. Namun, kinerja menurun secara parah untuk M ≥ 6,5, dengan nilai R² menjadi negatif untuk M 7+, menunjukkan prediksi lebih buruk dari mean sampel. Ini bukan kekurangan pemodelan tetapi batasan fisika fundamental yang terdokumentasi dengan baik dalam literatur EEWS (Yamada dan Mori, 2009; Zollo et al., 2010): fitur gelombang P awal (2–8 s) pada stasiun tunggal tidak dapat menangkap seluruh luas ruptur gempa bumi besar yang bidang patahan dapat melebihi 100 km. Model IDA-PTW tetap sangat efektif dalam domain operasionalnya (M 4,0–6,0, mewakili >95% sismisitas Indonesia), sementara pekerjaan masa depan pada koreksi bias tergantung magnitudo sadar saturasi dan fusi data multi-stasiun — seperti pendekatan gaya PLUM (Kodera et al., 2018) — dapat secara progresif mengatasi batasan ini untuk peristiwa besar yang langka (Figure 13).

#### 3.7.5 Diagnostik Residual (Diamati vs. Diprediksi, Q-Q Plot)

Plot pencar yang diamati versus diprediksi menunjukkan pengelompokan ketat di sekitar garis referensi 1:1 di seluruh 103 periode, dengan underprediksi yang karakteristik terlihat hanya pada nilai ekstrem tinggi yang sesuai dengan rekaman magnitudo besar dan near-field — konsisten dengan batasan saturasi yang diidentifikasi dalam Section 3.7.4. Plot Q-Q dari residual mengonfirmasi distribusi yang hampir normal dengan ekor berat yang sedikit, memvalidasi kerangka kerja statistik lognormal standar dalam seismologi rekayasa (Joyner dan Boore, 1981; Abrahamson dan Silva, 2008). Konfirmasi ganda ini dari model unbiasedness memastikan bahwa perhitungan hazard seismik probabilistik menggunakan prediksi IDA-PTW akan menghasilkan estimasi probabilitas pelampauan yang dapat diandalkan untuk pengambilan keputusan rekayasa di bawah kerangka kerja SNI 1726:2019 (Figure 14).

#### 3.7.6 Bias Tergantung Jarak

Mean bias sebagai fungsi jarak sumber-stasiun mengungkapkan ketergantungan jarak yang kuat. Pada jarak sangat pendek (D < 30 km), prediksi menunjukkan underprediksi parah dengan nilai bias berkisar −1,5 hingga −2,0 satuan log₁₀, disebabkan oleh efek near-field kompleks termasuk directivity dan amplifikasi hanging-wall. Di luar 100 km — di mana 94,1% dari rekaman EEWS BMKG operasional terjadi — model sangat terkalibrasi baik dengan bias tetap dekat nol. Ini merupakan batasan utama kedua bersama saturasi magnitudo, dan keduanya berbagi akar penyebab: representasi near-field dan magnitudo besar yang tidak cukup dalam data pelatihan (hanya 5,9% jejak pada D < 100 km). Namun, ini tidak penting secara operasional untuk konfigurasi deployment BMKG saat ini, karena sismisitas subduksi Sunda lepas pantai yang tercatat oleh jaringan akselerograf berbasis darat terutama far-field. Peningkatan masa depan melalui istilah koreksi bias tergantung jarak dan augmentasi data near-field yang ditargetkan dapat secara progresif mengatasi batasan ini seiring jaringan BMKG menjadi lebih padat (Figure 15).
## 4. Diskusi

### 4.1 Dikotomi Fitur sebagai Paradigma Desain

Kontribusi paling signifikan dari penelitian ini adalah demonstrasi empiris Dikotomi Fitur antara tahap klasifikasi dan regresi dalam EEWS dua tahap (Figure 4). Studi EEWS sebelumnya memperlakukan IM gelombang P sebagai kumpulan homogen, memberi makan semua parameter tanpa pandang bulu ke dalam model klasifikasi atau regresi (Wu dan Kanamori, 2005b; Zollo et al., 2010; Münchmeyer et al., 2021). Hasil kami menunjukkan bahwa algoritma XGBoost secara alami mempartisi fitur berdasarkan sifat saturasinya ketika diberikan kesempatan melalui arsitektur dua tahap.

Landasan fisik untuk partisi ini berakar dalam teori sumber seismik. τ_c mencerminkan frekuensi sudut spektrum sumber, yang berskala dengan magnitudo hingga M ≈ 6.5 sebelum mencapai plateau karena durasi ruptur yang terbatas (Kanamori, 2005; Lancieri dan Zollo, 2008). Batas ini menciptakan "langkah" diskret dalam ruang τ_c–magnitudo yang sangat cocok untuk klasifikasi. CVAD, sebagai integral waktu-kumulatif produk velocity–displacement, terus mengakumulasi informasi sepanjang jendela gelombang P tanpa saturasi, memberikan sinyal kontinu untuk regresi.

Paradigma ini menyarankan prinsip desain EEWS yang dapat digeneralisasi: **pisahkan masalah klasifikasi (Kategori apa gempabumi ini?) dari masalah prediksi (Bagaimana pergerakan tanah akan menjadi?), dan izinkan setiap tahap untuk memilih subset fitur optimalnya sendiri berdasarkan kandungan informasi fisik.** EEWS masa depan dapat memperluas ini dengan menggabungkan fitur saturasi tambahan (misalnya P_d) dalam pengklasifikasi dan fitur kumulatif tambahan (misalnya intensitas Arias) dalam regressor.

### 4.2 Kecukupan Routing Berbasis Intensitas dan Ketahanan Jarak

Temuan utama adalah bahwa klasifikasi intensitas Tahap 1 saja — tanpa informasi jarak apa pun — sudah memberikan routing jendela PTW yang efektif. Routing hanya intensitas (Lemah → 3 s, Terasa → 4 s, Merusak → 6 s) mencapai R² komposit = 0.7436, yang *lebih tinggi* daripada routing operasional yang mempertimbangkan jarak (R² = 0.7309) dengan ΔR² = +0.013. Ini terjadi karena strategi hanya-intensitas mengkonsentrasikan 94% jejak pada PTW = 3 s — sudah hampir optimal untuk dataset di mana lokasi sumber yang didominasi lepas pantai di zona subduksi Sunda menempatkan sebagian besar jalur sumber-stasiun dalam rezim jauh (94.1% pada D > 100 km, median 384 km), konsisten dengan cakupan operasional jaringan akselerograf berbasis darat BMKG.

Estimator jarak (Tahap 1.5) oleh karena itu tidak meningkatkan akurasi prediksi dalam dataset saat ini; sebaliknya, perannya adalah arsitektural: ia berfungsi sebagai **batasan keselamatan waktu emas** untuk memastikan bahwa PTW yang dipilih tidak pernah melebihi anggaran waktu perjalanan S−P, khususnya untuk peristiwa near-field di mana waktu peringatan paling ketat. Desain saluran modular memungkinkan Tahap 1.5 untuk dilewati tanpa degradasi kinerja dalam penyebaran yang didominasi jauh, sambil tetap tersedia untuk skenario dengan paparan near-field yang signifikan.

Yang penting, kinerja E2E yang hampir identik antara prediksi dan routing jarak oracle (R² = 0.7309 versus R² = 0.7311, ΔR² = 0.0002) menunjukkan bahwa ketika routing yang mempertimbangkan jarak diaktifkan, sistem sangat tahan terhadap kesalahan estimasi jarak. Ketahanan ini timbul dari dua sifat: (a) jarak hanya berfungsi sebagai sinyal routing diskret di seluruh 6 bin, bukan sebagai input prediksi kontinu — kesalahan yang tetap dalam bin yang sama menghasilkan seleksi PTW yang identik; dan (b) kelas intensitas sudah menangkap variabel routing yang dominan, mengurangi kontribusi marginal jarak. Kesetiaan routing 99.97% mengkonfirmasi bahwa akurasi estimator jarak yang sedang tidak menyebar ke output akhir.

Arsitektur ini menyederhanakan penyebaran operasional: **saluran memerlukan nol informasi eksternal dari jaringan seismik atau katalog. Stasiun tunggal dapat secara mandiri menentukan keseimbangan optimal antara waktu timbal dan akurasi berdasarkan analisis sinyal gelombang P miliknya sendiri saja.** Protokol "Evaluasi Jujur" mengkonfirmasi bahwa sistem mempertahankan akurasi kompetitif dalam kondisi sepenuhnya otonom, stasiun tunggal.

### 4.3 Perbandingan dengan Metode yang Ada

Tabel 8 menyajikan perbandingan kinerja dengan metode EEWS yang telah terbukti dan terbaru, sementara Tabel 9 menyediakan matriks kemampuan yang menyoroti cakupan metodologi setiap pendekatan.

**Tabel 8.** Perbandingan kinerja dengan metode EEWS dan prediksi pergerakan tanah yang ada. Studi dikelompokkan berdasarkan target prediksi. N_T = jumlah periode spektral yang diprediksi. Kinerja dilaporkan sebagai R² komposit di mana tersedia, atau standar deviasi residual σ (dalam unit log₁₀).

| Studi | Metode | Target | N_T | Stasiun Tunggal | PTW | Performa |
|---|---|---|---|---|---|---|
| Wu dan Kanamori (2005b) | τ_c–Pd empiris | PGA | 1 | Ya | Tetap 3 s | R² ≈ 0.70–0.75 |
| Zollo et al. (2010) | Pohon keputusan Bayesian | PGA/PGV | 2 | Ya | Tetap 2–4 s | R² ≈ 0.65–0.80 |
| Hsu dan Huang (2021) | CNN (multi-skala) | PGA | 1 | Ya | Tetap 3 s | Meningkatkan Pd–PGA |
| Wang et al. (2023) | LSTM | PGA | 1 | Ya | 1–30 s | σ = 0.656 (3 s) |
| Liu et al. (2024) | CNN (DLPGA) | PGA | 1 | Ya | Tetap 3–6 s | σ = 0.289 (3 s) |
| Iaccarino et al. (2024) | Gradient Boosting | PGA + Dist | 1 | Ya | Tetap 1–3 s | R² = 0.90–0.95 (3 s) |
| Münchmeyer et al. (2021) | Transformer (TEAM) | PGA (prob.) | 1 | Tidak (multi) | Variabel | σ = 0.29–0.33 |
| Iaccarino et al. (2020) | Empiris (Pd, Iv2) | Sa(T) | 9 | Ya | Tetap 3 s | σ ≈ 0.30–0.35 |
| Lara et al. (2023) | Random Forest | Sa(T) | ~20 | Tidak (jaringan) | Tetap 4 s | R² = 0.88–0.92 |
| Dai et al. (2024) | XGBoost | Sa(T) | Banyak | Ya | Tetap 3–10 s | MSE = 15.1×10⁻⁴ g |
| Abrahamson et al. (2014) | NGA-West2 GMPE | Sa(T) | ~20 | Tidak (jaringan+tapak) | Setelah peristiwa | σ_total ≈ 0.65–0.80 |
| **Studi ini** | **XGBoost Dua Tahap** | **Sa(T)** | **103** | **Ya** | **Adaptif 3–8 s** | **R² = 0.73** |

**Tabel 9.** Matriks kemampuan. Tanda centang (✓) menunjukkan studi mengatasi kemampuan yang sesuai; tanda hubung (–) menunjukkan tidak ditangani. Kemampuan: (A) Memprediksi Sa(T) spektral penuh, (B) Cakupan spektral > 20 periode, (C) Stasiun tunggal (tanpa agregasi jaringan), (D) Tanpa informasi katalog yang diperlukan (sepenuhnya otonom), (E) Jendela waktu gelombang P adaptif, (F) Melaporkan metrik keselamatan (misalnya Damaging recall), (G) Validasi silang dikelompokkan berdasarkan peristiwa.

| Studi | (A) Sa(T) | (B) >20 T | (C) Stasiun Tunggal | (D) Tanpa katalog | (E) PTW Adaptif | (F) Keselamatan | (G) Event-CV |
|---|---|---|---|---|---|---|---|
| Wu dan Kanamori (2005b) | – | – | ✓ | ✓ | – | – | – |
| Zollo et al. (2010) | – | – | ✓ | ✓ | – | ✓ | – |
| Hsu dan Huang (2021) | – | – | ✓ | ✓ | – | – | – |
| Wang et al. (2023) | – | – | ✓ | ✓ | – | – | – |
| Liu et al. (2024) | – | – | ✓ | ✓ | – | – | – |
| Iaccarino et al. (2024) | – | – | ✓ | ✓ | – | – | – |
| Münchmeyer et al. (2021) | – | – | – | – | – | ✓ | ✓ |
| Iaccarino et al. (2020) | ✓ | – | ✓ | ✓ | – | ✓ | – |
| Lara et al. (2023) | ✓ | – | – | – | – | – | – |
| Dai et al. (2024) | ✓ | – | ✓ | ✓ | – | – | – |
| Abrahamson et al. (2014) | ✓ | ✓ | – | – | – | – | – |
| **Studi ini** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Tabel 8 dan 9 mengungkapkan empat temuan utama mengenai penempatan karya ini. Pertama, tidak ada EEWS stasiun tunggal sebelumnya yang memprediksi Sa(T) pada lebih dari ~20 periode spektral; studi ini memperluas cakupan hingga 103 periode (T = 0.1–5.0 s), peningkatan 5–10× yang memungkinkan penilaian bahaya khusus struktur. Kedua, studi yang mencapai R² lebih tinggi (misalnya Lara et al., 2023: R² = 0.88–0.92; Iaccarino et al., 2024: R² = 0.90–0.95) beroperasi di bawah kondisi yang jauh lebih mudah — agregasi multi-stasiun, rentang jarak lebih pendek (< 300 km), target hanya-PGA, atau dataset regional dengan heterogenitas lebih rendah. R² = 0.73 kami dicapai dalam konfigurasi paling menantang: stasiun tunggal, tanpa karakterisasi tapak, jarak hingga 867 km, dan 103 target spektral secara bersamaan. Ketiga, kemampuan PTW adaptif (kolom E) unik untuk studi ini; semua metode sebelumnya menggunakan jendela waktu tetap, kehilangan tradeoff eksplisit antara akurasi prediksi dan своевременность peringatan. Keempat, matriks kemampuan menunjukkan bahwa studi ini adalah satu-satunya pendekatan yang memenuhi ketujuh kriteria secara bersamaan, memposisikannya sebagai kerangka kerja EEWS stasiun tunggal paling komprehensif yang saat ini tersedia untuk prediksi spektral kelas teknik.

### 4.4 Implikasi untuk EEWS Indonesia

Hasil ini menawarkan implikasi spesifik untuk meningkatkan infrastruktur InaTEWS Indonesia:

**Pelengkap near-field.** Saluran ini dapat berfungsi sebagai modul estimasi-cepat-awal yang beroperasi dalam 3 detik pertama kedatangan gelombang P, memberikan estimasi Sa(T) tingkat-stasiun sebelum solusi berbasis jaringan InaTEWS menyatu. Untuk gempabumi M 7.5 Palu 2018, di mana solusi jaringan memerlukan ~30 s, estimasi PTW adaptif stasiun tunggal bisa tersedia dalam 5 s.

**Ketahanan jaringan jarang.** Banyak bagian timur Indonesia memiliki jarak stasiun melebihi 200 km. Untuk peristiwa dalam kesenjangan ini, prediksi stasiun tunggal mungkin menjadi *satu-satunya* estimasi yang tersedia dalam jendela waktu emas. Routing yang mempertimbangkan jarak memastikan bahwa prediksi jauh ini menggunakan jendela PTW yang optimal panjang.

**Informasi teknik spektral.** InaTEWS saat ini menampilkan estimasi intensitas (MMI). Prediksi Sa(T) akan memungkinkan tautan langsung ke model kerentanan bangunan yang dikembangkan di bawah SNI 1726:2019 (kode desain seismik Indonesia), mendukung estimasi kerusakan otomatis dan prioritas respons.

### 4.5 Keterbatasan dan Pekerjaan Mendatang

Beberapa keterbatasan layak dibahas:

**Siklus Hidup Operasional.** Sementara batas-batas teoritis dari garis waktu peringatan awal terutama diatur oleh pemetaan perjalanan gelombang seismik (Minson et al., 2021), penyebaran dan pemeliharaan operasi stabil saluran EEWS pembelajaran mesin di lingkungan produksi berkelanjutan memperkenalkan tantangan manajemen data dan concept drift yang substansial (Ahn et al., 2023b). Penyebaran saluran yang diusulkan di masa depan harus mengintegrasikan pemantauan telemetri real-time yang ketat.

**Ketidakseimbangan kelas.** Kelas *Merusak* (PGA ≥ 62.0 Gal) terdiri dari hanya 101 jejak (0.3% dari dataset), mencerminkan kelangkaan alami pergerakan tanah kuat near-field dalam jaringan akselerograf BMKG: hanya 1.2% jalur sumber-stasiun jatuh dalam 50 km, dan peristiwa gerakan kuat (M ≥ 5.5) pada jarak sedemikian menyumbang kurang dari 0.05% dari semua perekaman. Ini bukan kekurangan dataset melainkan representasi setia dari konfigurasi seismisitas-cakupan kawasan Java-Sumatra. Sementara kami mengatasi ini melalui pelatihan berbobot kelas dan oversampling SMOTE dalam pelatihan Tahap 1 (Tabel S1), kerangka kerja IDA secara inheren mengakomodasi ketidakseimbangan ini menurut desain: routing berbasis intensitas tidak tergantung pada kepadatan sampel near-field, karena kelas intensitas ditentukan oleh PGA yang diamati di sensor terlepas dari jarak. Iterasi masa depan harus menggabungkan catatan near-field dari jaringan lokal padat atau catatan seismik sintetis untuk lebih meningkatkan Damaging recall.

**Spesifisitas regional.** Semua model dilatih pada data zona subduksi Sunda. Transferabilitas ke rezim tektonik yang berbeda (misalnya batas Lempeng Laut Filipina, patahan transform) memerlukan adaptasi domain. Paradigma Dikotomi Fitur itu sendiri, bagaimanapun, berbasis fisika dan harus digeneralisasi di seluruh pengaturan tektonik.

**Efek tapak.** Ketiadaan parameter karakterisasi tapak (V_S30, Z_2.5) adalah faktor dominan yang membatasi prediksi Sa periode pendek (T = 0.3–0.6 s, R² ≈ 0.54), seperti ditunjukkan oleh celah R² 23% antara periode pendek dan panjang (Bagian 3.5). Mengikuti konvensi NGA-West2 (Ancheta et al., 2014), kami mengklasifikasikan nilai V_S30 ke dalam dua kategori: *measured (terukur)* (survei lapangan oleh BMKG menggunakan HVSR, MASW, atau metode borehole) dan *inferred (inferensi)* (estimasi berbasis proxy dari model lereng topografi global Allen dan Wald (2009)). Dari 854 stasiun BMKG dalam inventaris jaringan kami, 338 (39.6%) memiliki V_S30 terukur mencakup kelas tapak NEHRP A hingga E (71.9–1949.3 m/s; Kelas D dominan pada 51.2%), sementara sisanya 516 (60.4%) mengandalkan nilai inferensi yang ditandai dengan lima penetapan proxy diskret (300, 320, 230, 600, dan 430 m/s) dengan ketidakpastian yang jauh lebih tinggi (σ_ln(V_S30) ≈ 0.36–0.44 untuk inferensi vs. ≈ 0.05–0.15 untuk measured; Seyhan dan Stewart, 2014).

Eksperimen integrasi V_S30 sistematis mengkonfirmasi dampak klasifikasi ini dalam dua fase. Dalam Fase 1, pelatihan ulang semua 515 model dengan 44 fitur (42 IM gelombang P + V_S30 + ln V_S30) menggunakan predominan V_S30 inferensi (203 stasiun terukur, ~27% dari data pelatihan) menghasilkan mean ΔR² = +0.030 (+5.1%) di semua 103 periode spektral tanpa terkecuali. Dalam Fase 2, dataset terukur diperluas ke 338 stasiun BMKG, menggantikan nilai inferensi dengan V_S30 terukur di 265 stasiun (117,965 jejak, 61.3% dari data pelatihan). Ini menghasilkan peningkatan yang jauh lebih besar: mean ΔR² = +0.143 (+24.6%) dibandingkan baseline, dengan 100% periode (103/103) meningkat. Kinerja periode pendek (T = 0.05–0.6 s) meningkat sebesar ΔR² = +0.194, sementara kinerja periode panjang (T ≥ 3.0 s) meningkat sebesar ΔR² = +0.171. R² prediksi PGA meningkat dari 0.516 menjadi 0.696, dan Sa(1.0 s) dari 0.527 menjadi 0.705. Deviasi standar V_S30 yang hampir berlipat ganda (130.3 → 204.9 m/s) memungkinkan XGBoost untuk lebih baik menyelesaikan efek amplifikasi tapak yang sebelumnya runtuh ke dalam bin proxy diskret. Hasil-hasil ini dengan jelas mengkonfirmasi bahwa transisi dari V_S30 inferensi ke terukur adalah jalur peningkatan paling berdampak tunggal. Memperluas pengukuran lapangan ke 516 stasiun inferensi yang tersisa, dikombinasikan dengan periode dominan tapak T_dom (Seyhan dan Stewart, 2014), akan menghasilkan keuntungan lebih lanjut, khususnya pada periode pendek di mana efek amplifikasi tapak paling kuat.

**Perbandingan deep learning.** Studi ini menggunakan pohon yang diperkuat gradien (XGBoost) secara eksklusif. Arsitektur jaringan saraf konvolusi yang beroperasi langsung pada gelombang mentah (Mousavi et al., 2020) dapat berpotensi mengekstrak fitur tambahan yang tidak ditangkap oleh set fitur 42-fitur buatan tangan kami, khususnya karakteristik gelombang transien yang relevan dengan estimasi jarak.

---

## 5. Kesimpulan

Kami mempresentasikan kerangka kerja Jendela Waktu Gelombang P Adaptif Dua Tahap yang Sadar-Saturasi untuk peringatan dini gempabumi stasiun tunggal dengan temuan kunci berikut:

1. **Dikotomi Fitur.** Pemisahan sistematis parameter gelombang P saturasi (τ_c, 51% pentingnya dalam Tahap 1) untuk klasifikasi intensitas dari parameter non-saturasi (CVAD, 89.5% pentingnya dalam Tahap 2) untuk regresi Sa(T) muncul secara alami dari fisika sumber dan memberikan paradigma desain baru untuk EEWS.

2. **Tradeoff Keselamatan Operasional.** IDA-PTW mencapai R² komposit = 0.73 sambil mempertahankan 72.3% Damaging recall dan >99% kepatuhan waktu emas. Distribusi routing konservatif (99.9% jejak pada PTW = 3–4 s) mewakili tradeoff yang disengaja memprioritaskan keselamatan operasional daripada keuntungan R² marginal dari jendela tetap yang lebih panjang dan tidak dibatasi.

3. **Prediksi spektral komprehensif.** Saluran memprediksi Sa(T) pada 103 periode (T = 0.1–5.0 s) dengan R² = 0.73 menggunakan hanya sinyal gelombang P 3-detik awal dari stasiun tunggal di kawasan studi Java-Sumatra.

4. **Ketahanan yang memperbaiki diri.** Sistem tahan terhadap kesalahan estimasi jarak: mekanisme seleksi PTW memperbaiki diri karena kesalahan routing menuju jendela lebih panjang dikompensasi oleh hubungan PTW–akurasi monoton.

5. **Diagnostik evaluasi komprehensif.** Mengikuti kerangka kerja Dai et al. (2024), model menunjukkan σ_total ≈ 0.752 log₁₀ (sebanding dengan GMPE NGA-West2 meskipun batasan stasiun tunggal), bias rata-rata hampir nol di semua 103 periode, 83.3% prediksi dalam ±1.0 log₁₀, dan distribusi residual hampir normal memvalidasi asumsi log-normal. Dua keterbatasan yang jelas diidentifikasi — saturasi magnitudo (M ≥ 6.5) dan bias near-field (D < 30 km) — memberikan arah konkret untuk peningkatan masa depan melalui koreksi yang menyadari saturasi dan pengayaan data Vs30.

6. **Validasi pengayaan V_S30.** Eksperimen sistematis mengintegrasikan data V_S30 terukur dari 338 stasiun BMKG (diperluas dari 203 estimasi berbasis proxy) ke dalam model 44-fitur (42 IM gelombang P + V_S30 + ln V_S30) mencapai ΔR² = +0.143 (+24.6%) dibandingkan baseline di semua 103 periode tanpa terkecuali, mengkonfirmasi bahwa karakterisasi tapak adalah jalah paling berdampak tunggal untuk mengurangi variabilitas intra-peristiwa (φ), yang mendominasi 70% ketidakpastian prediksi total.

Saluran 517-model (1 pengklasifikasi, 1 estimator jarak, 515 regressor Sa(T)), divalidasi dengan validasi silang ketat dikelompokkan berdasarkan peristiwa, tersedia secara terbuka di https://github.com/hanif7108/adaptive-ptw-eews untuk integrasi ke dalam infrastruktur EEWS operasional.

---

## Pernyataan Ketersediaan Data

Data gelombang seismik mentah yang digunakan dalam penelitian ini disediakan oleh Badan Meteorologi, Klimatologi, dan Geofisika Republik Indonesia (BMKG) dan tersedia atas permintaan yang masuk akal dari penyedia (https://dataexchange.bmkg.go.id/). Kode sumber untuk saluran dua tahap lengkap — termasuk ekstraksi fitur, naskah pelatihan model, naskah evaluasi, dan gambar publikasi — tersedia secara terbuka di https://github.com/hanif7108/adaptive-ptw-eews. Dataset Sunda v2 (Geomean) yang diproses, diformat dalam wadah HDF5 yang kompatibel dengan SeisBench, dan artefak model terlatih (517 model XGBoost) tersedia atas permintaan yang masuk akal kepada penulis yang sesuai karena ukuran filenya yang besar (~4 GB untuk gelombang, ~500 MB untuk fitur pra-ekstrak).

## Pernyataan Etika

Penelitian ini menggunakan data gelombang seismik yang diarsipkan secara terbuka dari jaringan operasional BMKG. Tidak ada subjek manusia, hewan, atau data terklasifikasi yang terlibat. Persetujuan etika tidak diperlukan.

## Kontribusi Penulis

**HAN:** Konseptualisasi, Metodologi, Perangkat Lunak, Analisis Formal, Investigasi, Kurasi Data, Penulisan — Draf Asli, Visualisasi. **DD:** Pengawasan, Validasi, Penulisan — Peninjauan & Pengeditan. **AHS:** Pengawasan, Sumber Daya, Penulisan — Peninjauan & Pengeditan. **SP:** Kurasi Data, Validasi, Penulisan — Peninjauan & Pengeditan. Semua penulis meninjau dan menyetujui versi yang diserahkan.

## Pendanaan

Penelitian ini merupakan bagian dari program doktor di Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA), Universitas Indonesia, didanai oleh Badan Meteorologi, Klimatologi, dan Geofisika Republik Indonesia (BMKG). Para penulis berterima kasih atas sumber daya komputasi yang disediakan oleh Departemen Fisika, Universitas Indonesia.

## Ucapan Terima Kasih

Kami berterima kasih kepada Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) atas penyediaan data gelombang akselerometri dari jaringan seismik Indonesia. Kami mengakui tim pengembangan SeisBench untuk kerangka kerja benchmark standar.

## Konflik Kepentingan

Para penulis menyatakan bahwa penelitian dilakukan tanpa adanya hubungan komersial atau keuangan apa pun yang dapat dianggap sebagai potensi konflik kepentingan.

## Materi Pelengkap

Materi Pelengkap untuk artikel ini dapat ditemukan secara online. Ini mencakup:

- **Tabel S1:** Hyperparameter XGBoost lengkap untuk ketiga tahap saluran (pengklasifikasi Intensity Gate, estimator jarak, dan 515 regressor Sa(T)).
- **Tabel S2:** Tabel pencarian PTW 6 x 6 Intensitas-Jarak dengan parameter batasan waktu emas.

---

## Referensi

---

Abrahamson, N. A., Silva, W. J., and Kamai, R. (2014). Summary of the ASK14 ground motion relation for active crustal regions. *Earthquake Spectra*, 30(3), 1025–1055.

Ahn, J., Cho, S., Hwang, E., Baek, W. (2023). Assessing network-based earthquake early warning systems in low-seismicity areas. *Frontiers in Earth Science*, 11. doi: 10.3389/feart.2023.1268064

Ahn, J., Park, E., Kim, B., Hwang, E., Hong, S. (2023). Stable operation process of earthquake early warning system based on machine learning: trial test and management perspective. *Frontiers in Earth Science*, 11. doi: 10.3389/feart.2023.1157742

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631.

Allen, R. M., and Melgar, D. (2019). Earthquake early warning: advances, scientific challenges, and societal needs. *Annual Review of Earth and Planetary Sciences*, 47, 361–388.

Anderson, J. G., and Hough, S. E. (1984). A model for the shape of the Fourier amplitude spectrum of acceleration at high frequencies. *Bulletin of the Seismological Society of America*, 74(5), 1969–1993.

Allen, T. I., and Wald, D. J. (2009). On the use of high-resolution topographic data as a proxy for seismic site conditions (VS30). *Bulletin of the Seismological Society of America*, 99(2A), 935–943.

Bommer, J. J., and Alarcon, J. E. (2006). The prediction and use of peak ground velocity. *Journal of Earthquake Engineering*, 10(1), 1–31.

Boore, D. M. (2003). Simulation of ground motion using the stochastic method. *Pure and Applied Geophysics*, 160(3–4), 635–676. doi: 10.1007/PL00012553

Borcherdt, R. D. (1994). Estimates of site-dependent response spectra for design (methodology and justification). *Earthquake Spectra*, 10(4), 617–653. doi: 10.1193/1.1585791

Bracale, M., Colombelli, S., Elia, L., Karakostas, V., Zollo, A. (2021). Design, Implementation and Testing of a Network-Based Earthquake Early Warning System in Greece. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.667160

Brown, H. M., Allen, R. M., and Grasso, V. F. (2009). Testing ElarmS in Japan. *Seismological Research Letters*, 80(5), 727–739.

Böse, M., Hauksson, E., Solanki, K., Kanamori, H., and Heaton, T. H. (2009). Real-time testing of the on-site warning algorithm in southern California. *Bulletin of the Seismological Society of America*, 99(2A), 497–517.

Carranza, M., Mattesini, M., Buforn, E., Zollo, A., Torrego, I. (2021). Four Years of Earthquake Early Warning in Southern Iberia: 2016–2019. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.696191

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Colombelli, S., Caruso, A., Zollo, A., Festa, G., and Kanamori, H. (2015). A P wave-based, on-site method for earthquake early warning. *Geophysical Research Letters*, 42(5), 1390–1398.

Cremen, G., and Galasso, C. (2020). Earthquake early warning: Recent advances and perspectives. *Earth-Science Reviews*, 205, 103184.

Dhakal, Y., Kunugi, T. (2021). An Evaluation of Strong-Motion Parameters at the S-net Ocean-Bottom Seismograph Sites Near the Kanto Basin for Earthquake Early Warning. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.699439

Festa, G., Zollo, A., and Lancieri, M. (2008). Earthquake magnitude estimation from early radiated energy. *Geophysical Research Letters*, 35(22), L22307.

Finazzi, F. (2020). The Earthquake Network Project: A Platform for Earthquake Early Warning, Rapid Impact Assessment, and Search and Rescue. *Frontiers in Earth Science*, 8. doi: 10.3389/feart.2020.00243

Dai, H., Zhou, Y., Liu, H., Li, S., Wei, Y., and Song, J. (2024). XGBoost-based prediction of on-site acceleration response spectra with multi-feature inputs from P-wave arrivals. *Soil Dynamics and Earthquake Engineering*, 178, 108490. doi: 10.1016/j.soildyn.2024.108490

Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer.

Hoshiba, M. (2021). Real-Time Prediction of Impending Ground Shaking: Review of Wavefield-Based (Ground-Motion-Based) Method for Earthquake Early Warning. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.722784

Hsu, T.-Y., and Huang, C.-W. (2021). Onsite Early Prediction of PGA Using CNN With Multi-Scale and Multi-Domain P-Waves as Input. *Frontiers in Earth Science*, 9, 626908. doi: 10.3389/feart.2021.626908

Hsu, T., Pratomo, A. (2022). Early Peak Ground Acceleration Prediction for On-Site Earthquake Early Warning Using LSTM Neural Network. *Frontiers in Earth Science*, 10. doi: 10.3389/feart.2022.911947

Iaccarino, A. G., Picozzi, M., Bindi, D., and Spallarossa, D. (2020). Onsite Earthquake Early Warning: Predictive Models for Acceleration Response Spectra Considering Site Effects. *Bulletin of the Seismological Society of America*, 110(3), 1289–1304. doi: 10.1785/0120190272

Iaccarino, A., Gueguen, P., Picozzi, M., Ghimire, S. (2021). Earthquake Early Warning System for Structural Drift Prediction Using Machine Learning and Linear Regressors. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.666444

Iaccarino, A. G., Cristofaro, A., Picozzi, M., Spallarossa, D., and Scafidi, D. (2024). Real-time prediction of distance and PGA from P-wave features using Gradient Boosting Regressor for on-site earthquake early warning applications. *Geophysical Journal International*, 236(1), 675–687. doi: 10.1093/gji/ggad443
