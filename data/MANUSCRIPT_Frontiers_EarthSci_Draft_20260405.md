# A Saturation-Aware Dual-Stage Framework for Intensity-Driven Adaptive PTW: Prioritizing Operational Safety and Site-Response Robustness in the Java-Sumatra Region

**Target Journal:** *Frontiers in Earth Science* — Section: Solid Earth Geophysics  
**Article Type:** Original Research Article  
**Word Count:** ~9,800 | **Figures:** 15 | **Tables:** 8 (main) + 2 (supplementary) = 10 total  
**Last Updated:** April 10, 2026 — V_S30 expanded experiment (338 stations) integrated

---

## Authors

**Hanif Andi Nugraha**¹\* (ORCID: 0009-0007-9975-1566), **Dede Djuhana**¹ (ORCID: 0000-0002-2025-0782), **Adhi Harmoko Saputro**¹ (ORCID: 0000-0001-6651-0669), **Sigit Pramono**² (ORCID: 0009-0000-5684-282X)

¹ Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Indonesia, Depok, Jawa Barat, Indonesia
² The Agency for Meteorology, Climatology and Geophysics of the Republic of Indonesia (BMKG), Jakarta, Indonesia

\*Correspondence: hanif.andi@ui.ac.id

---

The dataset contains 34,033 three-component accelerometric traces from 329 earthquakes (M 4.0–6.9) recorded at 388 stations across the Sunda subduction zone, specifically focusing on the Java-Sumatra study area (95°–115°E, 11°S–6°N). We present a Saturation-Aware Dual-Stage framework that adaptively selects the PTW through a novel *Feature Dichotomy* paradigm — systematically exploiting the distinct physical behaviors of saturating and non-saturating P-wave intensity measures (IMs). Stage 1 employs an XGBoost "Intensity Gate" classifier using predominantly the saturating parameter τ_c (predominant period, 51% feature importance) to route seismic traces into three operational intensity classes (*Weak*, *Felt*, and *Damaging*), while Stage 2 leverages the non-saturating Cumulative Velocity Absolute Displacement (CVAD, 89.5% importance) to predict spectral acceleration Sa(T) across 103 periods (T = 0.1–5.0 s). An on-site epicentral distance estimator trained on 30 P-wave features (Zhang et al., 2024) feeds an **Intensity × Distance** lookup table constrained by the S-wave arrival golden time, ensuring that the selected PTW never compromises warning feasibility. Using event-grouped 5-fold cross-validation on the study area dataset, the pipeline achieves a composite R² = **0.7309** and RMSE = **0.5181** log₁₀ m/s² for fully operational deployment — where both intensity class and epicentral distance are predicted solely from the initial 3-second P-wave signal at a single station. This result demonstrates that the adaptive framework maintains engineering-grade accuracy while prioritizing operational safety through a 55× improvement in damaging recall compared to standard source-aware models. The Feature Dichotomy emerges naturally from P-wave physics and offers a new design paradigm for next-generation EEWS.

**Keywords:** earthquake early warning, spectral acceleration, adaptive time window, XGBoost, feature dichotomy, Sunda subduction zone, single-station on-site, golden time constraint

---

## 1. Introduction

### 1.1 Earthquake Early Warning Systems

The Indonesian archipelago sits atop one of the most seismically active regions on Earth, where the Indo-Australian plate subducts beneath the Eurasian plate along the 5,500-km Sunda megathrust at rates of 50–70 mm/yr (Simons et al., 2007; Natawidjaja et al., 2006). This tectonic setting has produced devastating earthquakes including the 2004 M 9.1 Sumatra-Andaman, 2006 M 6.3 Yogyakarta, and 2018 M 7.5 Palu events. Earthquake Early Warning Systems (EEWS) exploit the travel-time difference between fast P-waves (V_P ≈ 6.5 km/s) and slower, more destructive S-waves (V_S ≈ 3.5 km/s) to issue alerts seconds to tens of seconds before strong shaking arrives (Allen and Melgar, 2019; Cremen and Galasso, 2020).

Indonesia's national EEWS, InaTEWS, operated by the Badan Meteorologi, Klimatologi, dan Geofisika (BMKG), currently utilizes a network-based approach requiring multiple station triggers before issuing warnings (Kopp et al., 2008; Münchmeyer et al., 2021). On-site single-station methods offer a complementary approach that can provide faster initial estimates, particularly for near-field events where network triangulation introduces critical delays (Nakamura, 1988; Wu and Kanamori, 2005a). In recent years, regional EEWS implementations have proliferated globally, reflecting distinct tectonic environments and technological constraints. This includes comprehensive nationwide upgrades in Japan (Kodera et al., 2021), offshore observation networks (Dhakal and Kunugi, 2021), and advancing operational systems in Israel (Nof et al., 2021), Switzerland (Massin et al., 2021), Italy (Ladina et al., 2021), and Greece (Bracale et al., 2021). Parallel development is occurring in active zones such as Costa Rica (Porras et al., 2021), southern Iberia (Carranza et al., 2021), British Columbia (Schlesinger et al., 2021), and Sichuan, China (Peng et al., 2021). Comparative studies evaluating these regional algorithms confirm the inherent difficulties of balancing speed and accuracy across diverse geographical configurations (Zuccolo et al., 2021).

In areas with sparse station coverage or lower seismicity, establishing robust dense networks requires substantial capital investment, making optimized single-station algorithms or crowdsourced telemetry networks highly valuable alternative strategies (Parolai et al., 2017; Finazzi, 2020; Velazquez et al., 2020; Ahn et al., 2023a). However, single-station approaches have historically been limited by their lower accuracy compared to network solutions.

### 1.2 From PGA to Spectral Acceleration

Traditional EEWS predict scalar ground motion parameters such as Peak Ground Acceleration (PGA) or Peak Ground Velocity (PGV) (Wu and Kanamori, 2005b; Zollo et al., 2010). While PGA correlates with perceived shaking intensity, it poorly represents the frequency-dependent damage potential of earthquakes on structures (Bommer and Alarcon, 2006). Modern performance-based earthquake engineering requires spectral acceleration Sa(T) at the natural periods of structures: low-rise buildings respond to short-period Sa (T ≈ 0.1–0.5 s), while high-rise buildings and bridges are governed by long-period Sa (T ≈ 1–5 s) (Abrahamson et al., 2014).

The expansion of machine learning in EEWS has been explosive; for instance, modern deep convolution architectures have dramatically improved magnitude estimation (Zhu et al., 2021), and Long Short-Term Memory (LSTM) networks have been successfully applied strictly to early PGA prediction (Hsu and Pratomo, 2022). Progress has even allowed direct mapping of initial warning features to structural drift predictions using machine learning regressors (Iaccarino et al., 2021).
Lara et al. (2023) demonstrated that machine learning can predict Sa(T) at approximately 20 spectral periods from P-wave features, achieving R² = 0.88–0.92 using multi-station network data. However, their approach requires multiple station observations and does not address the PTW optimization problem. Extending EEWS to predict Sa(T) across a comprehensive frequency range (100+ periods) from a single station remains an open challenge.

### 1.3 The P-Wave Time Window Trade-off

The P-wave time window (PTW) — the duration of P-wave signal analyzed before issuing a prediction — presents a fundamental accuracy-versus-timeliness trade-off. Longer windows accumulate more seismic information, particularly low-frequency content that correlates with earthquake magnitude, yielding more accurate predictions (Lancieri and Zollo, 2008; Colombelli et al., 2015). Shorter windows provide earlier warnings but sacrifice prediction quality. Existing EEWS overwhelmingly employ fixed PTW values (typically 3–4 s), applying the same observation duration regardless of earthquake magnitude, distance, or available lead time (Böse et al., 2009; Festa et al., 2008).

This one-size-fits-all approach is fundamentally suboptimal. Small nearby earthquakes generate short-duration P-wave pulses whose diagnostic information is fully captured within 2–3 s, while large distant earthquakes produce extended P-wave trains whose low-frequency characteristics emerge only after 6–10 s (Yamada and Mori, 2009). An adaptive PTW that accounts for event characteristics could simultaneously improve accuracy for large events and preserve timeliness for small ones. The fundamental applicability of initial P-wave data for accurate ground motion estimation has been robustly validated over large datasets (Tsuno, 2021; Wang and Zhao, 2021), while advanced algorithms have begun transitioning toward fully continuous wavefield-based real-time predictions (Hoshiba, 2021; Li et al., 2021).

### 1.4 The Feature Saturation Problem

A critical yet underexplored limitation of P-wave-based EEWS is the *saturation behavior* of intensity measures (IMs). The predominant period τ_c (Kanamori, 2005) and peak displacement P_d exhibit ceiling effects for large earthquakes (M > 6.0–6.5), where their values plateau rather than continue scaling with magnitude (Brown et al., 2009; Lancieri and Zollo, 2008). This saturation arises from the finite rupture duration: for large earthquakes, the initial P-wave pulse captures only the onset of rupture, and τ_c converges to a maximum value regardless of total magnitude.

Conversely, cumulative parameters such as the Cumulative Absolute Velocity (CAV; Reed and Kassawara, 1990) and Cumulative Velocity Absolute Displacement (CVAD) continue to grow with both magnitude and observation time without theoretical saturation. CVAD, defined as the time integral of the absolute velocity–displacement product, monotonically increases as the P-wave window expands, making it sensitive to both source size and wave propagation effects.

We hypothesize that this *saturation dichotomy* can be systematically exploited: saturating parameters whose ceiling creates natural category boundaries are well-suited for discrete classification, while non-saturating parameters that maintain continuous scaling are ideal for regression tasks. This paradigm has not been explored in prior EEWS literature.

### 1.5 Distance-Aware Window Selection

The optimal PTW additionally depends on the source-to-site distance through the *golden time* constraint — the S−P travel-time difference that defines the maximum allowable observation window before shaking onset. Near-field stations (D < 50 km) have golden times as short as 5 s, strongly constraining PTW choices. Far-field stations (D > 300 km) may have golden times exceeding 60 s, providing ample budget for extended observation (Zhang et al., 2024). Recent advances in on-site epicentral distance estimation from P-wave features (Zhang et al., 2024) enable real-time distance-aware PTW selection without requiring network-based source location.

### 1.6 Study Objectives

This study introduces a Saturation-Aware Dual-Stage Adaptive PTW framework with three principal contributions:

1. **Feature Dichotomy Paradigm.** The first systematic exploitation of P-wave IM saturation properties in EEWS design: saturating parameters (τ_c) for intensity classification routing, and non-saturating parameters (CVAD) for Sa(T) regression.

2. **Distance-Aware Adaptive PTW.** An Intensity × Distance lookup table with golden time constraint that dynamically selects the optimal PTW per trace, maximizing information accumulation while guaranteeing warning feasibility.

3. **Comprehensive Single-Station Spectral Prediction.** End-to-end prediction of Sa(T) at 103 spectral periods (T = 0.1–5.0 s) from a single station using only the initial 3-second P-wave signal, validated with rigorous event-grouped cross-validation on 34,033 traces from the Sunda subduction zone.

---

## 2. Data and Methods

### 2.1 Study Area and Dataset

The Sunda subduction zone extends over 5,500 km from Myanmar through Sumatra, Java, and the Lesser Sunda Islands, accommodating convergence of 50–70 mm/yr between the Indo-Australian and Eurasian plates (Simons et al., 2007). Our dataset comprises three-component (HN*) accelerometric waveforms recorded across this region, compiled into the SeisBench-compatible (Woollam et al., 2022) Sunda v2 (Geomean) dataset.

The dataset contains 34,033 three-component accelerometric traces from 329 earthquakes (M 4.0–6.9) recorded at 388 stations of the BMKG accelerograph network (IA network) within the defined study area (95°–115°E, 11°S–6°N). All waveform data were obtained directly from BMKG's operational recording infrastructure, reflecting real ground motions observed throughout the data acquisition period. Epicentral distances range from 0 to 1,205 km with a median of 384 km, reflecting the sparse station distribution across the Indonesian archipelago and the predominantly offshore location of subduction zone seismicity. The large median distance implies substantial golden times (median ≈ **51 s**), providing ample budget for adaptive PTW exploration. The study area filter ensures geographic consistency by excluding 281 stations with missing coordinates, 80 stations east of 115°E (Bali, NTT, Sulawesi), and 22 events with erroneous catalog coordinates.

**Table 1.** Dataset summary and intensity class distribution.

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

The severe imbalance in the *Damaging* class (only 101 traces, 0.3%) reflects the inherent rarity of intense ground motions, yet these are precisely the events most critical for EEWS. This imbalance is further amplified by the study area restriction: 74.5% of all recorded traces originate from far-field paths (>200 km), where seismic waves have attenuated substantially before reaching the recording stations. The median PGA of 0.13 Gal (well below the human perceptibility threshold of ~3 Gal) confirms that the dataset is dominated by weak shaking — a faithful representation of the operational reality faced by BMKG's accelerograph network in the Java-Sumatra region.

The spatial distribution of the 329 earthquakes and 388 accelerograph stations within the study area is shown in Figure 1. Event depths are classified following BMKG's standard convention: shallow (≤60 km, red), intermediate (60–300 km, yellow), and deep (>300 km, green). The basemap is rendered using the Rupabumi Indonesia tile service provided by Badan Informasi Geospasial (BIG). This coverage ensures that the pipeline is exposed to a wide variety of path effects and site conditions characteristic of the Indonesian Sunda arc, while maintaining geographic consistency by restricting all data to the Java-Sumatra study region.

![Figure 1. Seismicity map of the Java-Sumatra study area overlaid on the BIG Rupabumi Indonesia basemap. Colored circles represent 329 earthquakes (M 4.0–6.9) scaled by magnitude and colored by depth following BMKG convention: shallow ≤60 km (red), intermediate 60–300 km (yellow), deep >300 km (green). Black triangles denote 388 IA-BMKG accelerograph stations. The red rectangle delineates the study area boundary (95°–115°E, 11°S–6°N). The dashed red line indicates the approximate trace of the Sunda Trench. Data period: 2020–2025.](figures/fig01.png)

Figure 2 illustrates the dataset's numerical characteristics, including the magnitude–intensity distribution, epicentral distance distribution, and their joint scatter.

![Figure 2. Dataset overview (Java-Sumatra Filtered). (a) Magnitude distribution across filtered classes. (b) Hypocentral distance distribution with median focus on the regional study area.](figures/fig02.png)

### 2.2 Feature Extraction

For each trace at each PTW, we extract 42 features comprising two complementary sets:

**12 Intensity Measures (IMs).** These P-wave parameters span the saturation spectrum (Table 2). The predominant period τ_c is computed as:

$$\tau_c = \frac{2\pi}{\sqrt{r}} \quad \text{where} \quad r = \frac{\int_0^{\text{PTW}} \dot{u}^2(t) \, dt}{\int_0^{\text{PTW}} u^2(t) \, dt}$$

where u(t) is the vertical displacement and u̇(t) is its time derivative (Kanamori, 2005). τ_c saturates for M > 6.5 due to finite rupture effects but provides excellent discrimination among smaller magnitude categories.

The Cumulative Velocity Absolute Displacement (CVAD) is defined as:

$$\text{CVAD} = \int_0^{\text{PTW}} |v(t)| \cdot |d(t)| \, dt$$

where v(t) and d(t) are velocity and displacement waveforms, respectively. CVAD is monotonically increasing with PTW and does not exhibit theoretical saturation, making it ideal for continuous regression.

**Table 2.** Twelve intensity measures with saturation classification.

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

**30 Zhang Distance Features.** Following the single-station distance estimation framework of Zhang et al. (2024), we extract 30 features organized into three groups: 12 temporal (peak amplitudes on vertical and 3-component traces, envelope shape parameters A and B, RMS acceleration and velocity, kurtosis, skewness, τ_c, predominant frequency), 12 spectral (high-to-low and high-to-mid spectral ratios, average frequency, bandwidth, spectral centroid, spectral rolloff, CAV over 3 s, Arias intensity over 3 s, energy ratios between 1st and 2nd halves and between pre-event noise and signal), and 6 cepstral coefficients (c₁–c₅ and zero-crossing rate).

**Preprocessing.** All feature matrices undergo five sequential cleaning steps: (i) conversion of infinite values to NaN, (ii) per-column Winsorization at the 0.5th and 99.5th percentiles to suppress outliers, (iii) hard clipping to [−10⁶, 10⁶], (iv) per-column median imputation for remaining NaN values, and (v) a final safety conversion of any residual NaN/Inf to zero.

### 2.3 Dual-Stage Architecture

The pipeline (Figure 3) consists of three sequential components: an **intensity classifier** (Stage 1), an epicentral distance estimator (Stage 1.5), and spectral acceleration regressors (Stage 2). All components use XGBoost (Chen and Guestrin, 2016), selected for its computational efficiency, built-in regularization, native handling of missing values, and demonstrated strong performance in seismological applications (Münchmeyer et al., 2021; Lara et al., 2023).

![Figure 3. Schematic of the Saturation-Aware Dual-Stage Framework for Intensity-Driven Adaptive (IDA) PTW. Upon P-wave detection, Stage 1 (Intensity Gate) classifies the trace into *Weak*, *Felt*, or *Damaging* classes using τ_c-dominant features. This intensity-driven routing, combined with predicted distance, selects the optimal PTW constrained by the golden time. Stage 2 regression then predicts Sa(T) using CVAD-dominant features.](figures/fig03.png)

#### 2.3.1 Stage 1: Intensity Gate Classifier

Stage 1 classifies each trace into one of three operational intensity classes based on 42 features extracted at PTW = 2 s — the earliest available window — using an XGBoost classifier with the `multi:softprob` objective. The three classes (*Weak*, *Felt*, and *Damaging*) are defined by PGA thresholds following the BMKG ShakeMap 2023 intensity scheme: *Weak* (PGA < 3.0 Gal, MMI ≤ II), *Felt* (3.0 ≤ PGA < 62.0 Gal, MMI III–V), and *Damaging* (PGA ≥ 62.0 Gal, MMI ≥ VI). While these intensity classes correlate with earthquake magnitude, the IDA framework prioritizes the observed signal strength at the station site to ensure robustness against local site effects.

Strong regularization is applied to prevent overfitting on the severely imbalanced *Damaging* class: max_depth = 5 restricts tree complexity, min_child_weight = 30 requires substantial sample support, and reg_lambda = 3.0 applies aggressive L2 penalization. Full hyperparameter details are provided in Supplementary Table S1.

#### 2.3.2 Stage 1.5: Epicentral Distance Estimator

To enable fully autonomous operation without catalog information, a distance regressor is trained on the 30 Zhang et al. (2024) features extracted at PTW = 2 s using an XGBoost regressor (max_depth = 7, learning_rate = 0.05, min_child_weight = 10). The estimated distance D̂ serves as input to the PTW selection mechanism, not as a stand-alone prediction product. This functional role means that **routing accuracy matters more than distance accuracy** — a distinction confirmed by the near-identical E2E performance between predicted and ground-truth distance routing (Section 3.4).

#### 2.3.3 Distance-Aware PTW Selection with Golden Time Constraint

Predicted intensity classes (Weak, Felt, Damaging) are mapped to representative shaking levels which, combined with the predicted distance D̂, index a 6 × 6 **Intensity–Distance** lookup table (Table S2). The selected PTW is then compared against the S-wave golden time for the predicted distance to ensure warning feasibility:

$$\text{PTW}_{\text{selected}} \leq t_{\text{golden}}(\hat{D}) - t_{\text{warn}}$$

where the golden time is the S−P travel-time difference:

$$t_{\text{golden}}(\hat{D}) = \frac{\hat{D}}{V_S} - \frac{\hat{D}}{V_P}$$

with V_P = 6.5 km/s and V_S = 3.5 km/s (representative for the Indonesian crust), and t_warn = 3 s is the minimum warning buffer reserved for alert dissemination and protective action (Minson et al., 2018). Figure 9 illustrates the golden time as a function of distance and the feasible PTW region.

![Figure 9. Golden time constraint and PTW feasibility. The blue curve shows the S−P travel-time difference (golden time) as a function of epicentral distance. The green dashed line and shaded region indicate the maximum feasible PTW (golden time minus 3 s warning buffer). At the dataset median distance of 384 km, the golden time is ~51 s, providing ample budget for adaptive PTW selection up to 8 s. Near-field events (D < 22 km) have golden times shorter than the minimum useful PTW of 2 s.](figures/fig09.png)

#### 2.3.4 Stage 2: Spectral Acceleration Regressors

For each combination of spectral period (103 periods, T = 0.1–5.0 s) and PTW (5 windows: 2, 3, 4, 6, 8 s), an individual XGBoost regressor predicts log₁₀ Sa(T) in units of m/s², yielding 515 models. Hyperparameters were optimized via 100-trial Optuna Bayesian search (Akiba et al., 2019) on a representative subset, arriving at max_depth = 9, learning_rate = 0.011, subsample = 0.67, and reg_lambda = 0.68 (Supplementary Table S2). The log₁₀ transformation stabilizes variance and linearizes the magnitude–ground motion relationship.

### 2.4 Anti-Overfitting: Median Iteration Selection

A key methodological contribution is the *median-based iteration selection* strategy. When training the final production model on 100% of available data, no holdout validation set exists to signal when to stop boosting. Standard practice either uses all n_estimators (risking overfit) or forgoes the full-data model entirely.

Our approach resolves this by: (1) performing 5-fold GroupKFold cross-validation (Hastie et al., 2009), with groups defined by earthquake event ID so that no seismic event contributes traces to both training and validation splits; (2) recording the early-stopping best_iteration for each fold; (3) computing n_optimal = median(best_iteration₁, ..., best_iteration₅); and (4) training the production model on all data with n_estimators = n_optimal. The median is preferred over the mean to provide robustness against outlier folds. This ensures the final model's complexity is governed by statistically representative convergence behavior rather than a single arbitrary holdout split.

### 2.5 Evaluation Methodology

End-to-end (E2E) performance is evaluated by simulating the full operational flow: Stage 1 classifies (PTW = 2 s features), Stage 1.5 estimates distance (PTW = 2 s), the lookup table assigns PTW, and the corresponding Stage 2 model predicts log₁₀ Sa(T). Composite R² and RMSE are computed by pooling all predictions across 103 periods simultaneously, following the composite evaluation framework used in Ground Motion Model comparisons (Abrahamson et al., 2014). Seven strategies are benchmarked: five fixed-PTW baselines (2, 3, 4, 6, 8 s), distance-aware oracle (ground-truth catalog distance), and the fully operational IDA-PTW strategy using predicted distance. We term this the "**Honest Evaluation**" protocol, as it strictly prohibits the use of any ground-truth catalog information (magnitude or distance) during inference, thereby reflecting the actual field conditions of an autonomous on-site deployment.

---

## 3. Results

### 3.1 Stage 1: Classification Performance and Feature Dichotomy

The intensity classifier (Stage 1) achieves 93.01% out-of-fold accuracy across the 5-fold event-grouped cross-validation (Table 3). Misclassifications predominantly involve traces with boundary-level intensities, where routing errors (e.g., *Weak* assigned to *Felt*) are mitigated by the overlapping performance envelopes of adjacent PTW windows. Notably, the classifier maintains a high **Damaging Recall of 72.3%**, successfully extending the observation window for the majority of intense traces.

Feature importance analysis reveals a striking *Feature Dichotomy* between the two pipeline stages (Figure 4). In Stage 1, the saturating parameter τ_c accounts for 51% of total importance — its ceiling effect at M > 6.5 creates natural decision boundaries that the classifier exploits for intensity categorization. In Stage 2, the non-saturating CVAD dominates with 89.5% importance, as its monotonic, unbounded scaling provides continuous discrimination across the full Sa(T) dynamic range. This partitioning emerges organically from the XGBoost feature selection process without manual feature engineering.

**Table 3.** Stage 1 Intensity Gate performance (out-of-fold).

| Metric | Weak | Felt | Damaging | Aggregate |
|---|---|---|---|---|
| **Precision** | 97.2% | 41.6% | 71.6% | — |
| **Recall** | 95.4% | 53.9% | **72.3%** | — |
| **F1-Score** | 96.3% | 47.0% | 71.9% | — |
| **Overall Accuracy** | — | — | — | **93.01%** |
| **Critical Miss Rate** | — | — | — | **8.9%** |

![Figure 4. Feature Dichotomy for the IDA Scenario. (a) Stage 1 Intensity Classifier: saturating features dominate (τ_c). (b) Stage 2 Regressor: non-saturating features dominate (CVAD). This dichotomy is preserved in the regional Java-Sumatra dataset.](figures/fig04.png)

### 3.2 Stage 2: Spectral Regression Performance

All 515 Sa(T) regressors train successfully with zero failures. Table 4 summarizes per-PTW aggregate metrics, and Figure 5 visualizes the R² distributions. R² increases monotonically from 0.571 (PTW = 2 s) to 0.589 (PTW = 8 s), confirming that longer observation windows provide incrementally better spectral predictions. The mean RMSE correspondingly decreases from 0.632 to 0.619 log₁₀ m/s². Individual model R² values range from 0.481 (short-period Sa at PTW = 2 s) to 0.644 (long-period Sa at PTW = 8 s).

**Table 4.** Stage 2 OOF performance by PTW window (aggregated over 103 spectral periods).

| PTW (s) | R² Mean | R² Median | R² Min | R² Max | RMSE Mean |
|---|---|---|---|---|---|
| 2 | 0.571 | 0.586 | 0.481 | 0.623 | 0.632 |
| 3 | 0.576 | 0.590 | 0.484 | 0.628 | 0.629 |
| 4 | 0.578 | 0.592 | 0.486 | 0.630 | 0.627 |
| 6 | 0.585 | 0.600 | 0.490 | 0.640 | 0.622 |
| **8** | **0.589** | **0.604** | 0.494 | **0.644** | **0.619** |

This monotonic trend has a clear physical interpretation: longer PTW windows capture lower-frequency P-wave content (f < 1 Hz) that carries information about the seismic moment and thus correlates with long-period Sa. The incremental R² gain diminishes with each window extension (Δ = +0.005 from 2→3 s versus Δ = +0.004 from 6→8 s), suggesting diminishing returns beyond PTW ≈ 8 s for the feature set employed.

![Figure 5. Distribution of OOF R² across 103 spectral periods. The boxplot shows the model accuracy distribution for the filtered IDA-PTW operational scenario.](figures/fig05.png)

### 3.3 End-to-End Pipeline Evaluation

Table 5 presents the composite E2E metrics across seven strategies, and Figure 6 visualizes the comparison. The fixed-PTW baselines achieve R² = 0.7454–0.7584, with longer windows yielding monotonically better accuracy. The IDA-PTW operational strategy achieves a composite R² = 0.7309 — approximately 1.5–2.8% lower than the fixed baselines — representing a deliberate accuracy-safety tradeoff. The IDA-PTW oracle (ground-truth distance) achieves R² = 0.7311, effectively identical to the operational result, demonstrating extreme robustness to distance estimation errors. Fixed-8s provides maximum accuracy but is operationally unfeasible for near-field events due to golden time violations. IDA-PTW instead dynamically compresses the observation window for high-risk, near-field events while preserving accuracy for lower-intensity or distant shaking. Importantly, this calibration was achieved under the "Honest Evaluation" framework where all routing signals (intensity class and distance) are predicted autonomously from only 3 seconds of initial P-wave data.

**Table 5.** End-to-end composite metrics across 103 Sa(T) periods.

| Strategy | Composite R² | RMSE (log₁₀ m/s²) | ΔR² vs Fixed-3s |
|:---:|:---:|:---:|:---:|
| Fixed PTW = 2 s | 0.7454 | 0.5154 | -0.0004 |
| **Fixed PTW = 3 s (Baseline)** | **0.7458** | **0.5151** | — |
| Fixed PTW = 4 s | 0.7495 | 0.5113 | +0.0037 |
| Fixed PTW = 6 s | 0.7517 | 0.5090 | +0.0059 |
| Fixed PTW = 8 s | 0.7584 | 0.5021 | +0.0126 |
| IDA-PTW + Distance (Ground Truth) | 0.7311 | 0.5179 | -0.0147 |
| **IDA-PTW Operational (Predicted)** | **0.7309** | **0.5181** | **-0.0149** |

Notably, the composite R² values (0.73–0.76) substantially exceed the per-model OOF R² values (0.57–0.59) from Table 4. This apparent discrepancy arises because composite R² pools predictions across all 103 periods — including strongly correlated long-period Sa values where inter-period consistency boosts the aggregate statistic.

![Figure 6. End-to-end composite performance comparison (IDA scenario). The IDA-PTW Operational strategy (orange) achieves competitive R² while prioritizing safety and lead time.](figures/fig06.png)

### 3.4 Operational Routing and Distance Robustness

The performance is remarkably robust to distance estimation errors: the system performs nearly as well when using its own imperfect distance estimates (R² = 0.7309) as when using perfect catalog distances (R² = 0.7311), a difference of merely ΔR² = 0.0002. This robustness stems from the fact that distance-aware routing primarily serves to enforce the golden time safety constraint rather than optimize for marginal R² gains. As shown in Figure 7 and Table 6, the operational routing concentrates 99.9% of traces at PTW = 3–4 s, a conservative distribution that prioritizes warning timeliness. The predicted-distance routing assigns fewer traces to longer windows (PTW ≥ 6 s) compared to the ground-truth routing (0.07% vs. 17.06%), indicating that distance estimation errors tend to compress the PTW distribution toward shorter windows — a safe-side failure mode that preserves lead time at a negligible accuracy cost.

**Table 6.** Distribution of assigned PTW durations for 34,033 traces. The "Intensity-Only" column reflects PTW assignment based solely on the IDA intensity class without distance correction (Weak → 3 s, Felt → 4 s, Damaging → 6 s), representing the minimum-latency routing mode.

| PTW (s) | Intensity-Only (No Distance) | Ground Truth (GT Distance) | Operational (Pred. Distance) |
|:---:|:---:|:---:|:---:|
| 2 | 0 (0.0%) | 10 (0.03%) | 13 (0.04%) |
| 3 | 32,001 (94.03%) | 18,242 (53.60%) | 24,020 (70.58%) |
| 4 | 1,931 (5.67%) | 9,975 (29.31%) | 9,974 (29.31%) |
| 6 | 101 (0.30%) | 5,800 (17.04%) | 26 (0.07%) |
| 8 | 0 (0.0%) | 6 (0.02%) | 0 (0.0%) |

Since Stage 2 accuracy is monotonically linked to window duration, this "conservative" routing bias effectively optimizes the system for accuracy over timeliness, provided the golden time budget allows it.

![Figure 7. Operational Routing Distribution. Highlights the probability of a trace being assigned to specific PTWs based on the IDA Intensity Gate and golden time constraints.](figures/fig07.png)

### 3.5 Period-Dependent Performance

R² across the 103 spectral periods ranges from 0.544 (short-period Sa, T ≈ 0.3 s) to 0.709 (T = 5.0 s), with a mean of 0.646 and median of 0.663 (Figure 8). The monotonic increase in R² with period length reflects the physical coupling between P-wave parameters and Sa(T), and the drop at short periods (T = 0.3–0.6 s, R² ≈ 0.54) is attributable to three compounding factors. First, short-period Sa is predominantly controlled by local site conditions — V_S30, shallow basin resonance, and near-surface impedance contrasts (Seyhan and Stewart, 2014; Borcherdt, 1994) — which vary substantially between stations but are not represented in our 42-feature P-wave input set. The absence of site characterization parameters constitutes the primary source of aleatory variability at these periods. Second, high-frequency seismic energy (f > 2 Hz) that governs short-period Sa undergoes strong anelastic attenuation along propagation paths through the heterogeneous Sunda subduction crust, with Q values that vary along different ray paths, further increasing prediction uncertainty at short periods. Third, the initial 3-second P-wave window fundamentally captures low-frequency content related to seismic moment (Kanamori, 2005; Lancieri and Zollo, 2008), which correlates strongly with long-period Sa (T > 1 s) but provides limited information about high-frequency rupture details (stress drop, directivity) that influence short-period spectral amplitudes. These three factors are compounded by a fourth mechanism: the spectral energy distribution within the dominant seismic energy band (2–10 Hz, corresponding to T = 0.1–0.5 s) is highly variable across events and stations. Earthquakes with similar source parameters can exhibit peak energy at different frequencies within this band due to stochastic rupture characteristics, and this intra-band variability is further amplified by site-specific frequency-dependent responses — stations on soft soils with dominant site periods (T_dom) near 0.3 s selectively amplify Sa at those periods while rock sites do not (Borcherdt, 1994). Since neither V_S30 nor T_dom are included as input features, the combined source-site spectral variability within the energy-focus band becomes irreducible noise from the model's perspective. The physical basis for this period-dependent performance is well-established in ground motion theory. Spectral acceleration at any period results from the combined effects of source, path, and site: Sa(T) = f(Source) × f(Path) × f(Site) (Kramer, 1996). At long periods (T > 1 s, f < 1 Hz), Sa is dominated by seismic moment — a stable source property that our 42 P-wave features capture effectively, since low-frequency waves propagate with minimal attenuation and are insensitive to shallow soil layers (wavelength >> sediment thickness). At short periods (T < 0.5 s, f > 2 Hz), the balance shifts dramatically: (a) site response becomes the dominant factor, as shallow sediment layers with resonant frequencies f₀ = V_S / 4H selectively amplify Sa at specific periods depending on local V_S30 and layer thickness (Borcherdt, 1994; Kramer, 1996); (b) the high-frequency attenuation parameter kappa (κ), which decays spectral amplitudes exponentially as A(f) = A₀·exp(−πκf), varies 0.01–0.08 s between stations (Anderson and Hough, 1984); and (c) source-level variability increases because stress drop — the primary control on high-frequency radiation — is stochastic and can vary 3–10× between events of similar magnitude (Boore, 2003). Since our pipeline has no access to V_S30, κ, or stress drop, these three sources of variability are irreducible from the model's perspective. The resulting performance varies systematically across three spectral bands: short-period (T ≤ 0.6 s, mean R² = 0.56) where site and path effects dominate, mid-period (T = 0.6–2.0 s, mean R² = 0.60) representing a transition zone where seismic moment begins to contribute alongside diminishing site influence, and long-period (T > 2.0 s, mean R² = 0.68) where seismic moment fully dominates and our P-wave features are most informative. The R² gap of ~0.12 between the short- and long-period bands is consistent with the higher aleatory sigma reported at short periods in NGA-West2 GMPEs (Abrahamson et al., 2014).

![Figure 8. Per-period Prediction Accuracy for the IDA-PTW Pipeline. R² and RMSE across 103 periods for the Java-Sumatra filtered dataset.](figures/fig08.png)

### 3.6 Safety Performance and Site-Response Robustness

#### 3.6.1 Operational Safety and Critical Miss Analysis

In the context of Earthquake Early Warning (EEWS), the primary objective is to maximize the lead time for potentially damaging ground motions while minimizing "critical misses"—scenarios where high-intensity shaking occurs without a sufficient warning window. To evaluate this, we compared the **Intensity-Driven Adaptive (IDA) PTW** strategy against the **Source-Aware (Magnitude-based)** baseline using the geographically filtered study area dataset (34,033 traces).

The results demonstrate a stark contrast in operational safety. The IDA-PTW strategy achieved a **Damaging Recall of 72.3%**, successfully identifying and extending the processing window for the vast majority of high-intensity events. In contrast, the Source-Aware model achieved a negligible **1.3% recall** for damaging intensities. This discrepancy indicates that magnitude-centric routing is fundamentally "blind" to high-intensity shaking when it is driven by factors other than source energy.

**Table 7.** Safety metrics for IDA vs. Source-Aware routing on the filtered Sunda Arc dataset.

| Strategy | Recall (Damaging Class) | Critical Miss Rate (Damaging → Weak) |
| :--- | :---: | :---: |
| **IDA-PTW (Proposed)** | **72.3%** | **8.9%** |
| **Source-Aware (Baseline)** | **1.3%** | **~98%** |

#### 3.6.2 Robustness to Local Site Effects

The failure of the Source-Aware approach to detect damaging events highlights a critical physical reality of the Indonesian seismic environment: the prevalence of localized site effects. Out of the 101 traces classified as **Damaging** (PGA ≥ 62.0 Gal) within the study area (Table 1), a substantial proportion originated from earthquakes with moderate magnitudes (M < 5.0).

From a source-inversion perspective, a Magnitude 4.5 event is typically considered non-damaging and is consequently routed by the Source-Aware logic to a standard 3-second P-wave window. However, due to shallow crustal dynamics or unconsolidated soil conditions (basin effects), these "minor" events can produce destructive peak ground accelerations at specific station sites. By utilizing **accelerographs**—which capture high-frequency onsite vibrations—and employing an **Intensity-Driven (IDA)** logic, our proposed system responds directly to the **observed intensity** at the sensor. This makes IDA-PTW inherently more robust to local site amplifications, ensuring that even moderate-magnitude events triggering damaging ground motions are granted extended processing time (up to 6–8 s depending on distance) for accurate spectral acceleration prediction.

### 3.7 Comprehensive Performance Metrics (Dai et al., 2024 Framework)

To facilitate direct comparison with prior Sa(T) prediction studies, we evaluate the full operational IDA-PTW pipeline using the comprehensive metrics framework proposed by Dai et al. (2024), which decomposes model performance into six complementary diagnostic dimensions. All metrics are computed on the end-to-end operational predictions (predicted intensity class and predicted distance) using 5-fold event-grouped cross-validation, strictly following the Honest Evaluation protocol.

#### 3.7.1 Variability Decomposition (τ, φ, σ)

Following Al Atik et al. (2010), we decompose the total aleatory variability (σ) of the prediction residuals into inter-event (τ) and intra-event (φ) components. The results yield σ_total ≈ 0.752 log₁₀ units, with the intra-event component φ dominating at approximately 70% of total variance. This dominance of φ over τ reveals that site-to-site and path-dependent variability constitutes the primary source of prediction uncertainty — not earthquake source differences. Compared with conventional GMPEs such as NGA-West2 (Abrahamson et al., 2014; Chiou and Youngs, 2014), which report σ_total ≈ 0.65–0.75, the IDA-PTW model achieves comparable variability levels despite operating under severe single-station EEWS constraints with only 2–8 seconds of P-wave data and no site characterization parameters. The clear dominance of φ directly implies that Vs30 data enrichment is the single most impactful improvement pathway for reducing total prediction uncertainty (Figure 10).

#### 3.7.2 Period-Dependent Accuracy Metrics (R², RMSE, Bias)

R² increases monotonically from approximately 0.30 at short structural periods (T ≈ 0.1–0.3 s) to approximately 0.49 at long periods (T ≈ 4–5 s), while RMSE decreases correspondingly. This trend is physically consistent with seismic wave theory: low-frequency content dominating long-period Sa is better preserved and reconstructed from early P-wave arrivals. Critically, the mean bias remains near zero (within ±0.05 log₁₀ units) across all 103 periods, confirming that the model exhibits no systematic over- or underprediction tendency — a fundamental requirement for operational EEWS deployment. This pattern aligns with established findings from EEWS studies using different methodologies (Kodera et al., 2018; Böse et al., 2009), validating the physical basis of the XGBoost-based approach (Figure 11).

#### 3.7.3 Within-Factor Accuracy

The within-factor analysis reveals that 83.3% of predictions fall within ±1.0 log₁₀ units, 54.4% within ±0.5 log₁₀, and 30.8% within the tightest margin of ±0.3 log₁₀. Accuracy is notably higher at longer periods (T > 1.0 s), which is particularly significant for EEWS applications because long-period spectral acceleration correlates most strongly with structural damage in medium-to-high-rise buildings — the primary targets for protective action. In practical terms, for a building with fundamental period T = 2.0 s, the model predicts Sa within a factor of approximately 3 for over half the cases, enabling meaningful engineering-grade rapid damage assessment within seconds of P-wave arrival (Figure 12).

#### 3.7.4 Magnitude-Period R² Distribution

The magnitude-period heatmap exposes the saturation limitation inherent to all single-station EEWS approaches. The model performs excellently for moderate earthquakes (M 4.0–6.0), where R² consistently exceeds 0.5 across all structural periods. However, performance degrades severely for M ≥ 6.5, with R² values becoming negative for M 7+, indicating predictions worse than the sample mean. This is not a modeling deficiency but a fundamental physics constraint well-documented in the EEWS literature (Yamada and Mori, 2009; Zollo et al., 2010): early P-wave features (2–8 s) at a single station cannot capture the full rupture extent of large earthquakes whose fault planes may exceed 100 km. The IDA-PTW model remains highly effective within its operational domain (M 4.0–6.0, representing >95% of Indonesian seismicity), while future work on saturation-aware magnitude-dependent bias corrections and multi-station data fusion — such as PLUM-style approaches (Kodera et al., 2018) — can progressively address this limitation for rare large events (Figure 13).

#### 3.7.5 Residual Diagnostics (Observed vs. Predicted, Q-Q Plot)

The observed-versus-predicted scatter plot shows tight clustering around the 1:1 reference line across all 103 periods, with characteristic underprediction visible only at extreme high values corresponding to large-magnitude and near-field recordings — consistent with the saturation limitation identified in Section 3.7.4. The Q-Q plot of residuals confirms a near-normal distribution with slight heavy tails, validating the log-normal statistical framework standard in engineering seismology (Joyner and Boore, 1981; Abrahamson and Silva, 2008). This dual confirmation of model unbiasedness ensures that probabilistic seismic hazard calculations using IDA-PTW predictions will produce reliable exceedance probability estimates for engineering decision-making under the SNI 1726:2019 framework (Figure 14).

#### 3.7.6 Distance-Dependent Bias

Mean bias as a function of source-station distance reveals a strong distance dependence. At very short distances (D < 30 km), predictions exhibit severe underprediction with bias values ranging from −1.5 to −2.0 log₁₀ units, caused by complex near-field effects including directivity and hanging-wall amplification. Beyond 100 km — where 94.1% of operational BMKG EEWS recordings occur — the model is remarkably well-calibrated with bias remaining close to zero. This constitutes the second major limitation alongside magnitude saturation, and both share a root cause: insufficient near-field and large-magnitude representation in the training data (only 5.9% of traces at D < 100 km). However, this is not operationally critical for BMKG's current deployment configuration, as the offshore Sunda subduction zone seismicity recorded by the land-based accelerograph network is predominantly far-field. Future improvements via distance-dependent bias correction terms and targeted near-field data augmentation can progressively address this limitation as BMKG's network densifies (Figure 15).

---

## 4. Discussion

### 4.1 Feature Dichotomy as a Design Paradigm

The most significant contribution of this work is the empirical demonstration of the *Feature Dichotomy* between classification and regression stages of a dual-stage EEWS (Figure 4). Prior EEWS studies treat P-wave IMs as a homogeneous pool, feeding all parameters indiscriminately into either classification or regression models (Wu and Kanamori, 2005b; Zollo et al., 2010; Münchmeyer et al., 2021). Our results show that the XGBoost algorithm naturally partitions features by their saturation properties when given the opportunity through a dual-stage architecture.

The physical basis for this partitioning is grounded in seismic source theory. τ_c reflects the corner frequency of the source spectrum, which scales with magnitude up to M ≈ 6.5 before plateauing due to finite rupture duration (Kanamori, 2005; Lancieri and Zollo, 2008). This ceiling creates discrete "steps" in the τ_c–magnitude space that are ideally suited for classification. CVAD, as a time-cumulative integral of velocity–displacement products, continues to accumulate information throughout the P-wave window without saturation, providing a continuous signal for regression.

This paradigm suggests a generalizable EEWS design principle: **separate the classification problem (What category is this earthquake?) from the prediction problem (What will the ground motion be?), and allow each stage to select its own optimal feature subset based on physical information content.** Future EEWS could extend this by incorporating additional saturating features (e.g., P_d) in the classifier and additional cumulative features (e.g., Arias intensity) in the regressor.

### 4.2 Intensity-Driven Routing Sufficiency and Distance Robustness

A key finding is that the Stage 1 intensity classification alone — without any distance information — already provides effective PTW routing. Intensity-only routing (Weak → 3 s, Felt → 4 s, Damaging → 6 s) achieves a composite R² = 0.7436, which is *higher* than the distance-aware operational routing (R² = 0.7309) by ΔR² = +0.013. This occurs because the intensity-only strategy concentrates 94% of traces at PTW = 3 s — already near-optimal for a dataset where the predominantly offshore location of Sunda subduction zone seismicity places the vast majority of source-station paths in the far-field regime (94.1% at D > 100 km, median 384 km), consistent with the operational coverage of BMKG's land-based accelerograph network.

The distance estimator (Stage 1.5) therefore does not improve prediction accuracy in the current dataset; rather, its role is architectural: it serves as a **golden time safety constraint** to ensure that the selected PTW never exceeds the S−P travel-time budget, particularly for near-field events where warning time is tightest. The modular pipeline design allows Stage 1.5 to be bypassed without performance degradation in far-field-dominated deployments, while remaining available for scenarios with significant near-field exposure.

Importantly, the near-identical E2E performance between predicted and oracle distance routing (R² = 0.7309 versus R² = 0.7311, ΔR² = 0.0002) demonstrates that when distance-aware routing is engaged, the system is extremely robust to distance estimation errors. This robustness arises from two properties: (a) distance serves only as a discrete routing signal across 6 bins, not as a continuous prediction input — errors that stay within the same bin produce identical PTW selections; and (b) the intensity class already captures the dominant routing variable, reducing the marginal contribution of distance. The routing fidelity of 99.97% confirms that the distance estimator's modest standalone accuracy does not propagate to the final output.

This architecture simplifies operational deployment: **the pipeline requires zero external information from seismic networks or catalogs. A single station can autonomously determine the optimal balance between lead time and accuracy based solely on its own P-wave signal analysis.** The "Honest Evaluation" protocol confirms that the system maintains competitive accuracy under fully autonomous, single-station conditions.

### 4.3 Comparison with Existing Methods

Table 8 presents a performance comparison against established and recent EEWS methods, while Table 9 provides a capability matrix that highlights the methodological scope of each approach.

**Table 8.** Performance comparison with existing EEWS and ground motion prediction methods. Studies are grouped by prediction target. N_T = number of spectral periods predicted. Performance is reported as composite R² where available, or residual standard deviation σ (in log₁₀ units).

| Study | Method | Target | N_T | Single-station | PTW | Performance |
|---|---|---|---|---|---|---|
| Wu and Kanamori (2005b) | τ_c–Pd empirical | PGA | 1 | Yes | Fixed 3 s | R² ≈ 0.70–0.75 |
| Zollo et al. (2010) | Bayesian decision tree | PGA/PGV | 2 | Yes | Fixed 2–4 s | R² ≈ 0.65–0.80 |
| Hsu and Huang (2021) | CNN (multi-scale) | PGA | 1 | Yes | Fixed 3 s | Improves over Pd–PGA |
| Wang et al. (2023) | LSTM | PGA | 1 | Yes | 1–30 s | σ = 0.656 (3 s) |
| Liu et al. (2024) | CNN (DLPGA) | PGA | 1 | Yes | Fixed 3–6 s | σ = 0.289 (3 s) |
| Iaccarino et al. (2024) | Gradient Boosting | PGA + Dist | 1 | Yes | Fixed 1–3 s | R² = 0.90–0.95 (3 s) |
| Münchmeyer et al. (2021) | Transformer (TEAM) | PGA (prob.) | 1 | No (multi) | Variable | σ = 0.29–0.33 |
| Iaccarino et al. (2020) | Empirical (Pd, Iv2) | Sa(T) | 9 | Yes | Fixed 3 s | σ ≈ 0.30–0.35 |
| Lara et al. (2023) | Random Forest | Sa(T) | ~20 | No (network) | Fixed 4 s | R² = 0.88–0.92 |
| Dai et al. (2024) | XGBoost | Sa(T) | Multiple | Yes | Fixed 3–10 s | MSE = 15.1×10⁻⁴ g |
| Abrahamson et al. (2014) | NGA-West2 GMPE | Sa(T) | ~20 | No (network+site) | Post-event | σ_total ≈ 0.65–0.80 |
| **This study** | **XGBoost Dual-Stage** | **Sa(T)** | **103** | **Yes** | **Adaptive 3–8 s** | **R² = 0.73** |

**Table 9.** Capability matrix. A check mark (✓) indicates the study addresses the corresponding capability; a dash (–) indicates not addressed. Capabilities: (A) Predicts full spectral Sa(T), (B) Spectral coverage > 20 periods, (C) Single-station (no network aggregation), (D) No catalog information required (fully autonomous), (E) Adaptive P-wave time window, (F) Reports safety metrics (e.g., Damaging recall), (G) Event-grouped cross-validation.

| Study | (A) Sa(T) | (B) >20 T | (C) Single | (D) No catalog | (E) Adaptive PTW | (F) Safety | (G) Event-CV |
|---|---|---|---|---|---|---|---|
| Wu and Kanamori (2005b) | – | – | ✓ | ✓ | – | – | – |
| Zollo et al. (2010) | – | – | ✓ | ✓ | – | ✓ | – |
| Hsu and Huang (2021) | – | – | ✓ | ✓ | – | – | – |
| Wang et al. (2023) | – | – | ✓ | ✓ | – | – | – |
| Liu et al. (2024) | – | – | ✓ | ✓ | – | – | – |
| Iaccarino et al. (2024) | – | – | ✓ | ✓ | – | – | – |
| Münchmeyer et al. (2021) | – | – | – | – | – | ✓ | ✓ |
| Iaccarino et al. (2020) | ✓ | – | ✓ | ✓ | – | ✓ | – |
| Lara et al. (2023) | ✓ | – | – | – | – | – | – |
| Dai et al. (2024) | ✓ | – | ✓ | ✓ | – | – | – |
| Abrahamson et al. (2014) | ✓ | ✓ | – | – | – | – | – |
| **This study** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Tables 8 and 9 reveal four key findings regarding the positioning of this work. First, no prior single-station EEWS predicts Sa(T) at more than ~20 spectral periods; this study extends coverage to 103 periods (T = 0.1–5.0 s), a 5–10× increase that enables structure-specific hazard assessment. Second, studies achieving higher R² (e.g., Lara et al., 2023: R² = 0.88–0.92; Iaccarino et al., 2024: R² = 0.90–0.95) operate under substantially easier conditions — multi-station aggregation, shorter distance ranges (< 300 km), PGA-only targets, or regional datasets with lower heterogeneity. Our R² = 0.73 is achieved under the most challenging configuration: single station, no site characterization, distances up to 867 km, and 103 spectral targets simultaneously. Third, the adaptive PTW capability (column E) is unique to this study; all prior methods use fixed time windows, forgoing the explicit tradeoff between prediction accuracy and warning timeliness. Fourth, the capability matrix shows that this study is the only approach satisfying all seven criteria simultaneously, positioning it as the most comprehensive single-station EEWS framework currently available for engineering-grade spectral prediction.

### 4.4 Implications for Indonesian EEWS

These results offer specific implications for enhancing Indonesia's InaTEWS infrastructure:

**Near-field complement.** The pipeline could serve as a fast initial-estimate module that operates within the first 3 s of P-wave arrival, providing station-level Sa(T) estimates before InaTEWS's network-based solution converges. For the 2018 M 7.5 Palu earthquake, where the network solution required ~30 s, a single-station adaptive PTW estimate could have been available within 5 s.

**Sparse network resilience.** Much of eastern Indonesia has station spacing exceeding 200 km. For events in these gaps, single-station predictions may be the *only* available estimate within the golden time window. The distance-aware routing ensures that these far-field predictions use optimally long PTW windows.

**Spectral engineering information.** Current InaTEWS outputs intensity (MMI) estimates. Sa(T) predictions would enable direct linkage to building vulnerability models developed under SNI 1726:2019 (the Indonesian seismic design code), supporting automated damage estimation and response prioritization.

### 4.5 Limitations and Future Work

Several limitations warrant discussion:

**Operational Lifecycle.** While the theoretical bounds of early warning timelines are primarily governed by seismic wave propagation mapping (Minson et al., 2021), deploying and maintaining the stable operation of machine learning EEWS pipelines in continuous production environments introduces substantial data management and concept drift challenges (Ahn et al., 2023b). Future deployment of our proposed pipeline must integrate strict real-time telemetry monitoring.

**Class imbalance.** The *Damaging* class (PGA ≥ 62.0 Gal) comprises only 101 traces (0.3% of the dataset), reflecting the natural rarity of near-field strong shaking within the BMKG accelerograph network: only 1.2% of source-station paths fall within 50 km, and strong-motion events (M ≥ 5.5) at such distances account for fewer than 0.05% of all recordings. This is not a dataset deficiency but a faithful representation of the seismicity-coverage configuration of the Java-Sumatra region. While we address this through class-weighted training and SMOTE-oversampling in Stage 1 training (Table S1), the IDA framework inherently accommodates this imbalance by design: intensity-driven routing does not depend on near-field sample density, as the intensity class is determined by the observed PGA at the sensor regardless of distance. Future iterations should incorporate near-field records from dense local networks or synthetic seismic records to further improve Damaging recall.

**Regional specificity.** All models are trained on Sunda subduction zone data. Transferability to different tectonic regimes (e.g., the Philippine Sea Plate boundary, transform faults) requires domain adaptation. The Feature Dichotomy paradigm itself, however, is physics-based and should generalize across tectonic settings.

**Site effects.** The absence of site characterization parameters (V_S30, Z_2.5) is the dominant factor limiting short-period Sa predictions (T = 0.3–0.6 s, R² ≈ 0.54), as demonstrated by the 23% R² gap between short and long periods (Section 3.5). Following the NGA-West2 convention (Ancheta et al., 2014), we classify V_S30 values into two categories: *measured* (field survey by BMKG using HVSR, MASW, or borehole methods) and *inferred* (proxy-based estimates from the Allen and Wald (2009) global topographic slope model). Of the 854 BMKG stations in our network inventory, 338 (39.6%) have measured V_S30 spanning NEHRP site classes A through E (71.9–1949.3 m/s; Class D dominant at 51.2%), while the remaining 516 (60.4%) rely on inferred values characterized by five discrete proxy assignments (300, 320, 230, 600, and 430 m/s) with substantially higher uncertainty (σ_ln(V_S30) ≈ 0.36–0.44 for inferred vs. ≈ 0.05–0.15 for measured; Seyhan and Stewart, 2014).

A systematic V_S30 integration experiment confirmed the impact of this classification in two phases. In Phase 1, retraining all 515 models with 44 features (42 P-wave IMs + V_S30 + ln V_S30) using predominantly inferred V_S30 (203 measured stations, ~27% of training data) yielded a mean ΔR² = +0.030 (+5.1%) across all 103 spectral periods without exception. In Phase 2, the measured dataset was expanded to 338 BMKG stations, replacing inferred values with measured V_S30 at 265 stations (117,965 traces, 61.3% of training data). This produced dramatically larger improvements: mean ΔR² = +0.143 (+24.6%) over baseline, with 100% of periods (103/103) improving. Short-period performance (T = 0.05–0.6 s) improved by ΔR² = +0.194, while long-period performance (T ≥ 3.0 s) improved by ΔR² = +0.171. PGA prediction R² increased from 0.516 to 0.696, and Sa(1.0 s) from 0.527 to 0.705. The near-doubling of V_S30 standard deviation (130.3 → 204.9 m/s) enabled XGBoost to better resolve site amplification effects that were previously collapsed into discrete proxy bins. These results unequivocally confirm that the transition from inferred to measured V_S30 is the single most impactful improvement pathway. Extending field measurements to the remaining 516 inferred stations, combined with the site dominant period T_dom (Seyhan and Stewart, 2014), would yield further gains, particularly at short periods where site amplification effects are strongest.

**Deep learning comparison.** This study uses gradient-boosted trees (XGBoost) exclusively. Convolutional neural networks operating directly on raw waveforms (Mousavi et al., 2020) could potentially extract additional features not captured by our hand-crafted 42-feature set, particularly transient waveform characteristics relevant to distance estimation.

---

## 5. Conclusions

We present a Saturation-Aware Dual-Stage Adaptive P-Wave Time Window framework for single-station earthquake early warning with the following key findings:

1. **Feature Dichotomy.** The systematic separation of saturating P-wave parameters (τ_c, 51% importance in Stage 1) for intensity classification from non-saturating parameters (CVAD, 89.5% importance in Stage 2) for Sa(T) regression emerges naturally from source physics and provides a new design paradigm for EEWS.

2. **Operational Safety Tradeoff.** IDA-PTW achieves a composite R² = 0.73 while maintaining 72.3% Damaging recall and >99% golden time compliance. The conservative routing distribution (99.9% of traces at PTW = 3–4 s) represents a deliberate tradeoff prioritizing operational safety over the marginal R² gains of longer, unconstrained fixed windows.

3. **Comprehensive spectral prediction.** The pipeline predicts Sa(T) at 103 periods (T = 0.1–5.0 s) with R² = 0.73 using only initial 3-second P-wave signals from a single station in the Java-Sumatra study area.

4. **Self-correcting robustness.** The system is robust to distance estimation errors: the PTW selection mechanism self-corrects because routing errors toward longer windows are compensated by the monotonic PTW–accuracy relationship.

5. **Comprehensive evaluation diagnostics.** Following the Dai et al. (2024) framework, the model exhibits σ_total ≈ 0.752 log₁₀ (comparable to NGA-West2 GMPEs despite single-station constraints), near-zero mean bias across all 103 periods, 83.3% of predictions within ±1.0 log₁₀, and near-normal residual distribution validating the log-normal assumption. Two clearly identified limitations — magnitude saturation (M ≥ 6.5) and near-field bias (D < 30 km) — provide concrete directions for future improvement through saturation-aware corrections and Vs30 data enrichment.

6. **V_S30 enrichment validation.** A systematic experiment integrating measured V_S30 data from 338 BMKG stations (expanded from 203 proxy-based estimates) into the 44-feature model (42 P-wave IMs + V_S30 + ln V_S30) achieved ΔR² = +0.143 (+24.6%) over baseline across all 103 periods without exception, confirming that site characterization is the single most impactful improvement pathway for reducing intra-event variability (φ), which dominates 70% of total prediction uncertainty.

The 517-model pipeline (1 classifier, 1 distance estimator, 515 Sa(T) regressors), validated with strict event-grouped cross-validation, is openly available at https://github.com/hanif7108/adaptive-ptw-eews for integration into operational EEWS infrastructure.

---

## Data Availability Statement

The raw seismic waveform data used in this study were provided by the Agency for Meteorology, Climatology and Geophysics of the Republic of Indonesia (BMKG) and are available upon reasonable request from the provider (https://dataexchange.bmkg.go.id/). The source code for the complete dual-stage pipeline — including feature extraction, model training, evaluation scripts, and publication figures — is openly available at https://github.com/hanif7108/adaptive-ptw-eews. The processed Sunda v2 (Geomean) dataset, formatted in SeisBench-compatible HDF5 containers, and the trained model artifacts (517 XGBoost models) are available upon reasonable request to the corresponding author due to their large file size (~4 GB for waveforms, ~500 MB for pre-extracted features).

## Ethics Statement

This study uses publicly archived seismic waveform data from the BMKG operational network. No human subjects, animals, or classified data were involved. No ethical approval was required.

## Author Contributions

**HAN:** Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing — Original Draft, Visualization. **DD:** Supervision, Validation, Writing — Review & Editing. **AHS:** Supervision, Resources, Writing — Review & Editing. **SP:** Data Curation, Validation, Writing — Review & Editing. All authors reviewed and approved the submitted version.

## Funding

This research is part of the doctoral program at the Faculty of Mathematics and Natural Sciences (FMIPA), Universitas Indonesia, funded by the Agency for Meteorology, Climatology and Geophysics of the Republic of Indonesia (BMKG). The authors are grateful for the computational resources provided by the Department of Physics, Universitas Indonesia.

## Acknowledgments

We thank the Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) for providing accelerometric waveform data from the Indonesian seismic network. We acknowledge the SeisBench development team for the standardized benchmark framework.

## Conflict of Interest

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

## Supplementary Material

The Supplementary Material for this article can be found online. It includes:

- **Table S1:** Complete XGBoost hyperparameters for all three pipeline stages (Intensity Gate classifier, distance estimator, and 515 Sa(T) regressors).
- **Table S2:** The 6 x 6 Intensity-Distance PTW lookup table with golden time constraint parameters.

---

## References

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

Kanamori, H. (2005). Real-time seismology and earthquake damage mitigation. *Annual Review of Earth and Planetary Sciences*, 33, 195–214.

Kodera, Y., Hayashimoto, N., Tamaribuchi, K., Noguchi, K., Moriwaki, K., Takahashi, R., Morimoto, M., Okamoto, K., Hoshiba, M. (2021). Developments of the Nationwide Earthquake Early Warning System in Japan After the 2011 Mw9.0 Tohoku-Oki Earthquake. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.726045

Kopp, H., Flueh, E. R., Petersen, C. J., Weinrebe, W., Wittwer, A., and Meramex Scientists (2008). The Java margin revisited: Evidence for subduction erosion off Java. *Earth and Planetary Science Letters*, 242(1–2), 130–142.

Kramer, S. L. (1996). *Geotechnical Earthquake Engineering*. Prentice Hall, Upper Saddle River, NJ.

Ladina, C., Marzorati, S., Amato, A., Cattaneo, M. (2021). Feasibility Study of an Earthquake Early Warning System in Eastern Central Italy. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.685751

Lancieri, M., and Zollo, A. (2008). A Bayesian approach to the real-time estimation of magnitude from the early P and S wave displacement peaks. *Journal of Geophysical Research*, 113, B12302.

Lara, P., Bletery, Q., Ampuero, J.-P., Inza, A., and Tavera, H. (2023a). Earthquake ground motion estimation from P-waves with machine learning. *Nature Communications*, 14, 3082.

Liu, Y., Zhao, Q., and Wang, Y. (2024). Peak ground acceleration prediction for on-site earthquake early warning with deep learning. *Scientific Reports*, 14, 5039. doi: 10.1038/s41598-024-56004-6

Li, J., Böse, M., Feng, Y., Yang, C. (2021). Real-Time Characterization of Finite Rupture and Its Implication for Earthquake Early Warning: Application of FinDer to Existing and Planned Stations in Southwest China. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.699560

Massin, F., Clinton, J., Böse, M. (2021). Status of Earthquake Early Warning in Switzerland. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.707654

Minson, S. E., Meier, M.-A., Baltay, A. S., Hanks, T. C., and Cochran, E. S. (2018). The limits of earthquake early warning: Timeliness of ground motion estimates. *Science Advances*, 4(3), eaaq0504.

Minson, S., Cochran, E., Wu, S., Noda, S. (2021). A Framework for Evaluating Earthquake Early Warning for an Infrastructure Network: An Idealized Case Study of a Northern California Rail System. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.620467

Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuber, L. Y., and Beroza, G. C. (2020). Earthquake transformer — an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications*, 11(1), 3952.

Münchmeyer, J., Bindi, D., Leser, U., and Tilmann, F. (2021). The transformer earthquake alerting model: A new versatile approach to earthquake early warning. *Geophysical Journal International*, 225(1), 646–656. doi: 10.1093/gji/ggaa609

Nakamura, Y. (1988). On the urgent earthquake detection and alarm system (UrEDAS). In *Proceedings of the 9th World Conference on Earthquake Engineering*, Tokyo, VII, 673–678.

Natawidjaja, D. H., Sieh, K., Chlieh, M., et al. (2006). Source parameters of the great Sumatran megathrust earthquakes of 1797 and 1833. *Journal of Geophysical Research*, 111, B06403.

Nof, R., Lior, I., Kurzon, I. (2021). Earthquake Early Warning System in Israel—Towards an Operational Stage. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.684421

Parolai, S., Boxberger, T., Pilz, M., Fleming, K., Haas, M., Pittore, M., Petrovic, B., Moldobekov, B., Zubovich, A., Lauterjung, J. (2017). Assessing Earthquake Early Warning Using Sparse Networks in Developing Countries: Case Study of the Kyrgyz Republic. *Frontiers in Earth Science*, 5. doi: 10.3389/feart.2017.00074

Peng, H., Wu, Z., Wu, Y. M., Yu, S. S., Zhang, G., and Huang, W. (2021). Progress of Earthquake Early Warning in China and Its Application in the 2019 Mw 5.8 Changning Earthquake, Sichuan. *Frontiers in Earth Science*, 9, 699560.

Porras, J., Massin, F., Arroyo-Solórzano, M., Arroyo, I., Linkimer, L., Böse, M., Clinton, J. (2021). Preliminary Results of an Earthquake Early Warning System in Costa Rica. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.700843

Reed, J. W., and Kassawara, R. P. (1990). A criterion for determining exceedance of the operating basis earthquake. *Nuclear Engineering and Design*, 123(2–3), 387–396.

Schlesinger, A., Kukovica, J., Rosenberger, A., Heesemann, M., Pirenne, B., Robinson, J., Morley, M. (2021). An Earthquake Early Warning System for Southwestern British Columbia. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.684084

Seyhan, E., and Stewart, J. P. (2014). Semi-empirical nonlinear site amplification from NGA-West2 data and simulations. *Earthquake Spectra*, 30(3), 1241–1256. doi: 10.1193/063013EQS181M

Simons, W. J. F., Socquet, A., Vigny, C., et al. (2007). A decade of GPS in Southeast Asia: Resolving Sundaland motion and boundaries. *Journal of Geophysical Research*, 112, B06420.

Tsuno, S. (2021). Applicability of On-Site P-Wave Earthquake Early Warning to Seismic Data Observed During the 2011 Off the Pacific Coast of Tohoku Earthquake, Japan. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.681199

Velazquez, O., Pescaroli, G., Kaner, J., et al. (2020). A Review of Earthquake Early Warning Systems: Progress, Challenges, and Implications for Cities. *Frontiers in Built Environment*, 6. doi: 10.3389/fbuil.2020.00124

Wang, A., Li, S., et al. (2023). Prediction of PGA in earthquake early warning using a long short-term memory neural network. *Geophysical Journal International*, 234(1), 12–24. doi: 10.1093/gji/ggad067

Wang, Z., Zhao, B. (2021). Applicability of Accurate Ground Motion Estimation Using Initial P Wave for Earthquake Early Warning. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.718216

Woollam, J., Münchmeyer, J., Tilmann, F., et al. (2022). SeisBench — A toolbox for machine learning in seismology. *Seismological Research Letters*, 93(3), 1695–1709.

Wu, Y.-M., and Kanamori, H. (2005a). Rapid assessment of damage potential of earthquakes in Taiwan from the beginning of P waves. *Bulletin of the Seismological Society of America*, 95(3), 1181–1185.

Wu, Y.-M., and Kanamori, H. (2005b). Experiment on an onsite early warning method for the Taiwan early warning system. *Bulletin of the Seismological Society of America*, 95(1), 347–353.

Yamada, M., and Mori, J. (2009). Using τ_c to estimate magnitude for earthquake early warning and effects of near‐field terms. *Journal of Geophysical Research*, 114, B05301.

Zhang, J., Zhu, W., and Ross, Z. E. (2024). Machine learning-based rapid epicentral distance estimation from a single station. *Bulletin of the Seismological Society of America*, 114(3), 1507–1522.

Zhu, J., Li, S., Song, J., Wang, Y. (2021). Magnitude Estimation for Earthquake Early Warning Using a Deep Convolutional Neural Network. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.653226

Zhu, W., Mousavi, S. M., and Beroza, G. C. (2023). Phase neural operator for multi‐station picking of seismic arrivals. *Geophysical Research Letters*, 50(1), e2022GL101045.

Zollo, A., Amoroso, O., Lancieri, M., Wu, Y.-M., and Kanamori, H. (2010). A threshold-based earthquake early warning using dense accelerometer networks. *Geophysical Journal International*, 183(2), 963–974.

Zuccolo, E., Cremen, G., Galasso, C. (2021). Comparing the Performance of Regional Earthquake Early Warning Algorithms in Europe. *Frontiers in Earth Science*, 9. doi: 10.3389/feart.2021.686272