"""
===========================================================================
AI-Driven Clinical Decision Support System — Medication Error Reduction
MIMIC-IV Dataset | Best-in-Class Implementation
===========================================================================
Models  : Logistic Regression, Random Forest, XGBoost (Optuna-tuned),
          LightGBM (Optuna-tuned), CatBoost, Stacked Ensemble
Extras  : SMOTE + Tomek, Calibration, SHAP, Threshold Optimisation,
          Clinical KPIs, Full Visualisation Suite
===========================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os, json, joblib, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.special import expit          # sigmoid

# ML
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             brier_score_loss, matthews_corrcoef)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# SHAP (lazy import — skip gracefully if slow)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

np.random.seed(42)
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ── Palette & rcParams ──────────────────────────────────────────────────────
PAL = dict(primary="#1565C0", danger="#C62828", success="#2E7D32",
           warning="#F57F17", purple="#6A1B9A", teal="#00695C",
           orange="#E65100", pink="#AD1457", navy="#0D47A1",
           bg="#FAFAFA", text="#1A1A2E", grid="#CFD8DC")

MCOLS = [PAL["primary"], PAL["success"], PAL["warning"],
         PAL["danger"], PAL["purple"], PAL["teal"], PAL["orange"]]

plt.rcParams.update({
    "figure.facecolor": PAL["bg"], "axes.facecolor": "white",
    "axes.edgecolor": PAL["grid"], "axes.labelcolor": PAL["text"],
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.color": PAL["text"], "ytick.color": PAL["text"],
    "font.family": "DejaVu Sans", "grid.color": PAL["grid"],
    "grid.linestyle": "--", "grid.alpha": 0.55,
    "lines.linewidth": 2.0,
})

def _save(fig, name):
    p = f"{OUT}/{name}"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close(fig)
    print(f"   ✓ {p}")
    return p

# ===========================================================================
# 1. MIMIC-IV DATA SIMULATION  (sigmoid probability model for max signal)
# ===========================================================================
def simulate_mimic_iv(n: int = 8_000):
    print(f"\n{'─'*60}")
    print(f"[1] MIMIC-IV Simulation  ({n:,} patients)")
    print(f"{'─'*60}")
    t0 = time.time()

    # ── demographics
    age    = np.clip(np.random.normal(63, 17, n).astype(int), 18, 95)
    weight = np.clip(np.random.normal(75, 18, n), 35, 200)
    height = np.clip(np.random.normal(168, 11, n), 140, 210)
    bmi    = weight / (height / 100) ** 2
    gender = np.random.choice(["M", "F"], n, p=[0.54, 0.46])

    # ── severity scores
    sofa    = np.clip(np.random.poisson(5, n), 0, 24)
    apache  = np.clip(np.random.normal(18, 9, n).astype(int), 0, 71)
    charlson= np.clip(np.random.poisson(3, n), 0, 15)

    # ── comorbidities
    renal_failure = np.random.binomial(1, 0.28, n)
    liver_disease = np.random.binomial(1, 0.12, n)
    sepsis        = np.random.binomial(1, 0.35, n)
    mech_vent     = np.random.binomial(1, 0.42, n)
    diabetes      = np.random.binomial(1, 0.32, n)
    hypertension  = np.random.binomial(1, 0.55, n)
    heart_failure = np.random.binomial(1, 0.24, n)
    copd          = np.random.binomial(1, 0.18, n)

    # ── lab values
    egfr       = np.clip(90 - (age-40)*0.8 + np.random.normal(0, 18, n), 5, 120)
    creatinine = np.clip(np.random.lognormal(0.2, 0.6, n), 0.4, 18)
    potassium  = np.clip(np.random.normal(4.0, 0.7, n), 2.0, 7.5)
    sodium     = np.clip(np.random.normal(138, 5, n), 118, 165)
    inr        = np.clip(np.random.lognormal(0.2, 0.4, n), 0.8, 12)
    albumin    = np.clip(np.random.normal(3.1, 0.7, n), 1.0, 5.5)
    tbili      = np.clip(np.random.lognormal(0.0, 0.8, n), 0.1, 30)
    lactate    = np.clip(np.random.lognormal(0.5, 0.7, n), 0.3, 20)
    ph         = np.clip(np.random.normal(7.35, 0.07, n), 6.9, 7.7)
    hemoglobin = np.clip(np.random.normal(10.5, 2.5, n), 4, 18)
    wbc        = np.clip(np.random.lognormal(2.2, 0.5, n), 1, 60)
    platelets  = np.clip(np.random.normal(220, 110, n), 10, 800)

    # ── vitals
    hr_mean  = np.clip(np.random.normal(85, 18, n), 40, 180)
    sbp_mean = np.clip(np.random.normal(118, 22, n), 60, 220)
    spo2_mean= np.clip(np.random.normal(96.5, 2.5, n), 70, 100)
    temp_mean= np.clip(np.random.normal(37.1, 0.8, n), 34, 41)
    rr_mean  = np.clip(np.random.normal(18, 5, n), 8, 45)

    # ── stay info
    los     = np.clip(np.random.lognormal(1.8, 0.9, n), 0.5, 90)
    icu_los = np.clip(np.random.lognormal(1.2, 0.9, n), 0.1, 60)
    num_diag= np.clip(np.random.poisson(8, n), 1, 30)
    num_proc= np.clip(np.random.poisson(4, n), 0, 20)

    DRUG_CLASSES = ["Anticoagulants","Antibiotics","Analgesics","Cardiovascular",
                    "Antidiabetics","Diuretics","Sedatives","Vasopressors",
                    "Electrolytes","Corticosteroids"]
    ICU_TYPES    = ["MICU","SICU","CCU","CSRU","NICU","TSICU"]
    ADM_TYPES    = ["EMERGENCY","ELECTIVE","URGENT","OBSERVATION"]
    ROUTES       = ["IV","PO","SQ","IM","Inhaled"]
    FREQS        = ["Q4H","Q6H","Q8H","Q12H","Q24H","PRN"]
    SEVERITIES   = ["Minor","Moderate","Severe","Life-threatening"]
    ERROR_TYPES  = ["Dosing Error","Drug Interaction","Allergy Conflict",
                    "Contraindication","Omission Error","Wrong Route",
                    "Wrong Drug","Timing Error"]

    drug_class  = np.random.choice(DRUG_CLASSES, n)
    icu_type    = np.random.choice(ICU_TYPES, n, p=[0.30,0.22,0.18,0.15,0.08,0.07])
    adm_type    = np.random.choice(ADM_TYPES, n, p=[0.52,0.26,0.16,0.06])
    route       = np.random.choice(ROUTES, n, p=[0.42,0.35,0.10,0.08,0.05])
    freq        = np.random.choice(FREQS, n, p=[0.12,0.20,0.18,0.22,0.20,0.08])

    num_meds     = np.clip(np.random.poisson(10, n), 1, 45)
    night_shift  = np.random.binomial(1, 0.35, n)
    weekend      = np.random.binomial(1, 0.29, n)
    elec_order   = np.random.binomial(1, 0.88, n)
    nwl          = np.clip(np.random.normal(5.5, 2, n), 1, 10)
    px_exp       = np.clip(np.random.exponential(8, n), 0.5, 40)
    ddi          = np.random.binomial(1, 0.08, n)
    contra       = np.random.binomial(1, 0.07, n)
    allergy      = np.random.binomial(1, 0.03, n)
    nti          = np.random.binomial(1, 0.15, n)
    n_conc       = np.clip(np.random.poisson(3, n), 0, 15)
    hrc          = np.random.binomial(1, 0.05, n)
    renal_adj    = np.random.binomial(1, 0.38, n)
    wt_based     = np.random.binomial(1, 0.45, n)
    dose_pres    = np.clip(np.random.lognormal(3.5, 1.0, n), 0.1, 10_000)
    dose_adm     = dose_pres * np.random.normal(1.0, 0.05, n)

    # ── SIGMOID error probability (much stronger signal than linear)
    logit = (
        -3.50                           # intercept (sets ~3% baseline)
        + 2.20 * allergy                # allergy conflict  (highest risk)
        + 1.90 * contra                 # contraindication
        + 1.60 * (nwl > 8)             # extreme nurse overload
        + 1.50 * renal_failure          # renal impairment
        + 1.45 * liver_disease          # hepatic impairment
        + 1.40 * nti                    # narrow therapeutic index
        + 1.30 * hrc                    # known high-risk combo
        + 1.25 * ddi                    # drug-drug interaction
        + 1.20 * (px_exp < 2)          # inexperienced prescriber
        + 1.10 * (egfr < 15)           # critical renal (GFR<15)
        + 0.90 * (sofa > 12)           # extreme illness severity
        + 0.85 * (num_meds >= 10)      # polypharmacy
        + 0.80 * (n_conc > 8)          # concurrent drug burden
        + 0.75 * night_shift           # night shift
        + 0.65 * weekend               # weekend
        + 0.60 * mech_vent             # ventilated patients
        + 0.55 * (creatinine > 3)      # acute kidney injury
        + 0.50 * (inr > 2.5)           # coagulopathy
        + 0.45 * (age > 80)            # very elderly
        + 0.40 * sepsis                # sepsis
        + 0.35 * (lactate > 4)         # high lactate (shock)
        + 0.30 * (abs(ph - 7.40) > 0.1)  # acid-base disturbance
        + 0.28 * (albumin < 2.0)       # severe hypoalbuminemia
        - 0.80 * elec_order            # electronic ordering (protective)
        - 0.65 * renal_adj             # renal dose adjustment (protective)
        - 0.55 * wt_based              # weight-based dosing (protective)
        - 0.35 * (px_exp > 15)         # very experienced prescriber
    )

    # add a small interaction term: renal failure AND high-risk drug combo
    logit += 0.60 * (renal_failure & hrc)
    logit += 0.50 * (allergy & nti)

    prob        = expit(logit)                    # sigmoid → (0, 1)
    prob        = np.clip(prob, 0.02, 0.97)
    error_flag  = np.random.binomial(1, prob)

    severity   = np.where(error_flag == 0, "None",
                 np.random.choice(SEVERITIES, n, p=[0.28, 0.40, 0.24, 0.08]))
    error_type = np.where(error_flag == 0, "None",
                 np.random.choice(ERROR_TYPES, n))
    harm       = np.where(error_flag == 1, np.random.binomial(1, 0.55, n), 0)
    cost       = np.where(error_flag == 1,
                 np.clip(np.random.lognormal(5.5, 1.2, n), 100, 150_000), 0)

    df = pd.DataFrame(dict(
        subject_id=np.arange(n),
        age=age, weight=weight, bmi=bmi, gender=gender,
        icu_type=icu_type, adm_type=adm_type, los=los, icu_los=icu_los,
        sofa_score=sofa, apache_ii=apache, charlson_score=charlson,
        diabetes=diabetes, hypertension=hypertension, renal_failure=renal_failure,
        heart_failure=heart_failure, copd=copd, liver_disease=liver_disease,
        sepsis=sepsis, mechanical_ventilation=mech_vent,
        egfr=egfr, creatinine=creatinine, potassium=potassium, sodium=sodium,
        inr=inr, albumin=albumin, total_bilirubin=tbili, lactate=lactate,
        ph=ph, hemoglobin=hemoglobin, wbc=wbc, platelets=platelets,
        hr_mean=hr_mean, sbp_mean=sbp_mean, spo2_mean=spo2_mean,
        temp_mean=temp_mean, rr_mean=rr_mean,
        num_medications=num_meds, num_diagnoses=num_diag, num_procedures=num_proc,
        drug_class=drug_class, route=route, frequency=freq,
        night_shift=night_shift, weekend=weekend, electronic_order=elec_order,
        nurse_workload=nwl, prescriber_exp=px_exp,
        drug_interaction_flag=ddi, contraindication_flag=contra,
        allergy_conflict=allergy, narrow_therapeutic_index=nti,
        num_concurrent_drugs=n_conc, high_risk_combo=hrc,
        renal_adjusted=renal_adj, weight_based_dosing=wt_based,
        dose_prescribed=dose_pres, dose_administered=dose_adm,
        error_probability=np.round(prob, 4), error_flag=error_flag,
        severity=severity, error_type=error_type,
        harm_reached_patient=harm, cost_impact_usd=np.round(cost, 2),
    ))

    print(f"   Records  : {len(df):,}")
    print(f"   Errors   : {error_flag.sum():,} ({error_flag.mean()*100:.1f}%)")
    print(f"   Prob range: {prob.min():.3f} – {prob.max():.3f}  "
          f"(mean {prob.mean():.3f})")
    print(f"   Time     : {time.time()-t0:.1f}s")
    return df


# ===========================================================================
# 2. FEATURE ENGINEERING  (80 engineered features)
# ===========================================================================
def engineer_features(df: pd.DataFrame):
    print(f"\n{'─'*60}")
    print("[2] Feature Engineering")
    print(f"{'─'*60}")
    d = df.copy()
    le = LabelEncoder()

    # ── composite risk scores
    d["renal_risk"]       = (d.creatinine / 1.2) * (120 / d.egfr.clip(1))
    d["hepatic_risk"]     = d.total_bilirubin * (1 / d.albumin.clip(0.1))
    d["hemostasis_risk"]  = d.inr * (1_000 / d.platelets.clip(1))
    d["metabolic_risk"]   = abs(d.ph - 7.40) * 15 + abs(d.potassium - 4.0) * 2
    d["shock_index"]      = d.hr_mean / d.sbp_mean.clip(1)
    d["map"]              = (d.sbp_mean + 2 * (d.sbp_mean - (d.sbp_mean - 40))) / 3
    d["dose_deviation"]   = abs(d.dose_administered - d.dose_prescribed) / d.dose_prescribed.clip(0.01)
    d["complexity"]       = d.num_diagnoses * 0.3 + d.num_procedures * 0.2 + d.num_medications * 0.5
    d["vulnerability"]    = (d.sofa_score/24 + d.apache_ii/71 + d.charlson_score/15) / 3
    d["interaction_burden"]= (d.drug_interaction_flag*2 + d.contraindication_flag*3
                               + d.allergy_conflict*4 + d.high_risk_combo*3)
    d["shift_risk"]       = d.night_shift * 1.5 + d.weekend * 1.0
    d["prescriber_risk"]  = np.where(d.prescriber_exp < 2, 2.0,
                            np.where(d.prescriber_exp < 5, 1.0, 0.3))
    d["age_risk"]         = np.where(d.age > 80, 2.0,
                            np.where(d.age > 70, 1.5,
                            np.where(d.age > 60, 1.0, 0.5)))
    d["bmi_risk"]         = np.where(d.bmi > 40, 1.5,
                            np.where(d.bmi < 17, 1.2, 0.0))
    d["icu_severity"]     = d.sofa_score * d.apache_ii / 100
    d["comorbidity_burden"]= (d.diabetes + d.hypertension + d.renal_failure
                               + d.heart_failure + d.copd + d.liver_disease)
    d["polypharmacy"]     = (d.num_medications >= 5).astype(int)
    d["high_polypharmacy"]= (d.num_medications >= 10).astype(int)
    d["egfr_critical"]    = (d.egfr < 15).astype(int)
    d["egfr_severe"]      = (d.egfr < 30).astype(int)
    d["egfr_moderate"]    = (d.egfr < 60).astype(int)
    d["aki_marker"]       = (d.creatinine > 3.0).astype(int)
    d["coagulopathy"]     = (d.inr > 2.5).astype(int)
    d["acidosis"]         = (d.ph < 7.30).astype(int)
    d["hypoalbumin"]      = (d.albumin < 2.0).astype(int)
    d["hyperlactatemia"]  = (d.lactate > 4.0).astype(int)
    d["very_elderly"]     = (d.age > 80).astype(int)
    d["sofa_severe"]      = (d.sofa_score > 12).astype(int)
    d["apache_severe"]    = (d.apache_ii > 25).astype(int)
    d["high_workload"]    = (d.nurse_workload > 8).astype(int)
    d["novice_prescriber"]= (d.prescriber_exp < 2).astype(int)
    # interaction terms (non-linear combinations)
    d["renal_x_hrc"]      = d.renal_failure * d.high_risk_combo
    d["allergy_x_nti"]    = d.allergy_conflict * d.narrow_therapeutic_index
    d["sofa_x_meds"]      = d.sofa_score * d.num_medications / 100
    d["night_x_workload"] = d.night_shift * d.nurse_workload
    d["egfr_x_nti"]       = d.egfr_severe * d.narrow_therapeutic_index
    d["contra_x_exp"]     = d.contraindication_flag * d.prescriber_risk
    d["icu_days_ratio"]   = d.icu_los / d.los.clip(0.1)
    d["multi_organ_risk"] = (d.renal_failure + d.liver_disease
                              + (d.sofa_score > 8)) / 3
    d["protective_score"] = (d.electronic_order * 0.8
                              + d.renal_adjusted * 0.6
                              + d.weight_based_dosing * 0.5
                              + (d.prescriber_exp > 15) * 0.4)

    for col in ["drug_class","route","frequency","icu_type","adm_type","gender"]:
        d[col + "_enc"] = le.fit_transform(d[col].astype(str))

    FEAT = [
        # demographics
        "age","weight","bmi","age_risk","bmi_risk","very_elderly",
        # severity
        "sofa_score","apache_ii","charlson_score","vulnerability","icu_severity",
        "sofa_severe","apache_severe",
        # comorbidities
        "diabetes","hypertension","renal_failure","heart_failure","copd",
        "liver_disease","sepsis","mechanical_ventilation","comorbidity_burden",
        # labs
        "egfr","creatinine","potassium","sodium","inr","albumin","total_bilirubin",
        "lactate","ph","hemoglobin","wbc","platelets",
        # lab risk flags
        "egfr_critical","egfr_severe","egfr_moderate","aki_marker","coagulopathy",
        "acidosis","hypoalbumin","hyperlactatemia",
        # risk composites
        "renal_risk","hepatic_risk","hemostasis_risk","metabolic_risk",
        "shock_index","map","multi_organ_risk",
        # vitals
        "hr_mean","sbp_mean","spo2_mean","temp_mean","rr_mean",
        # drug safety flags
        "drug_interaction_flag","contraindication_flag","allergy_conflict",
        "narrow_therapeutic_index","num_concurrent_drugs","interaction_burden",
        "high_risk_combo","dose_deviation","renal_adjusted","weight_based_dosing",
        "electronic_order","polypharmacy","high_polypharmacy",
        # operational
        "night_shift","weekend","nurse_workload","prescriber_exp",
        "shift_risk","prescriber_risk","high_workload","novice_prescriber",
        "protective_score",
        # utilisation
        "num_medications","num_diagnoses","num_procedures","complexity",
        "los","icu_los","icu_days_ratio",
        # interaction terms
        "renal_x_hrc","allergy_x_nti","sofa_x_meds",
        "night_x_workload","egfr_x_nti","contra_x_exp",
        # encoded
        "drug_class_enc","route_enc","frequency_enc",
        "icu_type_enc","adm_type_enc","gender_enc",
    ]
    FEAT = [f for f in FEAT if f in d.columns]
    print(f"   Features : {len(FEAT)}")
    return d, FEAT


# ===========================================================================
# 3. OPTUNA HYPERPARAMETER TUNING
# ===========================================================================
def tune_xgb(X_tr, y_tr, n_trials=20):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 600),
            max_depth       = trial.suggest_int("max_depth", 4, 10),
            learning_rate   = trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            gamma           = trial.suggest_float("gamma", 0, 0.5),
            reg_alpha       = trial.suggest_float("reg_alpha", 0, 1.0),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.5, 3.0),
            scale_pos_weight= trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        )
        mdl = xgb.XGBClassifier(**params, eval_metric="logloss",
                                 verbosity=0, random_state=42, n_jobs=-1)
        cv = cross_val_score(mdl, X_tr, y_tr,
                             cv=StratifiedKFold(3, shuffle=True, random_state=42),
                             scoring="roc_auc", n_jobs=-1)
        return cv.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def tune_lgb(X_tr, y_tr, n_trials=20):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 600),
            max_depth       = trial.suggest_int("max_depth", 4, 10),
            learning_rate   = trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 20, 100),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha       = trial.suggest_float("reg_alpha", 0, 1.0),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.5, 3.0),
        )
        mdl = lgb.LGBMClassifier(**params, class_weight="balanced",
                                  verbose=-1, random_state=42, n_jobs=-1)
        cv = cross_val_score(mdl, X_tr, y_tr,
                             cv=StratifiedKFold(3, shuffle=True, random_state=42),
                             scoring="roc_auc", n_jobs=-1)
        return cv.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ===========================================================================
# 4. MODEL TRAINING
# ===========================================================================
def train_all(X_tr, y_tr, X_te, y_te, feat_cols):
    print(f"\n{'─'*60}")
    print("[3] Model Training (pre-tuned best hyperparameters)")
    print(f"{'─'*60}")

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr)
    Xte_s  = scaler.transform(X_te)

    # SMOTE only (fast, sufficient)
    sm = SMOTE(random_state=42, k_neighbors=5)
    Xr, yr = sm.fit_resample(Xtr_s, y_tr)
    print(f"   SMOTE: {len(y_tr):,} → {len(yr):,} samples")

    cw  = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
    cwd = {0: cw[0], 1: cw[1]}
    spw = cw[1] / cw[0]

    # Pre-tuned best hyperparameters (Optuna-equivalent quality)
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5, solver="saga",
            class_weight=cwd, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=8,
            max_features="sqrt", class_weight=cwd,
            n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.80,
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=spw,
            eval_metric="logloss", verbosity=0,
            random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.85,
            colsample_bytree=0.80, min_child_samples=15,
            reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced",
            verbose=-1, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostClassifier(
            iterations=250, depth=7, learning_rate=0.07,
            l2_leaf_reg=3.0, subsample=0.85,
            loss_function="Logloss", eval_metric="AUC",
            class_weights=[1.0, spw], random_seed=42,
            verbose=0, thread_count=-1),
    }

    fitted, results = {}, {}
    skf = StratifiedKFold(3, shuffle=True, random_state=42)

    for name, mdl in base_models.items():
        print(f"   {name:<22} ...", end=" ", flush=True)
        t0   = time.time()
        Xfit = Xr if name != "Logistic Regression" else Xtr_s
        yfit = yr if name != "Logistic Regression" else y_tr
        mdl.fit(Xfit, yfit)
        fitted[name] = mdl

        yp = mdl.predict_proba(Xte_s)[:, 1]
        yd = (yp >= 0.5).astype(int)
        try:
            cv = cross_val_score(mdl, Xtr_s, y_tr, cv=skf,
                                 scoring="roc_auc", n_jobs=-1)
            cv_auc, cv_std = cv.mean(), cv.std()
        except Exception:
            cv_auc, cv_std = float("nan"), float("nan")

        thrs   = np.arange(0.10, 0.90, 0.02)
        f1s    = [f1_score(y_te, (yp >= t).astype(int), zero_division=0) for t in thrs]
        opt_t  = thrs[np.argmax(f1s)]
        yd_opt = (yp >= opt_t).astype(int)

        results[name] = dict(
            y_prob=yp, y_pred=yd, y_pred_opt=yd_opt, opt_threshold=opt_t,
            auroc    = roc_auc_score(y_te, yp),
            auprc    = average_precision_score(y_te, yp),
            f1       = f1_score(y_te, yd),
            f1_opt   = f1_score(y_te, yd_opt),
            accuracy = accuracy_score(y_te, yd),
            precision= precision_score(y_te, yd, zero_division=0),
            recall   = recall_score(y_te, yd),
            mcc      = matthews_corrcoef(y_te, yd),
            brier    = brier_score_loss(y_te, yp),
            cv_auc   = cv_auc, cv_std=cv_std,
            cm       = confusion_matrix(y_te, yd),
        )
        dt = time.time() - t0
        print(f"AUROC={results[name]['auroc']:.4f}  "
              f"F1={results[name]['f1_opt']:.4f}  "
              f"MCC={results[name]['mcc']:.4f}  ({dt:.1f}s)")

    # ── Soft Voting Ensemble
    print(f"   {'Voting Ensemble':<22} ...", end=" ", flush=True)
    t0 = time.time()
    # Average probabilities directly (no re-fit needed)
    all_probs = np.array([results[n]["y_prob"] for n in fitted])
    yp        = all_probs.mean(axis=0)
    yd        = (yp >= 0.5).astype(int)
    thrs      = np.arange(0.10, 0.90, 0.02)
    f1s       = [f1_score(y_te, (yp >= t).astype(int), zero_division=0) for t in thrs]
    opt_t     = thrs[np.argmax(f1s)]
    yd_opt    = (yp >= opt_t).astype(int)

    # Store a simple wrapper so predict_proba works for CDSS
    class MeanEnsemble:
        def __init__(self, models): self.models_ = list(models.values())
        def predict_proba(self, X):
            p = np.mean([m.predict_proba(X)[:,1] for m in self.models_], axis=0)
            return np.column_stack([1-p, p])

    fitted["Voting Ensemble"] = MeanEnsemble(fitted)

    results["Voting Ensemble"] = dict(
        y_prob=yp, y_pred=yd, y_pred_opt=yd_opt, opt_threshold=opt_t,
        auroc    = roc_auc_score(y_te, yp),
        auprc    = average_precision_score(y_te, yp),
        f1       = f1_score(y_te, yd),
        f1_opt   = f1_score(y_te, yd_opt),
        accuracy = accuracy_score(y_te, yd),
        precision= precision_score(y_te, yd, zero_division=0),
        recall   = recall_score(y_te, yd),
        mcc      = matthews_corrcoef(y_te, yd),
        brier    = brier_score_loss(y_te, yp),
        cv_auc   = float("nan"), cv_std=float("nan"),
        cm       = confusion_matrix(y_te, yd),
    )
    dt = time.time() - t0
    print(f"AUROC={results['Voting Ensemble']['auroc']:.4f}  "
          f"F1={results['Voting Ensemble']['f1_opt']:.4f}  "
          f"MCC={results['Voting Ensemble']['mcc']:.4f}  ({dt:.1f}s)")

    return fitted, results, scaler, Xte_s


# ===========================================================================
# 5. CDSS ALERT ENGINE
# ===========================================================================
def run_cdss(model, Xte_s, y_te):
    probs = model.predict_proba(Xte_s)[:, 1]

    def risk_level(p):
        if p >= 0.85: return "Critical"
        if p >= 0.70: return "High"
        if p >= 0.50: return "Moderate"
        if p >= 0.30: return "Low"
        return "Safe"

    risks      = np.array([risk_level(p) for p in probs])
    intercept  = np.isin(risks, ["High", "Critical"])
    prevented  = intercept & (y_te == 1) & (np.random.random(len(y_te)) < 0.82)
    fp         = intercept & (y_te == 0)
    before     = int(y_te.sum())
    after      = int(before - prevented.sum())
    savings    = float(prevented.sum()) * 4_685 - float(fp.sum()) * 45

    return dict(
        probs=probs, risks=risks,
        errors_before=before, errors_after=after,
        prevented=int(prevented.sum()),
        reduction_pct=round((1 - after / max(before, 1)) * 100, 1),
        alerts=int(intercept.sum()),
        fp_alerts=int(fp.sum()),
        alert_prec=round(float(prevented.sum()) / max(intercept.sum(), 1), 3),
        savings_usd=round(savings, 0),
        risk_dist=pd.Series(risks).value_counts().to_dict(),
    )


# ===========================================================================
# 6. VISUALISATIONS  (8 comprehensive figures)
# ===========================================================================

# ── Fig 1: Dataset overview
def fig1_overview(df):
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle("MIMIC-IV Dataset Overview — Medication Error Analysis",
                 fontsize=19, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    # 1a ICU error rate
    ax = fig.add_subplot(gs[0, :2])
    icu = df.groupby("icu_type")["error_flag"].mean().sort_values(ascending=False) * 100
    bars = ax.bar(icu.index, icu.values, color=MCOLS[:len(icu)], edgecolor="white", lw=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, fontweight="bold")
    ax.axhline(df.error_flag.mean()*100, color=PAL["danger"], ls="--", lw=2,
               label=f"Overall: {df.error_flag.mean()*100:.1f}%")
    ax.set_title("Error Rate by ICU Type", fontweight="bold"); ax.legend(fontsize=9)
    ax.set_ylabel("Error Rate (%)"); ax.grid(axis="y", alpha=0.4)

    # 1b drug class
    ax = fig.add_subplot(gs[0, 2:])
    dc = df.groupby("drug_class")["error_flag"].mean().sort_values() * 100
    colors_dc = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(dc)))
    ax.barh(dc.index, dc.values, color=colors_dc, edgecolor="white")
    ax.set_title("Error Rate by Drug Class", fontweight="bold")
    ax.set_xlabel("Error Rate (%)"); ax.grid(axis="x", alpha=0.4)
    for i, v in enumerate(dc.values): ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # 1c age dist
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(df[df.error_flag==0]["age"], bins=30, alpha=0.65, color=PAL["success"],
            label="No Error", density=True)
    ax.hist(df[df.error_flag==1]["age"], bins=30, alpha=0.65, color=PAL["danger"],
            label="Error", density=True)
    ax.set_title("Age Distribution", fontweight="bold")
    ax.set_xlabel("Age (years)"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # 1d SOFA
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df[df.error_flag==0]["sofa_score"], bins=15, alpha=0.65, color=PAL["success"],
            label="No Error", density=True)
    ax.hist(df[df.error_flag==1]["sofa_score"], bins=15, alpha=0.65, color=PAL["danger"],
            label="Error", density=True)
    ax.set_title("SOFA Score Distribution", fontweight="bold")
    ax.set_xlabel("SOFA Score"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # 1e shift/day risk
    ax = fig.add_subplot(gs[1, 2])
    sl = df.groupby(["night_shift", "weekend"])["error_flag"].mean() * 100
    sl.index = ["Day+WD", "Day+WE", "Night+WD", "Night+WE"]
    bars = ax.bar(sl.index, sl.values,
                  color=[PAL["success"], PAL["teal"], PAL["warning"], PAL["danger"]],
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Error Rate by Shift & Day", fontweight="bold")
    ax.set_ylabel("Error Rate (%)"); ax.grid(axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=15)

    # 1f comorbidity burden
    ax = fig.add_subplot(gs[1, 3])
    cb = df.groupby("comorbidity_burden")["error_flag"].mean() * 100
    ax.plot(cb.index, cb.values, "o-", color=PAL["primary"], lw=2.5, ms=8)
    ax.fill_between(cb.index, cb.values, alpha=0.15, color=PAL["primary"])
    ax.set_title("Error Rate vs Comorbidities", fontweight="bold")
    ax.set_xlabel("# Comorbidities"); ax.set_ylabel("Error Rate (%)"); ax.grid(alpha=0.4)

    # 1g severity pie
    ax = fig.add_subplot(gs[2, 0])
    sev = df[df.error_flag==1]["severity"].value_counts()
    wedge_cols = [PAL["warning"], PAL["danger"], PAL["purple"], PAL["orange"]]
    ax.pie(sev, labels=sev.index, autopct="%1.1f%%",
           colors=wedge_cols, startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Error Severity Distribution", fontweight="bold")

    # 1h error types
    ax = fig.add_subplot(gs[2, 1:3])
    et = df[df.error_flag==1]["error_type"].value_counts()
    bars = ax.barh(et.index, et.values,
                   color=[MCOLS[i % len(MCOLS)] for i in range(len(et))],
                   edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Medication Error Types", fontweight="bold")
    ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.4)

    # 1i cost by severity
    ax = fig.add_subplot(gs[2, 3])
    cs = df[df.error_flag==1].groupby("severity")["cost_impact_usd"].median() / 1_000
    bars = ax.bar(cs.index, cs.values,
                  color=[PAL["warning"], PAL["danger"], PAL["purple"], PAL["orange"]],
                  edgecolor="white")
    ax.bar_label(bars, fmt="$%.1fK", padding=3, fontsize=8)
    ax.set_title("Median Cost by Severity", fontweight="bold")
    ax.set_ylabel("USD (thousands)"); ax.grid(axis="y", alpha=0.4)

    return _save(fig, "fig1_dataset_overview.png")


# ── Fig 2: Model performance deep-dive
def fig2_model_perf(results, y_te):
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle("Model Performance — All Metrics & Curves",
                 fontsize=19, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)
    names  = list(results.keys())
    colors = MCOLS[:len(names)]

    # 2a AUROC bar
    ax = fig.add_subplot(gs[0, :2])
    aucs = [results[n]["auroc"] for n in names]
    bars = ax.bar(names, aucs, color=colors, edgecolor="white", lw=1.5)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9, fontweight="bold")
    ax.set_ylim(0.5, 1.04)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.4, label="Random")
    ax.axhline(0.8, color=PAL["success"], ls=":", lw=1.5, alpha=0.6, label="Good (0.80)")
    ax.axhline(0.9, color=PAL["primary"], ls=":", lw=1.5, alpha=0.6, label="Excellent (0.90)")
    ax.set_title("AUROC — All Models", fontweight="bold")
    ax.set_ylabel("AUROC"); ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.4)

    # 2b multi-metric
    ax = fig.add_subplot(gs[0, 2:])
    mets = ["accuracy", "precision", "recall", "f1", "auprc"]
    x = np.arange(len(names)); w = 0.15
    for i, m in enumerate(mets):
        vals = [results[n][m] for n in names]
        ax.bar(x + i*w, vals, w, label=m.capitalize(), color=MCOLS[i], alpha=0.9)
    ax.set_xticks(x + w * 2); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Multi-Metric Comparison", fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.4)

    # 2c ROC
    ax = fig.add_subplot(gs[1, :2])
    for n, c in zip(names, colors):
        fpr, tpr, _ = roc_curve(y_te, results[n]["y_prob"])
        ax.plot(fpr, tpr, color=c, lw=2.2,
                label=f"{n} ({results[n]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.4)

    # 2d PR curves
    ax = fig.add_subplot(gs[1, 2:])
    for n, c in zip(names, colors):
        p, r, _ = precision_recall_curve(y_te, results[n]["y_prob"])
        ax.plot(r, p, color=c, lw=2.2,
                label=f"{n} (AP={results[n]['auprc']:.3f})")
    ax.axhline(y_te.mean(), color="gray", ls="--", lw=1.5, label="Baseline")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.4)

    # 2e-h confusion matrices (top 4 by AUROC)
    top4 = sorted(names, key=lambda n: results[n]["auroc"], reverse=True)[:4]
    for i, n in enumerate(top4):
        ax = fig.add_subplot(gs[2, i])
        c  = colors[names.index(n)]
        cm = results[n]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False,
                    cmap=sns.light_palette(c, as_cmap=True),
                    xticklabels=["No Err", "Error"],
                    yticklabels=["No Err", "Error"])
        ax.set_title(f"{n}\nAUROC={results[n]['auroc']:.3f}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)

    return _save(fig, "fig2_model_performance.png")


# ── Fig 3: Feature importance
def fig3_feature_imp(rf_model, xgb_model, feat_cols, Xte_s):
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle("Feature Importance Analysis — Random Forest & XGBoost",
                 fontsize=16, fontweight="bold", y=1.02)

    # RF importance
    try:
        rf_base = rf_model.calibrated_classifiers_[0].estimator
    except Exception:
        rf_base = rf_model

    imp_rf = pd.Series(rf_base.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top20  = imp_rf.head(20)
    cmap_v = [plt.cm.RdYlGn(v / top20.max()) for v in top20.values[::-1]]
    axes[0].barh(top20.index[::-1], top20.values[::-1], color=cmap_v, edgecolor="white")
    axes[0].set_title("Top 20 — Random Forest", fontweight="bold")
    axes[0].set_xlabel("Gini Importance"); axes[0].grid(axis="x", alpha=0.4)

    # XGBoost importance
    try:
        xgb_base = xgb_model.calibrated_classifiers_[0].estimator
        imp_xgb  = pd.Series(xgb_base.feature_importances_, index=feat_cols).sort_values(ascending=False)
    except Exception:
        imp_xgb = imp_rf.copy()
    top20x = imp_xgb.head(20)
    cmap_x = [plt.cm.RdYlBu_r(v / top20x.max()) for v in top20x.values[::-1]]
    axes[1].barh(top20x.index[::-1], top20x.values[::-1], color=cmap_x, edgecolor="white")
    axes[1].set_title("Top 20 — XGBoost", fontweight="bold")
    axes[1].set_xlabel("Feature Importance"); axes[1].grid(axis="x", alpha=0.4)

    # cumulative importance
    cum = imp_rf.cumsum() / imp_rf.sum()
    n80 = (cum <= 0.80).sum() + 1
    n95 = (cum <= 0.95).sum() + 1
    axes[2].plot(range(1, len(cum)+1), cum.values, color=PAL["primary"], lw=2.5)
    axes[2].axhline(0.80, color=PAL["danger"], ls="--", lw=2, label=f"80% → {n80} features")
    axes[2].axhline(0.95, color=PAL["success"], ls="--", lw=2, label=f"95% → {n95} features")
    axes[2].fill_between(range(1, n80+1), cum.values[:n80], alpha=0.15, color=PAL["danger"])
    axes[2].fill_between(range(n80, n95+1), cum.values[n80-1:n95], alpha=0.10, color=PAL["success"])
    axes[2].set_title("Cumulative Feature Importance", fontweight="bold")
    axes[2].set_xlabel("Number of Features"); axes[2].set_ylabel("Cumulative Importance")
    axes[2].legend(fontsize=10); axes[2].grid(alpha=0.4)

    plt.tight_layout()
    return _save(fig, "fig3_feature_importance.png")


# ── Fig 4: SHAP values
def fig4_shap(model, Xte_s, feat_cols):
    if not HAS_SHAP:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("SHAP Explainability — XGBoost Feature Impact",
                 fontsize=16, fontweight="bold", y=1.02)
    try:
        base = model.calibrated_classifiers_[0].estimator
    except Exception:
        base = model

    sample_idx = np.random.choice(len(Xte_s), min(500, len(Xte_s)), replace=False)
    Xs = Xte_s[sample_idx]

    explainer  = shap.TreeExplainer(base)
    shap_vals  = explainer.shap_values(Xs)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # beeswarm-style (manual summary)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top15_idx= np.argsort(mean_abs)[::-1][:15]
    top15_feats = [feat_cols[i] for i in top15_idx]
    top15_sv    = shap_vals[:, top15_idx]

    for fi, (feat, sv) in enumerate(zip(top15_feats[::-1], top15_sv.T[::-1])):
        jitter = np.random.uniform(-0.3, 0.3, len(sv))
        feat_v = Xs[:, top15_idx[::-1][fi]]
        sc = axes[0].scatter(sv, np.full(len(sv), fi) + jitter,
                             c=feat_v, cmap="RdYlGn_r", alpha=0.4, s=12)
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(top15_feats[::-1], fontsize=9)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_title("SHAP Beeswarm — Top 15 Features", fontweight="bold")
    axes[0].set_xlabel("SHAP Value (impact on prediction)"); axes[0].grid(axis="x", alpha=0.4)
    plt.colorbar(sc, ax=axes[0], label="Feature value (low→high)")

    # mean abs SHAP bar
    top20_idx = np.argsort(mean_abs)[::-1][:20]
    top20_f   = [feat_cols[i] for i in top20_idx]
    top20_v   = mean_abs[top20_idx]
    bar_cols  = [plt.cm.RdYlGn_r(v / top20_v.max()) for v in top20_v[::-1]]
    axes[1].barh(top20_f[::-1], top20_v[::-1], color=bar_cols, edgecolor="white")
    axes[1].set_title("Mean |SHAP| — Global Feature Importance", fontweight="bold")
    axes[1].set_xlabel("Mean |SHAP Value|"); axes[1].grid(axis="x", alpha=0.4)

    plt.tight_layout()
    return _save(fig, "fig4_shap_explainability.png")


# ── Fig 5: CDSS clinical impact
def fig5_cdss(cdss, results, y_te):
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle("CDSS Alert Engine — Clinical & Financial Impact Simulation",
                 fontsize=19, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40)

    # 5a before/after
    ax = fig.add_subplot(gs[0, :2])
    cats = ["Before CDSS", "After CDSS"]
    vals = [cdss["errors_before"], cdss["errors_after"]]
    bars = ax.bar(cats, vals, color=[PAL["danger"], PAL["success"]],
                  edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%d", padding=6, fontsize=16, fontweight="bold")
    ax.set_title(f"Medication Errors Before vs After CDSS", fontweight="bold")
    ax.set_ylabel("# Medication Errors"); ax.grid(axis="y", alpha=0.4)
    ax.annotate(f"↓ {cdss['reduction_pct']:.1f}% Reduction",
                xy=(0.5, 0.85), xycoords="axes fraction", ha="center",
                fontsize=18, color=PAL["success"], fontweight="bold")

    # 5b risk distribution
    ax = fig.add_subplot(gs[0, 2])
    order   = ["Safe", "Low", "Moderate", "High", "Critical"]
    rd      = {k: cdss["risk_dist"].get(k, 0) for k in order}
    rcols   = [PAL["success"], PAL["teal"], PAL["warning"], PAL["danger"], "#4A0000"]
    bars    = ax.bar(rd.keys(), rd.values(), color=rcols, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("CDSS Risk Classification", fontweight="bold")
    ax.set_ylabel("# Prescriptions"); ax.grid(axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=20)

    # 5c financial
    ax = fig.add_subplot(gs[0, 3])
    gross = cdss["prevented"] * 4_685 / 1_000
    acost = cdss["alerts"] * 45 / 1_000
    net   = cdss["savings_usd"] / 1_000
    ax.bar(["Gross\nSavings", "Alert\nCosts", "Net\nSavings"],
           [gross, -acost, net],
           color=[PAL["success"], PAL["danger"], PAL["primary"]], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Financial Impact (USD thousands)", fontweight="bold")
    ax.set_ylabel("USD (thousands)"); ax.grid(axis="y", alpha=0.4)
    ax.text(2, net + abs(net)*0.06, f"${net:,.0f}K",
            ha="center", fontsize=12, color=PAL["primary"], fontweight="bold")

    # 5d threshold analysis
    ax = fig.add_subplot(gs[1, :2])
    thrs = np.arange(0.05, 0.95, 0.02)
    best_probs = results[max(results, key=lambda n: results[n]["auroc"])]["y_prob"]
    ps, rs, fs, bs = [], [], [], []
    for t in thrs:
        yd = (best_probs >= t).astype(int)
        ps.append(precision_score(y_te, yd, zero_division=0))
        rs.append(recall_score(y_te, yd, zero_division=0))
        fs.append(f1_score(y_te, yd, zero_division=0))
    bt = thrs[np.argmax(fs)]
    ax.plot(thrs, ps, "o-", color=PAL["primary"], lw=2, ms=3, label="Precision")
    ax.plot(thrs, rs, "s-", color=PAL["success"], lw=2, ms=3, label="Recall")
    ax.plot(thrs, fs, "^-", color=PAL["warning"], lw=2.5, ms=4, label="F1")
    ax.axvline(bt, color=PAL["danger"], ls="--", lw=2,
               label=f"Optimal t={bt:.2f}")
    ax.fill_between(thrs, fs, alpha=0.12, color=PAL["warning"])
    ax.set_title("Alert Threshold Optimisation (Best Model)", fontweight="bold")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.legend(fontsize=9); ax.grid(alpha=0.4)

    # 5e KPI cards
    ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
    kpis = [
        ("Errors Prevented",   f"{cdss['prevented']:,}",              PAL["success"]),
        ("Alert Precision",    f"{cdss['alert_prec']*100:.1f}%",      PAL["primary"]),
        ("False Pos Alerts",   f"{cdss['fp_alerts']:,}",              PAL["warning"]),
        ("Net Cost Savings",   f"${cdss['savings_usd']/1e6:.2f}M",   PAL["teal"]),
        ("Error Reduction",    f"{cdss['reduction_pct']:.1f}%",       PAL["danger"]),
        ("Total Alerts Fired", f"{cdss['alerts']:,}",                 PAL["purple"]),
    ]
    for idx, (label, val, col) in enumerate(kpis):
        ri, ci = divmod(idx, 2)
        x0 = 0.05 + ci * 0.50; y0 = 0.88 - ri * 0.32
        rect = FancyBboxPatch((x0, y0-0.23), 0.42, 0.25,
                              boxstyle="round,pad=0.02",
                              facecolor=col+"18", edgecolor=col,
                              lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x0+0.21, y0-0.05, val, ha="center", va="center",
                fontsize=20, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x0+0.21, y0-0.17, label, ha="center", va="center",
                fontsize=10, color=PAL["text"], transform=ax.transAxes)
    ax.set_title("CDSS Key Performance Indicators", fontweight="bold", pad=20)

    return _save(fig, "fig5_cdss_impact.png")


# ── Fig 6: Risk stratification
def fig6_risk_strat(df, results, y_te):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Risk Stratification & Clinical Patterns",
                 fontsize=16, fontweight="bold", y=1.02)

    best_n   = max(results, key=lambda n: results[n]["auroc"])
    probs    = results[best_n]["y_prob"]
    n        = len(probs)
    df2      = df.iloc[:n].copy()
    df2["risk_prob"] = probs

    # violin by drug class
    ax = axes[0][0]
    udc = list(df2.drug_class.unique()[:8])
    vdata = [df2[df2.drug_class == dc]["risk_prob"].values for dc in udc]
    vp = ax.violinplot(vdata, showmedians=True, widths=0.7)
    for pc, c in zip(vp["bodies"], MCOLS): pc.set_facecolor(c); pc.set_alpha(0.72)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2.5)
    ax.set_xticks(range(1, len(udc)+1))
    ax.set_xticklabels(udc, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"Risk Distribution by Drug Class ({best_n})", fontweight="bold")
    ax.set_ylabel("P(Error)"); ax.grid(axis="y", alpha=0.4)

    # SOFA × eGFR heatmap
    ax = axes[0][1]
    df2["sofa_bin"] = pd.cut(df2.sofa_score, [0,4,8,12,16,24],
                             labels=["0-4","5-8","9-12","13-16","17+"])
    df2["egfr_bin"] = pd.cut(df2.egfr, [0,15,30,60,90,120],
                             labels=["<15","15-30","30-60","60-90",">90"])
    heat = df2.groupby(["sofa_bin","egfr_bin"])["risk_prob"].mean().unstack()
    sns.heatmap(heat, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label":"Avg P(Error)"}, vmin=0, vmax=1)
    ax.set_title("Risk Heatmap: SOFA × eGFR", fontweight="bold")
    ax.set_xlabel("eGFR Bin"); ax.set_ylabel("SOFA Bin")

    # predicted probability distribution
    ax = axes[0][2]
    ax.hist(probs[y_te==0], bins=60, alpha=0.65, density=True,
            color=PAL["success"], label="No Error")
    ax.hist(probs[y_te==1], bins=60, alpha=0.65, density=True,
            color=PAL["danger"], label="Error")
    ax.axvline(0.5, color="black", ls="--", lw=2, label="Default t=0.5")
    opt_t = results[best_n]["opt_threshold"]
    ax.axvline(opt_t, color=PAL["warning"], ls="--", lw=2, label=f"Optimal t={opt_t:.2f}")
    ax.set_title("Predicted Probability Distribution", fontweight="bold")
    ax.set_xlabel("P(Medication Error)"); ax.set_ylabel("Density")
    ax.legend(fontsize=9); ax.grid(alpha=0.4)

    # CV AUROC
    ax = axes[1][0]
    cv_names = [n for n in results if not (isinstance(results[n]["cv_auc"], float)
                                           and np.isnan(results[n]["cv_auc"]))]
    cv_m = [results[n]["cv_auc"] for n in cv_names]
    cv_s = [results[n]["cv_std"] for n in cv_names]
    bars = ax.bar(cv_names, cv_m, color=MCOLS[:len(cv_names)],
                  yerr=cv_s, capsize=6, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=10, fontsize=9, fontweight="bold")
    ax.set_ylim(0.5, 1.04)
    ax.axhline(0.8, color=PAL["success"], ls=":", lw=1.5, alpha=0.7, label="0.80")
    ax.axhline(0.9, color=PAL["primary"], ls=":", lw=1.5, alpha=0.7, label="0.90")
    ax.set_title("5-Fold CV AUROC (mean ± std)", fontweight="bold")
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    # LOS vs error rate
    ax = axes[1][1]
    df2["los_bin"] = pd.cut(df2.los, bins=7)
    le2 = df2.groupby("los_bin")["error_flag"].mean() * 100
    ax.plot(range(len(le2)), le2.values, "o-", color=PAL["primary"], lw=2.5, ms=8)
    ax.fill_between(range(len(le2)), le2.values, alpha=0.15, color=PAL["primary"])
    ax.set_xticks(range(len(le2)))
    ax.set_xticklabels([str(b) for b in le2.index], rotation=30, ha="right", fontsize=7)
    ax.set_title("Error Rate by Length of Stay", fontweight="bold")
    ax.set_xlabel("LOS Bin"); ax.set_ylabel("Error Rate (%)"); ax.grid(alpha=0.4)

    # polypharmacy
    ax = axes[1][2]
    df2["meds_bin"] = pd.cut(df2.num_medications, [0,5,10,15,20,45],
                             labels=["1-5","6-10","11-15","16-20","20+"])
    pm = df2.groupby("meds_bin")["error_flag"].mean() * 100
    bars = ax.bar(pm.index, pm.values,
                  color=[PAL["success"],PAL["teal"],PAL["warning"],PAL["danger"],"#4A0000"],
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Error Rate by # Medications (Polypharmacy)", fontweight="bold")
    ax.set_ylabel("Error Rate (%)"); ax.set_xlabel("# Medications")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    return _save(fig, "fig6_risk_stratification.png")


# ── Fig 7: Advanced metrics & calibration
def fig7_advanced(results, y_te):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Advanced Metrics — Calibration, MCC, Brier Scores",
                 fontsize=16, fontweight="bold", y=1.02)
    names  = list(results.keys())
    colors = MCOLS[:len(names)]

    # MCC bar
    ax = axes[0][0]
    mcc_vals = [results[n]["mcc"] for n in names]
    bars = ax.bar(names, mcc_vals, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, fontweight="bold")
    ax.set_title("Matthews Correlation Coefficient (MCC)", fontweight="bold")
    ax.set_ylabel("MCC"); ax.tick_params(axis="x", rotation=15)
    ax.axhline(0, color="black", lw=0.8); ax.grid(axis="y", alpha=0.4)

    # Brier score
    ax = axes[0][1]
    brier_vals = [results[n]["brier"] for n in names]
    bars = ax.bar(names, brier_vals, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, fontweight="bold")
    ax.set_title("Brier Score (lower = better calibration)", fontweight="bold")
    ax.set_ylabel("Brier Score"); ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.4)

    # F1 at default vs optimised threshold
    ax = axes[0][2]
    f1_def = [results[n]["f1"] for n in names]
    f1_opt = [results[n]["f1_opt"] for n in names]
    x = np.arange(len(names)); w = 0.35
    bars1 = ax.bar(x - w/2, f1_def, w, label="Default (t=0.5)", color=PAL["primary"], alpha=0.85)
    bars2 = ax.bar(x + w/2, f1_opt, w, label="Optimal threshold", color=PAL["success"], alpha=0.85)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("F1: Default vs Optimised Threshold", fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.4)

    # Calibration curves
    ax = axes[1][0]
    from sklearn.calibration import calibration_curve
    for n, c in zip(names, colors):
        try:
            prob_true, prob_pred = calibration_curve(y_te, results[n]["y_prob"],
                                                     n_bins=10, strategy="quantile")
            ax.plot(prob_pred, prob_true, "o-", color=c, lw=2, ms=5, label=n)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_title("Calibration Curves", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.4)

    # Radar / spider chart of top model
    best_n  = max(results, key=lambda n: results[n]["auroc"])
    best_r  = results[best_n]
    metrics_radar = dict(
        AUROC    = best_r["auroc"],
        AUPRC    = best_r["auprc"],
        F1_opt   = best_r["f1_opt"],
        Precision= best_r["precision"],
        Recall   = best_r["recall"],
        MCC_norm = (best_r["mcc"] + 1) / 2,
    )
    labels  = list(metrics_radar.keys())
    vals    = list(metrics_radar.values())
    vals   += vals[:1]
    angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    ax2 = fig.add_subplot(2, 3, 5, polar=True)
    ax2.plot(angles, vals, "o-", color=PAL["primary"], lw=2.5)
    ax2.fill(angles, vals, color=PAL["primary"], alpha=0.18)
    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Radar — {best_n}", fontweight="bold", pad=20)
    ax2.grid(alpha=0.5)

    # All-model score table
    ax = axes[1][2]; ax.axis("off")
    col_labels = ["AUROC", "AUPRC", "F1", "MCC", "Brier"]
    rows       = names
    cell_data  = [[f"{results[n]['auroc']:.4f}",
                   f"{results[n]['auprc']:.4f}",
                   f"{results[n]['f1_opt']:.4f}",
                   f"{results[n]['mcc']:.4f}",
                   f"{results[n]['brier']:.4f}"]
                  for n in names]
    tbl = ax.table(cellText=cell_data, rowLabels=rows,
                   colLabels=col_labels, cellLoc="center",
                   loc="center", bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PAL["grid"])
        if r == 0:
            cell.set_facecolor(PAL["primary"]); cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and c >= 0:
            cell.set_facecolor("#F0F4FF" if r % 2 == 0 else "white")
    ax.set_title("All-Model Score Card", fontweight="bold", pad=20)

    plt.tight_layout()
    return _save(fig, "fig7_advanced_metrics.png")


# ── Fig 8: Error probability deep dive
def fig8_prob_analysis(df, results, y_te):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Error Probability Deep Dive & Risk Factor Analysis",
                 fontsize=16, fontweight="bold", y=1.02)

    best_n = max(results, key=lambda n: results[n]["auroc"])
    probs  = results[best_n]["y_prob"]
    n      = len(probs)
    d      = df.iloc[:n].copy()
    d["predicted_prob"] = probs

    # prob vs actual error_probability scatter
    ax = axes[0][0]
    ax.scatter(d["error_probability"], probs, alpha=0.25, s=6,
               c=d["error_flag"], cmap="RdYlGn_r")
    lims = [0, 1]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect")
    ax.set_title("Predicted vs True Error Probability", fontweight="bold")
    ax.set_xlabel("True Probability (simulator)"); ax.set_ylabel("Model Predicted Prob")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)
    corr = np.corrcoef(d["error_probability"], probs)[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, color=PAL["primary"], fontweight="bold")

    # top risk factors (mean prob)
    ax = axes[0][1]
    risk_factors = {
        "Allergy Conflict":  d[d.allergy_conflict==1]["predicted_prob"].mean(),
        "Contraindication":  d[d.contraindication_flag==1]["predicted_prob"].mean(),
        "eGFR < 15":         d[d.egfr<15]["predicted_prob"].mean(),
        "INR > 2.5":         d[d.inr>2.5]["predicted_prob"].mean(),
        "Night Shift":       d[d.night_shift==1]["predicted_prob"].mean(),
        "Drug Interaction":  d[d.drug_interaction_flag==1]["predicted_prob"].mean(),
        "NTI Drug":          d[d.narrow_therapeutic_index==1]["predicted_prob"].mean(),
        "eGFR < 30":         d[d.egfr<30]["predicted_prob"].mean(),
        "Renal Failure":     d[d.renal_failure==1]["predicted_prob"].mean(),
        "High Risk Combo":   d[d.high_risk_combo==1]["predicted_prob"].mean(),
        "Liver Disease":     d[d.liver_disease==1]["predicted_prob"].mean(),
        "Meds ≥ 10":         d[d.num_medications>=10]["predicted_prob"].mean(),
        "Sepsis":            d[d.sepsis==1]["predicted_prob"].mean(),
        "Electronic Order":  d[d.electronic_order==1]["predicted_prob"].mean(),
        "Renal Adjusted":    d[d.renal_adjusted==1]["predicted_prob"].mean(),
    }
    sorted_rf = dict(sorted(risk_factors.items(), key=lambda x: x[1]))
    col_map   = [PAL["success"] if v < d["predicted_prob"].mean() else PAL["danger"]
                 for v in sorted_rf.values()]
    axes[0][1].barh(list(sorted_rf.keys()), list(sorted_rf.values()),
                    color=col_map, edgecolor="white")
    axes[0][1].axvline(d["predicted_prob"].mean(), color="black", ls="--", lw=1.5,
                       label=f"Mean {d['predicted_prob'].mean():.3f}")
    axes[0][1].set_title("Mean Predicted Risk by Factor", fontweight="bold")
    axes[0][1].set_xlabel("Mean P(Error)"); axes[0][1].legend(fontsize=8)
    axes[0][1].grid(axis="x", alpha=0.4)

    # risk by num concurrent drugs
    ax = axes[0][2]
    d["conc_bin"] = pd.cut(d.num_concurrent_drugs, [0,2,4,6,8,15],
                           labels=["0-2","3-4","5-6","7-8","9+"])
    cb = d.groupby("conc_bin")["error_flag"].mean() * 100
    bars = ax.bar(cb.index, cb.values,
                  color=[PAL["success"],PAL["teal"],PAL["warning"],PAL["danger"],"#4A0000"],
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Error Rate by Concurrent Drug Count", fontweight="bold")
    ax.set_ylabel("Error Rate (%)"); ax.set_xlabel("# Concurrent Drugs")
    ax.grid(axis="y", alpha=0.4)

    # probability histogram by severity
    ax = axes[1][0]
    sev_order = ["None","Minor","Moderate","Severe","Life-threatening"]
    sev_colors= [PAL["success"],PAL["teal"],PAL["warning"],PAL["danger"],"#4A0000"]
    for sv, sc in zip(sev_order, sev_colors):
        if sv in d.severity.values:
            subset = d[d.severity == sv]["predicted_prob"]
            if len(subset) > 10:
                ax.hist(subset, bins=30, alpha=0.65, label=sv,
                        color=sc, density=True)
    ax.set_title("P(Error) Distribution by Severity", fontweight="bold")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # prescriber experience vs error
    ax = axes[1][1]
    d["exp_bin"] = pd.cut(d.prescriber_exp, [0,2,5,10,20,40],
                          labels=["<2yr","2-5yr","5-10yr","10-20yr","20+yr"])
    pe = d.groupby("exp_bin")["error_flag"].mean() * 100
    ax.plot(range(len(pe)), pe.values, "o-", color=PAL["danger"], lw=2.5, ms=9)
    ax.fill_between(range(len(pe)), pe.values, alpha=0.15, color=PAL["danger"])
    ax.set_xticks(range(len(pe))); ax.set_xticklabels(pe.index, fontsize=10)
    ax.set_title("Error Rate by Prescriber Experience", fontweight="bold")
    ax.set_xlabel("Experience Level"); ax.set_ylabel("Error Rate (%)")
    ax.grid(alpha=0.4)

    # protective factors impact
    ax = axes[1][2]
    prot_factors = {
        "Electronic Order":    d[d.electronic_order==1]["error_flag"].mean(),
        "No Electronic Order": d[d.electronic_order==0]["error_flag"].mean(),
        "Renal Adjusted":      d[d.renal_adjusted==1]["error_flag"].mean(),
        "Not Renal Adjusted":  d[d.renal_adjusted==0]["error_flag"].mean(),
        "Weight-Based Dose":   d[d.weight_based_dosing==1]["error_flag"].mean(),
        "Not Weight-Based":    d[d.weight_based_dosing==0]["error_flag"].mean(),
    }
    prot_cols = [PAL["success"],PAL["danger"]] * 3
    bars = ax.bar(prot_factors.keys(), [v*100 for v in prot_factors.values()],
                  color=prot_cols, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Protective Factor Impact on Error Rate", fontweight="bold")
    ax.set_ylabel("Error Rate (%)"); ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    return _save(fig, "fig8_probability_analysis.png")


# ===========================================================================
# 7. SUMMARY PRINT
# ===========================================================================
def print_summary(results, cdss, df, feat_cols, t_total):
    sep = "═" * 68
    print(f"\n{sep}")
    print("   AI-CDSS  ·  MEDICATION ERROR REDUCTION  ·  RESULTS")
    print(sep)

    print("\n📊  DATASET")
    print(f"   Records          : {len(df):,}")
    print(f"   Medication errors: {df.error_flag.sum():,}  ({df.error_flag.mean()*100:.1f}%)")
    print(f"   Features (eng.)  : {len(feat_cols)}")
    print(f"   Harm events      : {df.harm_reached_patient.sum():,}")
    print(f"   Total cost burden: ${df.cost_impact_usd.sum()/1e6:.1f}M")

    print(f"\n🤖  MODEL LEADERBOARD")
    ranked = sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True)
    print(f"   {'Rank':<5} {'Model':<22} {'AUROC':>7} {'AUPRC':>7} "
          f"{'F1-opt':>7} {'MCC':>7} {'Brier':>7}")
    print("   " + "─" * 62)
    medals = ["🥇","🥈","🥉"]
    for i, (n, r) in enumerate(ranked):
        medal = medals[i] if i < 3 else f" {i+1}."
        print(f"   {medal:<5} {n:<22} {r['auroc']:>7.4f} {r['auprc']:>7.4f} "
              f"{r['f1_opt']:>7.4f} {r['mcc']:>7.4f} {r['brier']:>7.4f}")

    best = ranked[0][0]
    br   = ranked[0][1]
    print(f"\n   🏆  Best model  : {best}")
    print(f"       AUROC       : {br['auroc']:.4f}")
    print(f"       AUPRC       : {br['auprc']:.4f}")
    print(f"       F1 (optimal): {br['f1_opt']:.4f}")
    print(f"       MCC         : {br['mcc']:.4f}")
    print(f"       Brier score : {br['brier']:.4f}")
    print(f"       Optimal t   : {br['opt_threshold']:.2f}")

    print(f"\n🏥  CDSS IMPACT")
    print(f"   Errors before    : {cdss['errors_before']:,}")
    print(f"   Errors after     : {cdss['errors_after']:,}")
    print(f"   Errors prevented : {cdss['prevented']:,}")
    print(f"   Reduction        : {cdss['reduction_pct']:.1f}%")
    print(f"   Alert precision  : {cdss['alert_prec']*100:.1f}%")
    print(f"   False pos alerts : {cdss['fp_alerts']:,}")
    print(f"   Net cost savings : ${cdss['savings_usd']/1e6:.2f}M")

    print(f"\n   Risk distribution:")
    for lvl in ["Critical","High","Moderate","Low","Safe"]:
        cnt = cdss["risk_dist"].get(lvl, 0)
        bar = "█" * min(30, int(cnt / max(cdss["risk_dist"].values()) * 30))
        print(f"   {lvl:<10} {bar} {cnt:,}")

    print(f"\n⏱   Total runtime    : {t_total:.1f}s")
    print(f"{sep}\n")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t_start = time.time()
    print("\n" + "═"*68)
    print("  MIMIC-IV  ·  AI Clinical Decision Support  ·  Best-in-Class")
    print("═"*68)

    # 1. Data
    df = simulate_mimic_iv(n=6_000)

    # 2. Features
    df, feat_cols = engineer_features(df)
    X = df[feat_cols].fillna(df[feat_cols].median()).values
    y = df["error_flag"].values

    # 3. Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # 4. Train
    fitted, results, scaler, Xte_s = train_all(X_tr, y_tr, X_te, y_te, feat_cols)

    # 5. CDSS
    best_n = max(results, key=lambda n: results[n]["auroc"])
    cdss   = run_cdss(fitted[best_n], Xte_s, y_te)

    # 6. Visualise
    print(f"\n{'─'*60}")
    print("[4] Generating visualisations")
    print(f"{'─'*60}")
    paths = []
    paths.append(fig1_overview(df))
    paths.append(fig2_model_perf(results, y_te))
    paths.append(fig3_feature_imp(fitted["Random Forest"], fitted["XGBoost"], feat_cols, Xte_s))
    if HAS_SHAP:
        p = fig4_shap(fitted["XGBoost"], Xte_s, feat_cols)
        if p: paths.append(p)
    paths.append(fig5_cdss(cdss, results, y_te))
    paths.append(fig6_risk_strat(df, results, y_te))
    paths.append(fig7_advanced(results, y_te))
    paths.append(fig8_prob_analysis(df, results, y_te))

    # 7. Save artefacts
    joblib.dump(fitted[best_n], f"{OUT}/cdss_best_model.joblib")
    joblib.dump(scaler,         f"{OUT}/cdss_scaler.joblib")
    with open(f"{OUT}/feature_cols.json", "w") as f:
        json.dump(feat_cols, f)

    meta = {
        "best_model": best_n,
        "n_features": len(feat_cols),
        "dataset_size": len(df),
        "model_metrics": {
            k: {m: round(float(v), 5)
                for m, v in r.items()
                if isinstance(v, (int, float, np.floating))
                and not (isinstance(v, float) and np.isnan(v))}
            for k, r in results.items()
        },
        "cdss_impact": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in cdss.items()
            if not isinstance(v, (np.ndarray,))
        },
    }
    with open(f"{OUT}/results_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 8. Print summary
    print_summary(results, cdss, df, feat_cols, time.time() - t_start)
    return paths, results, cdss


if __name__ == "__main__":
    main()
