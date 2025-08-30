# ChemSolutions-AI-QC
Реализация ML-системы контроля качества фармацевтических веществ, включая предметную аналитику оборудования и мониторинг чистоты партий с помощью PAT-инструментов

Ниже готовый Jupyter Notebook, полностью self-contained, который можно просто скопировать в файл .ipynb и загрузить на GitHub. Он включает: проверку данных, синтетику, обучение XGBoost, SHAP-анализ и демонстрацию «что если» для рекомендаций.
Механизм А 

---
# ChemSolutions AI-QC Prototype

## 📖 Описание
Данный прототип создан для демонстрации концепции **AI-Driven Quality Control** в фармацевтическом производстве.  
Система использует **машинное обучение** и **инструменты PAT (Process Analytical Technology)** для:
- Мониторинга критических параметров процессов (CPP)
- Прогнозирования качества партий
- Снижения риска брака и простоев оборудования
- Повышения стабильности качества продукции

### Основные модули:
1. **Контур 1 — Контроль качества сырья и промежуточных соединений**  
   Прогноз вероятности брака ещё до начала синтеза.
2. **Контур 2 — Мониторинг критических стадий синтеза**  
   Онлайн-анализ (pH, температура, NIR, Raman) для предотвращения ошибок.
3. **Контур 3 — Прогноз качества готового продукта**  
   Итоговый контроль партии и предиктивная аналитика выхода.

---
# SmartQC — AI для предсказания качества и обнаружения аномалий

## 🚀 Как запустить
1. Скопируйте любой из кодовых блоков ниже в Jupyter Notebook или Google Colab.
2. Запустите ячейки по порядку.
3. Результат: графики SHAP, ROC AUC и рекомендации для операторов.

---

## 🔹 Контур A: Предсказание качества (Soft Sensors)

```python
pip install xgboost shap scikit-learn pandas matplotlib

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# --- Проверка наличия данных ---
if os.path.exists("batches_aggregated.csv"):
    df = pd.read_csv("batches_aggregated.csv")
else:
    # создаём синтетический датасет
    np.random.seed(42)
    n = 800
    df = pd.DataFrame({
        "temp_mean": np.random.normal(72,3,n),
        "pH_mean": np.random.normal(7.2,0.4,n),
        "pressure_mean": np.random.normal(1.2, 0.15,n),
        "dosing_var": np.random.normal(0.02, 0.01, n),
        "torque_max": np.random.normal(40,5,n),
    })
    df["purity"] = 100 - abs(df["temp_mean"]-72)*0.8 - abs(df["pH_mean"]-7.2)*5 - df["dosing_var"]*100
    df["label_good"] = (df["purity"] >= 98).astype(int)

print("Данные загружены / сгенерированы:")
print(df.head())

# --- Разделение train/test (time-aware) ---
train = df.iloc[:640]
test = df.iloc[640:]

features = ["temp_mean","pH_mean","pressure_mean","dosing_var","torque_max"]

# --- Обучение XGBoost ---
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model.fit(train[features], train["label_good"])
probs = model.predict_proba(test[features])[:,1]

print("\nROC AUC на тесте:", roc_auc_score(test["label_good"], probs))

# --- SHAP Explainability ---
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(test[features])
shap.summary_plot(shap_vals, test[features], show=False)
plt.savefig("shap_summary.png")  # сохраняем график для презентации
plt.close()

# --- Простая recommendation rule (локальная симуляция) ---
i = test.index[-1]
row = test.loc[i].copy()
base_prob = model.predict_proba(row[features].values.reshape(1,-1))[0,1]

suggestions = []
for feat in ["temp_mean","pH_mean","dosing_var"]:
    modified = row[features].copy()
    if feat == "temp_mean":
        modified["temp_mean"] -= 1.0
    elif feat == "pH_mean":
        modified["pH_mean"] += 0.1
    elif feat == "dosing_var":
        modified["dosing_var"] *= 0.95
    new_prob = model.predict_proba(modified.values.reshape(1,-1))[0,1]
    suggestions.append((feat, base_prob, new_prob, new_prob-base_prob))

print("\nBase prob:", round(base_prob,3))
print("Рекомендации (feature, base_prob, new_prob, delta):")
for s in sorted(suggestions, key=lambda x: -x[3]):
    print(s)

# --- Визуализация predicted vs real purity ---
plt.scatter(test["purity"], probs*100)
plt.xlabel("Real purity (%)")
plt.ylabel("Predicted P(good) * 100")
plt.title("Predicted vs Real Purity")
plt.savefig("pred_vs_real.png")
plt.show() 












∆∆∆∆Контур B — Обнаружение аномалий (Anomaly Detection / SPC 2.0) — полный, практичный разбор

Коротко: цель — в реальном времени обнаруживать отклонения в процессе (датчики, спектры, поведение реактора), которые предшествуют браку/неправильной чистоте партии или аварии оборудования, и давать раннее, интерпретируемое предупреждение оператору.

Ниже — подробная «инструкция до винтика»: что вы получите, какие данные нужны, как моделить/валидация/деплой, какие метрики смотреть, как это влияет на бизнес и на критерии жюри. В конце — готовый рабочий скрипт (проверяет наличие файла, иначе генерит синтетику)
---

1) Что выдаёт контур B — конкретные артефакты

Аномалийный скор (анти-«норма»): число на каждом временном шаге.

Бинарный флаг аномалии (multi-tier: предупреждение/критично).

Время обнаружения (lead time до лабораторного OOS / отказа).

Причинная вкладка: какие сенсоры дали наибольший вклад в рекомбинационную ошибку / скор (reconstruction error per sensor, top-k features).

Алерт-пакет: временной участок, скоры, рекомендуемое действие (verify sampler / take corrective action), уникальный id инцидента для трекинга.

Дашборд-виджеты: timeline с отметками аномалий; распределение скорoв; recent incidents list.



---

2) Какие данные нужны (и как их получить)

SCADA / DCS (онлайн): T, pH, P, flow rates, torque, power, насосные расход/скорость — частота 1–60 s.

Спектры (если есть): NIR / Raman / FT-IR (высокая частота или периодические inline-сканы).

MES / Batch metadata: batch_id, recipe_id, operator, lot сырья.

LIMS / QC labels: HPLC/GC/NMR результаты (для ретроспективной супервайз-валидации).

Логи оборудования: vibration, temp bearings, current.

История ремонтов/инцидентов — для метрик PdM.


Если inline спектры отсутствуют — всё равно работает многоуровневый мониторинг по физическим сенсорам.


---

3) Preprocessing — критически важно

Синхронизация: привести все сигналы к общему тайм-индексу; биннинг/ресемплинг (например 5s/10s).

Детрендинг: убрать сезонность/длинные тренды, чтобы не взрывать false positives.

Фильтрация шума: медианная/скользящая фильтра или low-pass для вибраций/шумных каналов.

Нормализация: per-sensor scaling (robust: median/IQR) — важна для multivariate detection.

Заполнение пропусков: mark missing с флагом, не просто ффилл — пропуски сами могут быть сигналом.

Windowing: скользящие окна (например 30–300 s) и агрегации (mean/std/slope/min/max/percentiles).

Feature drift monitoring: сохранять baseline distribution и проверять drift (KS / population stability index).



---

4) Feature engineering — что реально работает

Time-domain aggregates: mean/std/median/last/slope/area_under_curve per window.

Derivatives & rates: dT/dt, d(pH)/dt, relative change in flow.

Cross-features: difference (temp - coolant_temp), ratio (feedA/feedB).

Spectrum->features: PCA scores, select band integrals, wavelet coefficients.

Frequency features (для вибраций): FFT peaks, spectral centroid, band energy.

Latent features: bottleneck output из автоэнкодера / PCA — сами становятся признаками.

Time-series embeddings: sliding-window flattened vectors, or sequence models (LSTM embedding).



---

5) Алгоритмы — выбор и обоснование (с плюсами/минусами)

Ненадзорные / полунадзорные (обычно)

Isolation Forest — быстрый, масштабируемый, хорошо для табличных агрегатов. (++ низкая сложность, — чувствителен к настройке contamination)

LOF / KNN-based — локальная плотность, хорошо на «локальных» аномалиях.

One-Class SVM — работает при малых данных норм, но медленный.

Autoencoder (Dense/LSTM/Conv1D) — reconstruction error per-feature; хорошо для multivariate temporal anomalies. (++ гибкий, может выдавать per-feature ошибки, — требует тюнинга + данные)

Variational Autoencoder / LSTM-AE — устойчивее к шуму, но сложнее.

Multivariate SPC / MPCA / Hotelling T2 — классика для промышленности (легко объяснить аудитории).

Change-point detection (ruptures, Bayesian) — ловит структурные сдвиги (drift, step-change).

Ensemble / voting: сочетание нескольких методов снижает FP.


Стриминг/онлайн

River (ранее creme) для онлайн-обучения / адаптации.

EWMA / CUSUM — быстрые простые детекторы на одном сигнале.


Как выбирать

Если есть лейблы (historic OOS/отказы) — можно перейти к supervised (XGBoost на предиктах/фичах), либо train isolation with labels for thresholding.

Без лейблов — начните с IsolationForest + MPCA + простых change-point. Добавляйте AE, если хотите поймать сложные multivariate паттерны.



---

6) Валидация и метрики (реально важные)

Аномалия — class imbalance. Нужны специализированные метрики:

Precision@k: важен, т.к. операторы не должны тонуть в ложных тревогах.

Recall / Recall@lead_time: насколько быстро модель замечает предвестники OOS.

F1 (balanced view).

PR-AUC вместо ROC-AUC при сильном дисбалансе.

False Alarm Rate (FAR) per day/shift — бизнес-показатель.

Mean Time To Detect (MTTD) / median lead time (в минутах/часах до OOS).

Detection Delay distribution (смещение ранних срабатываний).

Stability metrics: Frequency of alerts per unit времени (удерживать в приемлемых пределах).


Валидация-подходы:

Если лейблы есть — сделать time-aware split (train on older batches).

Если лейблов нет — inject synthetic anomalies (контрастные сценарии), run backtest on historical normal data, measure FP rate.

Shadow mode: 2–4 недели — система пишет предупреждения, операторы не вмешиваются; потом сравнивать предсказания с результатами QC.



---

7) Thresholding и сигнал-треугольник

Динамический порог: percentile-based (например, score > 99.5-percentile of last 24h) или EWMA adapt.

Три уровня тревог: INFO / WARNING / CRITICAL, с разными SOP: INFO — мониторить, WARNING — взять пробу, CRITICAL — остановить/корректировать.

Consensus rule: требовать совпадения 2/3 моделей (IsolationForest + AE + MPCA) для CRITICAL — уменьшает FP.

Human-in-loop: каждый алерт логируется, оператор подтверждает/отклоняет — это метки для будущей supervised-обучалки.



---

8) Explainability / Root cause

Reconstruction error per sensor (для AE) — самый прямой канал: «вклад» по переменной.

Feature-contribution via SHAP — применимо если используете supervised detector (или pseudo-labels).

Correlation-break detection: до/после сравнение корр. матриц — показывает, какие связи разорвались.

Nearest-anomalies search: найти исторические похожие случаи с известной причиной.

Автоматический пакет для оператора: top-3 sensors, time window, suggested action.


Объяснимость — ключ для одобрения регулятора и доверия операторов.


---

9) Deployment (практичная архитектура)

Ingest SCADA → Kafka / MQTT.

Preprocessing (stream processor): Spark Streaming / Flink / lightweight service.

Scoring: контейнер FastAPI (models in Torch/Sklearn) → produce anomaly score to Kafka topic.

Alerting: rule engine (Grafana alerts / custom) → send to Slack/SMS/SCADA alarm.

Storage: TSDB (Influx/Grafana) для визуализаций + archival in S3 for offline retraining.

Model ops: MLflow for model registry; scheduled retrain (weekly) with data drift checks.

Monitoring: Prometheus (latency), Grafana (scores, number of alerts), data-quality (Great Expectations).



---

10) Риски и митигейты

False positives → ensemble + dynamic thresholds + operator feedback loop.

Data drift → drift detection triggers retrain + retraining pipeline.

Sensor failures mistaken for anomalies → redundancy checks, sensor health channel.

Operator distrust → shadow mode + обучение + explainability tiles.

Regulatory trace → audit trails (why alert, model version, input values).



---

11) ROI & бизнес-эффект (кратко, цифры для слайда)

Уменьшение OOS/брака на 30% (контур A+B в связке) — экономия десятки % от текущих потерь.

Предотвращение аварий/незапланированных простоев (PdM интегр.) — экономия $3–5M/год (оценочно, зависит от масштаба).

Быстрая локализация проблемы (root-cause) — сокращение времени реагирования на 50% → уменьшение переработок.
(Для слайда: укажите реальные числа компании + приведите conservative estimate; используйте расчёт, как в Contour A ROI-слайде.)



---

12) MVP для хакатона (что нужно предоставить жюри)

1. Скрипт / ноутбук: загружает data_B.csv (если есть) или генерит синтетику.


2. Модель IsolationForest (обязательно) + опция Autoencoder.


3. Визуализации:

timeline: sensor(s) + anomaly flag (цветом)

histogram anomaly scores + chosen threshold

table: найденные incidents (start, end, top-3 contributing sensors)

metrics: precision/recall (если есть labels) / expected lead time



4. README: как запустить, объяснение что показывает.


5. Код в отдельном файле contour_b.py + PNG скринов для презентации.




---

13) Как это повышает оценку по критериям жюри

Актуальность: вы ловите реальные предвестники брака — прямое уменьшение OOS.

Инновации: сочетание multivariate AE + MPCA + streaming detection + explainability.

Бизнес: быстродействие → быстрый ROI, меньше отходов, меньше простоев.

Технический вклад: полноценный pipeline, демонстрируемый в notebook/репо.

Презентация: четкие графики/метрики — легко понять.



---




---

15) Готовый рабочий код (вставь в contour_b.py)

Этот скрипт: пытается загрузить data_B.csv, иначе генерит реалистичную синтетику, обучает IsolationForest, рисует графики и (если метки есть) считает precision/recall. Можно залить в GitHub как MVP.

# contour_b.py
# Python 3.8+
# pip install numpy pandas matplotlib scikit-learn seaborn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

sns.set(style="whitegrid")

# --- 1. Load or generate data ---
if os.path.exists("data_B.csv"):
    df = pd.read_csv("data_B.csv", parse_dates=["timestamp"])
    # Expect columns: timestamp,temp,pH,flow,torque,vibration,label (optional 0/1)
else:
    np.random.seed(42)
    T = 2000
    time = pd.date_range("2025-01-01", periods=T, freq="T")  # 1-minute steps
    temp = 72 + np.random.normal(0, 0.3, T).cumsum()*0.001 + np.sin(np.linspace(0, 20, T))*0.2
    pH = 7.2 + np.random.normal(0, 0.02, T)
    flow = 1.0 + np.random.normal(0, 0.02, T)
    torque = 40 + np.random.normal(0, 0.5, T)
    vibration = np.random.normal(0, 0.2, T)

    # inject anomalies: batches with drift + spikes
    labels = np.zeros(T, dtype=int)
    for start in [600, 1200, 1600]:
        idx = np.arange(start, start+30)
        temp[idx] += np.linspace(0.5, 2.0, len(idx))  # drift up
        pH[idx] += np.random.normal(0.2, 0.05, len(idx))
        flow[idx] *= 0.8
        torque[idx] += np.random.normal(3.0, 0.5, len(idx))
        vibration[idx] += np.random.normal(1.5, 0.4, len(idx))
        labels[idx] = 1

    df = pd.DataFrame({
        "timestamp": time,
        "temp": temp,
        "pH": pH,
        "flow": flow,
        "torque": torque,
        "vibration": vibration,
        "label": labels
    })

# --- 2. Feature engineering: sliding window aggregates (window=5) ---
df = df.sort_values("timestamp").reset_index(drop=True)
window = 5
agg = pd.DataFrame()
for col in ["temp", "pH", "flow", "torque", "vibration"]:
    agg[f"{col}_mean"] = df[col].rolling(window, min_periods=1).mean()
    agg[f"{col}_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)
    agg[f"{col}_last"] = df[col].shift(0)
    agg[f"{col}_slope"] = df[col].diff().rolling(window, min_periods=1).mean().fillna(0)

agg = agg.fillna(method="bfill").fillna(0)

# --- 3. Train IsolationForest on first 70% (simulate time-aware) ---
split_idx = int(0.7 * len(agg))
X_train = agg.iloc[:split_idx].values
X_test  = agg.iloc[split_idx:].values

clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
clf.fit(X_train)

scores = -clf.decision_function(agg.values)  # higher = more anomalous
threshold = np.percentile(scores[:split_idx], 98)  # dynamic threshold based on train

anomaly_flag = (scores > threshold).astype(int)

df["anomaly_score"] = scores
df["anomaly_flag"] = anomaly_flag

# --- 4. Metrics (if label exists) ---
if "label" in df.columns:
    # align labels (test period only) - evaluate on the test segment
    true = df["label"].values[split_idx:]
    pred = df["anomaly_flag"].values[split_idx:]
    prec = precision_score(true, pred, zero_division=0)
    rec = recall_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred, zero_division=0)
    print(f"Evaluation on test period: Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

# --- 5. Visualizations ---
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["temp"], label="temp")
plt.scatter(df["timestamp"][df["anomaly_flag"]==1], df["temp"][df["anomaly_flag"]==1],
            color="red", s=20, label="anomaly")
plt.title("Temp timeline with anomalies")
plt.legend()
plt.tight_layout()
plt.savefig("contourB_temp_timeline.png")
plt.show()

plt.figure(figsize=(14,3))
plt.plot(df["timestamp"], df["anomaly_score"], label="anomaly_score")
plt.axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
plt.title("Anomaly score timeline")
plt.legend()
plt.tight_layout()
plt.savefig("contourB_score_timeline.png")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df["anomaly_score"], bins=80, kde=True)
plt.axvline(threshold, color="red", linestyle="--")
plt.title("Anomaly score distribution")
plt.tight_layout()
plt.savefig("contourB_score_hist.png")
plt.show()

# --- 6. Incident table (merge contiguous anomaly windows) ---
df["anomaly_group"] = (df["anomaly_flag"].diff(1) != 0).cumsum() * df["anomaly_flag"]
incidents = []
for gid, g in df.groupby("anomaly_group"):
    if gid == 0: continue
    start = g["timestamp"].iloc[0]
    end = g["timestamp"].iloc[-1]
    top_features = agg.iloc[g.index].mean().sort_values(ascending=False).head(3).index.tolist()
    incidents.append({"group":int(gid), "start":start, "end":end, "top_features":top_features})
inc_df = pd.DataFrame(incidents)
print("\nDetected incidents:")
print(inc_df)

# Save results
df.to_csv("contourB_results.csv", index=False)
inc_df.to_csv("contourB_incidents.csv", index=False)
print("\nSaved results: contourB_results.csv, contourB_incidents.csv")

Что делает этот скрипт:

Генерит реалистичную временную серию с 5 каналами и тремя встроенными аномалиями.

Делает простые window-агрегации.

Обучает IsolationForest на «нормальной» части (time-aware).

Выдаёт anomaly score, бинарный флаг по порогу, рисует timeline и histogram.

Экспортирует contourB_results.csv и contourB_incidents.csv — эти файлы можно загрузить в презентацию/README.



---


---

17) Как это связать с другими контурами (синергия)

Contour A (soft-sensors) даёт предсказание purity; Contour B даёт ранние аномалии в процессе. B → A: ранняя тревога может включать пересчёт A (soft-sensor) и пересылку рекомендации.

Contour C (PdM / Optimization): аномалии оборудования из B могут триггерить PdM. Оптимизатор C использует B как сигнал плохого режима для DoE / BO.
Вместе даёт: раннее предупреждение (B) → быстрые корректировки (A) → долгосрочное улучшение процесса и планирование тех.обслуживания (C).



---

18) Следующие шаги (быстро и прагматично)

1. Запустить contour_b.py локально — собрать PNG и CSV.


2. Положить скрипт + PNG + README в GitHub (отдельная папка contours/contour_b/).


3. Получить и анализировать текст: timeline image, hist image, table incidents (2–3 строчки), краткую метрику (Precision, Recall, если есть лейбл) и бизнес-эффект (lead time → $ saved estimate).


4. Перекрёстно ссылать на Contour A (show how B reduces false positives & increases lead time).




















 Контур C (Predictive Maintenance / PdM) так же подробно и практичнои: что он даёт, какие данные и фичи нужны, какие модели и метрики, deployment, ROI, риски — и готовый рабочий скрипт contour_c.py, который можно прямо положить в репозиторий (если нет реальных данных — он сам сгенерит правдоподобную синтетику и выведет графики/CSV для презентации).
Ниже — компактный, но полный блок, который вы можете вставить в README.md или в слайд «Контур C».

Контур C — Predictive Maintenance (PdM) — полный практичный разбор

Коротко: цель — предсказать поломку/снижение ресурса оборудования (RUL / вероятность отказа в заданном горизонте) заранее, чтобы планировать техобслуживание до аварии и сократить незапланированные простои.


---

1) Что выдаёт контур C — артефакты

RUL (remaining useful life) — прогноз оставшегося времени/циклов до отказа (регрессия).

Probability of failure в горизонте H (напр. 24–72 часа) — классификация (вероятность / риск).

Alert / maintenance ticket — машина X: P(failure)<0.2 — мониторить, >0.6 — плановое ТО, >0.85 — срочное вмешательство.

Диагностика/фичи с наибольшим вкладом — какие показатели (вибрация, ток, температура) растут перед отказом.

Временная линия для каждой машины: метрики + окно предупреждения (lead time).

Файлы: contourC_results.csv, contourC_metrics.txt, PNG-графики (RUL scatter, ROC, feature-importance, timeline).



---

2) Какие данные нужны

Онлайн (SCADA/DCS): вибрация, температура подшипников, ток/мощность двигателя, давление, скорость/крутящий момент — частота, например 1–60s.

Логи CMMS: история ремонтов, тип поломки, время простоя.

MES: machine_id, shift, оператор, batch_id (если релевантно).

Дата-штампами: timestamps синхронизированы.

Метки (если есть): время отказа, тип отказа (для supervised/ survival). Без меток — можно делать anomaly-based PdM / unsupervised detection.



---

3) Как формулируем задачу (цели/таргеты)

RUL (regression): прогнозировать оставшееся время/циклы до отказа.

Failure-in-horizon (classification): предсказать, случится ли отказ в ближайшие H единиц времени (например, H=48 часов/50 циклов).

Агрегируемой целью для бизнеса: сократить незапланированные простои на X% (целевое), максимизировать lead time для техобслуживания.



---

4) Preprocessing (критично)

Синхронизация таймштампов, ресемплинг до единой частоты (напр. 1 мин).

Разделение по machine_id — все агрегации внутри машины.

Детрендинг / фильтрация шумов (low-pass / median) для вибрации.

Маркировка пропусков (flag) — их наличие важно.

Windowing: скользящие окна (например, 30–300 с) и агрегаты — mean/std/max/min/slope.

Создание таргетов: rul (time to failure) и label_horizon (rul <= H).



---

5) Feature engineering (работающие фичи)

Time-domain: rolling mean/std/last/slope для vibration/temp/current.

Rate features: d(vibration)/dt, relative change in current.

Frequency features для вибрации: FFT-peak, band energy, spectral centroid.

Health indexes: RMS vibration, kurtosis (для подшипников).

Usage features: running_hours_since_last_repair, cycles_count.

Aggregатные признаки: %time_above_threshold за окно, peaks_count.

Латентные признаки: bottleneck из автоэнкодера (опционально).



---

6) Алгоритмы — что и почему

Supervised (если есть метки):

RUL regression: GradientBoostingRegressor / XGBoost / LightGBM — быстрый, устойчивый к шуму.

Failure classifier (horizon): RandomForest / XGBoost — probability output, легко объясним.

Survival analysis: CoxPH (lifelines) — если важна модель времени до события и ценится цензурированная оценка.


Unsupervised / semi-supervised:

Autoencoder (LSTM/Conv1D) для reconstruction error → предвестники деградации.

One-Class / IsolationForest при малом количестве отказов.


Онлайн: модели LightGBM/XGBoost онлайн-скорятся через FastAPI; drift detection — триггер retrain.


---

7) Валидация и метрики

Regression (RUL): MAE (часы/циклы), RMSE, R².
Classification (failure-in-H): ROC-AUC, PR-AUC, Precision@K, Recall@lead_time, F1.
Operational metrics: Lead time (median/mean) — сколько заранее модель предупреждает; reduction in MTTR/MTBF; reduction of unplanned downtime (часы/год).
Валидация: time-aware split (train на ранних данных, test — на новых), CV: grouped by machine / rolling windows. Shadow mode — 2–6 недель для оценки.


---

8) Deployment & MLOps (практично)

Ingest → Kafka/MQTT (при стриме) или файловая выгрузка в S3.

Preproc → Spark/Flink / lightweight service.

Score API → FastAPI (модель), сохраняет прогнозы в TSDB (Influx) + в S3 CSV.

Alerts → rule engine → SCADA/Grafana/Slack/SMS.

Model registry → MLflow; мониторинг drift → автоматический retrain pipeline.

Audit: лог версий модели, входных данных, SHAP-отчётов (для регулятора и инженеров).



---

9) Explainability / Operator UX

Top features (per-prediction SHAP) — почему именно риск поднялся.

Health dashboard: для каждой машины — P(failure), RUL, top-3 contributing sensors, recommended action (inspect bearing / schedule replacement).

Ticketing: auto-create maintenance ticket с priority и predicted window.



---

10) Риски и митигейты

False positives → ensemble + dynamic threshold + confirmation rule (проверочный сенсор / взять пробу).

Data drift → drift detection triggers retrain.

Sensor failure ≠ machine failure → sensor health channel + redundancy.

Operator distrust → shadow mode + обучение + explainable tiles.

Regulatory trace → audit trail: модель, версия, вход, объяснение.



---

11) ROI & бизнес-эффект (кратко, числа)

(оценки ориентировочные; используйте реальные цифры ChemSolutions для точности)

Уменьшение незапланированных простоев: −25–40% → экономия $2–5M/год.

Сокращение аварийных ремонтов и запасов запчастей: −10–20% затрат на запчасти.

Увеличение OEE и On-Time Delivery (подтверждение контрактов) → дополнительная выручка.

Типичный CAPEX на PdM-пилот: $80–200k; окупаемость — 3–9 месяцев при масштабе производства среднего размера.



---




---

12) Как это повышает оценки жюри (mapping)

Актуальность: уменьшает дорогостоящие незапланированные простои → прямой бизнес-болл.

Инновации: гибрид supervised+autosensing, survival modeling и explainable outputs.

Бизнес-реализуемость: модульное подключение к SCADA / CMMS — быстрый пилот.

Технический вклад: RUL + probabilistic failure, real-time scoring, drift detection.

Презентация: графики (lead time, RUL scatter, ROC) — понятные судье.



---

14) MVP-код: contour_c.py (вставляйте в репо)

> Инструкции:

1. Положите contour_c.py в папку репозитория.


2. pip install -r requirements.txt (requirements внизу).


3. Запустите python contour_c.py.


4. Если есть data_C.csv — скрипт использует реальную таблицу (формат описан в коде). Если нет — генерирует синтетику.


5. Скрипт сохранит PNG и CSV — их вставляете в слайды/README.





# contour_c.py
# Python 3.8+
# pip install numpy pandas matplotlib scikit-learn seaborn joblib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from joblib import dump

sns.set(style="whitegrid", rc={"figure.figsize": (10,5)})

def generate_synthetic_pd_data(n_machines=8, seed=42):
    np.random.seed(seed)
    rows = []
    for mid in range(n_machines):
        life = np.random.randint(400, 1200)  # cycles until failure for this machine
        baseline_vib = 0.5 + np.random.rand()*0.5
        baseline_temp = 50 + np.random.rand()*10
        baseline_cur  = 5 + np.random.rand()*2
        for t in range(life):
            # slow degradation + noise
            vib = baseline_vib + 0.0008*t + np.random.normal(0, 0.05) + 0.01*np.sin(t/10)
            temp = baseline_temp + 0.0006*t + np.random.normal(0, 0.2)
            cur  = baseline_cur + 0.0004*t + np.random.normal(0, 0.05)
            rows.append((mid, t, vib, temp, cur, life - t))
    df = pd.DataFrame(rows, columns=["machine_id","t","vibration","temperature","current","rul"])
    # failure label for horizon H (we'll compute later)
    return df

def fe_engineer(df, window=5):
    df = df.sort_values(["machine_id","t"]).reset_index(drop=True)
    df_grouped = []
    for mid, g in df.groupby("machine_id"):
        g = g.copy()
        for col in ["vibration","temperature","current"]:
            g[f"{col}_mean"] = g[col].rolling(window, min_periods=1).mean()
            g[f"{col}_std"]  = g[col].rolling(window, min_periods=1).std().fillna(0)
            g[f"{col}_slope"] = g[col].diff().rolling(window, min_periods=1).mean().fillna(0)
            g[f"{col}_last"] = g[col]
        g["running_hours"] = g["t"]
        df_grouped.append(g)
    return pd.concat(df_grouped).reset_index(drop=True)

def prepare_labels(df, horizon=50):
    # classification label: will fail within horizon?
    df["label_horizon"] = (df["rul"] <= horizon).astype(int)
    return df

def train_models(df, horizon=50):
    features = [c for c in df.columns if any(x in c for x in ["vibration_","temperature_","current_","running_hours"])]
    # time-aware split: for each machine take first 70% timestamps as train, rest as test
    train_idx = []
    test_idx = []
    for mid, g in df.groupby("machine_id"):
        n = len(g)
        split = int(0.7 * n)
        train_idx.extend(g.index[:split].tolist())
        test_idx.extend(g.index[split:].tolist())
    train = df.loc[train_idx].reset_index(drop=True)
    test  = df.loc[test_idx].reset_index(drop=True)

    X_train = train[features].fillna(0)
    y_rul_train = train["rul"]
    y_clf_train = train["label_horizon"]

    X_test = test[features].fillna(0)
    y_rul_test = test["rul"]
    y_clf_test = test["label_horizon"]

    # Regression: RUL
    reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    reg.fit(X_train, y_rul_train)
    y_rul_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_rul_test, y_rul_pred)
    rmse = mean_squared_error(y_rul_test, y_rul_pred, squared=False)
    r2 = r2_score(y_rul_test, y_rul_pred)

    # Classification: failure within horizon
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X_train, y_clf_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc = roc_auc_score(y_clf_test, y_prob)
    prec = precision_score(y_clf_test, y_pred, zero_division=0)
    rec = recall_score(y_clf_test, y_pred, zero_division=0)

    metrics = {
        "rul_mae": mae, "rul_rmse": rmse, "rul_r2": r2,
        "clf_roc_auc": roc, "clf_precision": prec, "clf_recall": rec
    }

    results = {
        "model_reg": reg,
        "model_clf": clf,
        "X_test": X_test, "y_rul_test": y_rul_test, "y_rul_pred": y_rul_pred,
        "y_clf_test": y_clf_test, "y_prob": y_prob,
        "metrics": metrics, "features": features, "test_idx": test.index
    }
    return results

def plots_and_export(df, results, out_prefix="contourC"):
    # RUL scatter
    plt.figure(figsize=(7,5))
    plt.scatter(results["y_rul_test"], results["y_rul_pred"], alpha=0.5)
    m = max(results["y_rul_test"].max(), np.nanmax(results["y_rul_pred"]))
    plt.plot([0,m],[0,m], "r--")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs Actual RUL")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rul_scatter.png")
    plt.close()

    # ROC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(results["y_clf_test"], results["y_prob"])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Failure within horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png")
    plt.close()

    # feature importances (clf)
    importances = results["model_clf"].feature_importances_
    feat_imp = pd.Series(importances, index=results["features"]).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8,4))
    feat_imp.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top feature importances (classifier)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_feat_importance.png")
    plt.close()

    # timeline for a sample machine (last machine in dataset)
    sample_mid = df["machine_id"].unique()[-1]
    sample = df[df["machine_id"]==sample_mid].copy()
    # map predictions onto sample times if indices align
    test_idx_map = results["test_idx"].tolist()
    sample_test = sample[sample.index.isin(test_idx_map)]
    if not sample_test.empty:
        sample_test = sample_test.copy()
        sample_test["pred_prob"] = results["y_prob"][-len(sample_test):]
        sample_test["pred_rul"] = results["y_rul_pred"][-len(sample_test):]
        plt.figure(figsize=(12,4))
        plt.plot(sample_test["t"], sample_test["vibration_last"], label="vibration")
        plt.plot(sample_test["t"], sample_test["pred_prob"]*sample_test["vibration_last"].max(), label="pred_prob (scaled)")
        plt.scatter(sample_test["t"][sample_test["label_horizon"]==1], sample_test["vibration_last"][sample_test["label_horizon"]==1], color="red", s=20, label="actual failure window")
        plt.title(f"Machine {sample_mid} timeline (vibration + predicted prob)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_timeline_machine_{sample_mid}.png")
        plt.close()

    # save CSV results
    df_results = sample_test[["machine_id","t","vibration_last","temperature_last","current_last","pred_prob","pred_rul","label_horizon"]].copy() if not sample_test.empty else pd.DataFrame()
    df_results.to_csv(f"{out_prefix}_sample_timeline.csv", index=False)
    # Save metrics
    with open(f"{out_prefix}_metrics.txt","w") as f:
        for k,v in results["metrics"].items():
            f.write(f"{k}: {v}\n")
    print("Saved plots and metrics files.")

def main():
    # 1) Load data_C.csv if exists, else generate synthetic
    if os.path.exists("data_C.csv"):
        df = pd.read_csv("data_C.csv")
        # Expected columns: machine_id,timestamp,vibration,temperature,current, (optionally failure/time_to_failure)
        # Minimal required: machine_id,t (time or cycle index), vibration, temperature, current
        # If time_to_failure or rul present, use it; else we'll not have supervised labels.
        if "rul" not in df.columns:
            print("data_C.csv found but no 'rul' column — supervised RUL won't be available. Consider adding 'rul' or run unsupervised mode.")
    else:
        print("No data_C.csv found — generating synthetic dataset...")
        df = generate_synthetic_pd_data(n_machines=8)
    df_fe = fe_engineer(df, window=5)
    df_label = prepare_labels(df_fe, horizon=50)
    # Train models (supervised)
    results = train_models(df_label, horizon=50)
    # Save models
    dump(results["model_reg"], "contourC_reg.joblib")
    dump(results["model_clf"], "contourC_clf.joblib")
    # Attach some columns for timeline plotting (last/mean etc.)
    # rename last cols for readability
    for c in ["vibration_last","temperature_last","current_last"]:
        if c not in df_label.columns:
            # if not exist, map from base columns
            base = c.split("_")[0]
            df_label[c] = df_label[base]
    plots_and_export(df_label, results, out_prefix="contourC")
    # Save full test-results CSV (merge indices)
    test_idx = results["test_idx"]
    test_df = df_label.loc[test_idx].copy()
    test_df["pred_rul"] = results["y_rul_pred"]
    test_df["pred_prob"] = results["y_prob"]
    test_df.to_csv("contourC_results.csv", index=False)
    print("Saved contourC_results.csv")
    print("Metrics:", results["metrics"])
    print("Models saved: contourC_reg.joblib, contourC_clf.joblib")

if __name__ == "__main__":
    main()


---

15) requirements.txt (минимум)

numpy
pandas
matplotlib
seaborn
scikit-learn
joblib


---



---

16) Коротко — что нужно показать жюри (слайды/README)

RUL scatter (MAE, RMSE) — демонстрирует точность предсказания срока жизни.

ROC + Precision/Recall — демонстрирует способность предупреждать об отказе.

Timeline sample — показывает lead time и как визуально оператор видит предупреждение.

Feature importance & SHAP (опционально) — объяснимость.

Business slide — сколько часов простоев сэкономлено, ROI (пример: −25% незаплан. простоев → $2–5M/год).













🛠 Инструкция для жюри

Чтобы воспроизвести работу прототипа:

1. Откройте Google Colab

Перейдите на https://colab.research.google.com/.

Нажмите File → New Notebook.



2. Скопируйте код

Возьмите блок кода из этого репозитория (в README уже вставлен).

Вставьте его в первую ячейку Colab.



3. Установите зависимости

Запустите команду (первая строка кода):

!pip install xgboost shap scikit-learn pandas matplotlib



4. Запустите весь код по порядку

Нажмите Runtime → Run all.

После выполнения появятся:

метрики модели (ROC-AUC, MAE и др.);

графики (SHAP summary plot, predicted vs real);

простые рекомендации («уменьшите temp на 1℃ → +Δ к вероятности качества»).




5. Что смотреть в выводе

Метрики → качество предсказания.

Графики SHAP → какие параметры влияют на прогноз.

Рекомендации → примеры автоматических советов для оператора.





---

⚡ Примечание: для ускорения мы используем синтетические данные, но код полностью готов работать и с реальными датасетами (достаточно заменить batches_aggregated.csv).



