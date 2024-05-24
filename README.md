# Rilevamento automatico delle emozioni mediante dati raccolti da dispositivi indossabili

Riconoscimento automatico delle emozioni attraverso la classificazione di valenza e attivazione e l'analisi di segnali biometrici catturati dal dispositivo Empatica E4 wristband. 

I dati utilizzati appartengono allo studio "Introducing wesad, a multimodal dataset for wearable stress and affect detection." di Schmidt et al. 

Vengono utilizzati due approcci differenti: modelli di Classic Machine Learning (CML) e modello di Deep Learning transformer.

## Configurazione

Una volta scaricato il repository, estrarre l'achivio WESAD, contenente i dati biometrici e i valori di valenza e attivazione necessari alla task di classificazione. La cartella ottenuta deve chiamarsi "WESAD" e deve contenere al suo interno 15 cartelle chiamate con numeri da 1 a 15 corrispondenti ai 15 soggetti del dataset.

Prima di poter classificare mediante i modelli CML o con il transformer, è necessario effettuare il preprocessing dei dati. Il preprocessing crea i segmenti di dati di input per il transformer ed estrae le feature di input per i modelli CML. 

## Preprocessing dei dati

Aprire il prompt dei comandi nella cartella principale del repository.

Creare l'ambiente virtuale:

```bash
python -m venv preprocessing
```

Installare le librerie necessarie al preprocessing:

```bash
pip install -r preprocessing/requirements.txt
```

Eseguire il preprocessing:

```bash
python preprocessing/main.py --segmentation_window_size 60 --segmentation_step_size 10 --neutral_range 0.2
```

I parametri utilizzabili sono i seguenti:

* segmentation_window_size: dimensione della finestra di segmentazione (secondi). Valori possibili: [30, 60, 120], valore consigliato: 60.
* segmentation_step_size: dimensione del passo di segmentazione (secondi). Valori possibili: [5, 10, 15], valore consigliato: 10.
* neutral_range: range dalla media delle etichette neutral. Valori possibili: [0.2, 0.35, 0.5], valore consigliato: 0.2. Non utilizzare il valore 0.5 se si vuole classificare con il modello transformer con split dei dati per soggetto in quanto si generano troppi segmenti neutral.

## Classificazione mediante modelli CML

Aprire il prompt dei comandi nella cartella principale del repository.

Creare l'ambiente virtuale:

```bash
python -m venv CML
```

Installare le librerie necessarie:

```bash
pip install -r CML/requirements.txt
```

Prima di ottimizzare gli iperparametri, o classificare direttamente, bisogna estrarre le features dai dati con il seguente comando:

```bash
python CML/feature_extraction.py
```

Questo comando non necessita di alcun parametro.

Una volta eseguito il comando, verrà creato un file chiamato "features.csv" che verrà utilizzato per l'ottimizzazione degli iperparametri o per la classificazione. Non è necessario ripetere il comando per ogni ottimizzazione o classificazione se il file è stato già creato.

### Ottimizzazione degli iperparametri

Eseguire il seguente comando per ottenere i migliori iperparametri per un determinato modello e una determinata etichetta:

```bash
python CML/param_optimization.py --label arousal --model xgb
```

I parametri utilizzabili sono i seguenti:
* label: etichetta del quale ottimizzare gli iperparametri. Valori possibili: [valence, arousal]. Default: valence. 
* model: sigla del modello da utilizzare. xgb=XGBoost, knn=kNN, rf=random forest, dt=decision tree. Valori possibili: [xgb, knn, rf, dt]. Default: xgb. 

### Classificazione

Eseguire il seguente comando per classificare con un determinato modello, etichetta e metodo di split:

```bash
python CML/main.py --label valence --model xgb --split_type LOSO --xgb_max_depth 3 --xgb_n_estimators 50 --xgb_learning_rate 0.01
```

I parametri utilizzabili sono i seguenti:
* label: etichetta da classificare. Valori possibili: [valence, arousal] Valore di default: valence. 
* model: sigla del modello da utilizzare. xgb=XGBoost, knn=kNN, rf=random forest, dt=decision tree. Valori possibili: [xgb, knn, rf, dt]. Default: xgb.
* split_type: sigla tipo di split dei dati. LOSO=Leave One Subject Out, L2SO, L3SO=Leave 2, 3 subjects out, KF5, KF10=K-Fold Cross Validation k=5, 10. Valori possibili: [xgb, knn, rf, dt]
* xgb_max_depth: profondita' massima. Valori possibili: [3, 5, 10, 20, 30], default: 3.
* xgb_n_estimators: numero di alberi da valutare. Valori possibili: [50, 100, 200], default: 50.
* xgb_learning_rate: tasso apprendimento. Valori possibili: [0.01, 0.1, 0.3, 0.5], default: 0.01.
* knn_n_neighbors: numero di vicini. Valori possibili: [1, 3, 5, 7, 9, 11, 13, 15], default: 3.
* knn_weights: metodo di peso dei vicini. Valori possibili: [uniform, distance], default: uniform.
* knn_metric: metrica per calcolare la distanza. Valori possibili: [euclidean, manhattan, minkowski], default: manhattan.
* rf_max_depth: profondita' massima. Valori possibili: [None, 10, 20, 30], default: 30.
* rf_n_estimators: numero di alberi da valutare. Valori possibili: [50, 100, 200], default: 50.
* rf_min_samples_split: numero minimo di campioni richiesti per dividere un nodo interno. Valori possibili: [2, 5, 10], default: 10.
* rf_min_samples_leaf: numero minimo di campioni che deve avere un nodo foglia. Valori possibili: [1, 2, 4], default: 1.
* dt_max_depth: profondita' massima. Valori possibili: [None, 10, 20, 30], default: 30.
* dt_min_samples_split: numero minimo di campioni richiesti per dividere un nodo interno. Valori possibili: [2, 10, 20], default: 10.
* dt_min_samples_leaf: numero minimo di campioni che deve avere un nodo foglia. Valori possibili: [1, 5, 10], default: 1.
* dt_criterion: funzione di misurazione qualita' di una divisione. Valori possibili: [gini, entropy], default: gini.
* dt_splitter: strategia di scelta della divisione. Valori possibili: [best, random], default: random.

Inserire parametri non relativi al modello scelto non ha alcun effetto.

Per eseguire la classificazione mediante trivial classifier per una determinata etichetta eseguire il seguente comando:

```bash
python CML/trivial_classifier.py --label valence
```

I parametri utilizzabili sono i seguenti:
* label: etichetta da classificare. Valori possibili: [valence, arousal] Valore di default: valence. 

## Classificazione mediante transformer

### Ottimizzazione degli iperparametri

### Classificazione