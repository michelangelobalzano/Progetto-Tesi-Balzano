# Rilevamento automatico delle emozioni mediante dati raccolti da dispositivi indossabili

Riconoscimento automatico delle emozioni attraverso la classificazione di valenza e attivazione e l'analisi di segnali biometrici catturati dal dispositivo Empatica E4 wristband. 

I dati utilizzati appartengono allo studio "Introducing wesad, a multimodal dataset for wearable stress and affect detection." di Schmidt et al. 

Vengono utilizzati due approcci differenti: modelli di Classic Machine Learning (CML) e modello di Deep Learning transformer.

## Configurazione

Una volta scaricato il repository, estrarre l'achivio WESAD, contenente i dati biometrici e i valori di valenza e attivazione necessari alla task di classificazione. La cartella ottenuta deve chiamarsi "WESAD" e deve contenere al suo interno 15 cartelle chiamate con numeri da 1 a 15 corrispondenti ai 15 soggetti del dataset.

Prima di poter classificare mediante i modelli CML o con il transformer, è necessario effettuare il preprocessing dei dati. Il preprocessing crea i segmenti di dati di input per il transformer ed estrae le feature di input per i modelli CML. 

## Preprocessing dei dati

Per processare i dati, creare un ambiente virtuale relativo al modulo di preprocessing. Dalla cartella principale del repository, spostarsi nella cartella del preprocessing con il comando:

```bash
cd preprocessing
```

Creare l'ambiente virtuale del preprocessing:

```bash
python -m venv preprocessing
```

Installare le librerie necessarie al preprocessing:

```bash
pip install -r requirements.txt
```

Eseguire il preprocessing:

```bash
python main.py --segmentation_window_size 60 --segmentation_step_size 10 --neutral_range 0.2
```

I parametri utilizzabili sono i seguenti:

* segmentation_window_size: (opzionale) dimensione della finestra di segmentazione (secondi). Valori possibili: [30, 60, 120], valore consigliato: 60.
* segmentation_step_size: (opzionale) dimensione del passo di segmentazione (secondi). Valori possibili: [5, 10, 15], valore consigliato: 10.
* neutral_range: range dalla media delle etichette neutral. Valori possibili: [0.2, 0.35, 0.5], valore consigliato: (opzionale) 0.2. Non utilizzare il valore 0.5 se si vuole classificare con il modello transformer con split dei dati per soggetto in quanto si generano troppi segmenti neutral.

## Classificazione mediante modelli CML

Dalla cartella principale del repository, spostarsi nella cartella CML con il seguente comando:

```bash
cd CML
```

Creare l'ambiente virtuale:

```bash
python -m venv CML
```

Installare le librerie necessarie:

```bash
pip install -r requirements.txt
```

### Ottimizzazione degli iperparametri (opzionale)

### Classificazione

## Classificazione mediante transformer

### Ottimizzazione degli iperparametri (opzionale)

### Classificazione