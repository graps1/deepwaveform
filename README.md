## Dokumentation
Kann unter https://graps1.github.io/deepwaveform/_build/html/index.html gefunden werden.

## Abhängigkeiten
* pytorch
* setuptools
* pandas
* matplotlib
* numpy
* sklearn
* python >=3.5
* für die Beispiele: jupyter notebook

## Installation
Unter Linux: `sudo python3 setup.py install` aufrufen.

## Benutzung
### Datenformat
Datensätze werden als CSV eingelesen. Jeder Datensatz muss die Spalten `index`, `x`, `y` und `z` enthalten. Zusätzlich kommen Spalten für die Waveform und eine optionale Spalte für die Klasse, deren Namen aber nicht vordefiniert sind. Separiert werden Einträge durch ein Semikolon `;`. Beispiele für gültige Datensätze sind [hier](https://cloudstore.zih.tu-dresden.de/index.php/s/cq9Wra6PERtFCKf/download) enthalten.

### Beispiele
Um die Beispiele ausführen zu können, müssen zuerst ein paar Datensätze heruntergeladen werden. Dazu kann (unter Linux) `download_data.sh` ausgeführt werden. Die Datensätze werden in diesem Fall in dem Unterordner `./data` gespeichert. Unter Windows kann diese [.zip-Datei](https://cloudstore.zih.tu-dresden.de/index.php/s/cq9Wra6PERtFCKf/download) heruntergeladen und dann im `./data`-Ordner entpackt werden. Es wird empfohlen, die Notebooks in der Reihenfolge abzuarbeiten, wie sie hier aufgeführt sind:

`examples/load_dataset.ipynb`:
* Laden eines Datensatzes
* Plotten der Punktwolke
* Plotten von Waveforms

`examples/classifier.ipynb`:
* Trainieren eines Klassifikators (Convolutional Neural Networks) auf einem gegebenen Datensatz
* Visualisieren von Trainingszwischenergebnissen
* Plotten der Confusion Matrix des Klassifikators über den Datensatz
* Speichern und Laden eines trainierten Modells
* Annotieren eines Datensatzes mit den vorhergesagten Klassen
* Plotten der Punktwolke mit den vorhergesagten Wahrscheinlichkeiten

`examples/autoencoder.ipynb`:
* Trainieren eines Autoencoders auf einem gegebenen Datensatz
* Visualisieren von Trainingszwischenergebnissen
* Annotieren eines Datensatzes mit dem trainierten Autoencoder
* Visualisieren der Rekonstruktionen
* Speichern des Autoencoders

`examples/clustering.ipynb`:
* Laden eines gespeicherten Autoencoders
* Clustering mit Hilfe von K-Means über die Waveforms und die Kodierungen der Waveforms
* Vergleich der gefundenen Cluster
* Visualisierung der Zentroiden

### Command Line Tool
Nach der Installation und dem Download der Datensätze (s.o.) können bspw. folgende Kommandos in der Kommandozeile ausgeführt werden:

#### Klassifikator trainieren und Datensatz annotieren

    cd data
    dwf train classifier elbabschnitt.csv -e=15 -bs=1024 -c="class" -o="classifier.pt"
    dwf annotate mit_uferuebergang.csv classifier.pt -o="mit_uferuebergang.classified.csv" -l="['Land', 'Water']"
    dwf renderprob mit_uferuebergang.classified.csv -pc="['Land', 'Water']" -co="['green','blue']"
    dwf renderclass mit_uferuebergang.classified.csv -c="predicted" -l="['Land','Water']" -co="['green','blue']"


#### Autoencoder trainieren und Datensatz annotieren

    cd data
    dwf train autoencoder mit_uferuebergang.csv -e=15 -bs=1024 -o="autoencoder.pt" -hd=12
    dwf annotate elbabschnitt.csv autoencoder.pt -o="elbabschnitt.encoded.csv"


#### Datensatz clustern und annotieren
Wenn ein Autoencoder trainiert und der Datensatz entsprechend annotiert wurde:

    cd data
    dwf cluster elbabschnitt.encoded.csv -f="list(map(str, range(64)))" -o="elbabschnitt.encoded.clusteredwaveforms.csv" -cc=5
    dwf cluster elbabschnitt.encoded.csv -f="['hidden_%d' % idx for idx in range(12)]" -o="elbabschnitt.encoded.clusteredhidden.csv" -cc=5
    dwf renderclass elbabschnitt.encoded.clusteredwaveforms.csv -c="cluster"
    dwf renderclass elbabschnitt.encoded.clusteredhidden.csv -c="cluster"


## Todo
* Windows-Installation