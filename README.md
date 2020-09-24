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
### Command Line Tool
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

## Todo
* Windows-Installation
* Command Line Tool; benutzbar unter Windows
* Kommentare / Github-Seite via Sphinx aufsetzen