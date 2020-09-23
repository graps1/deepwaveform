## Abhängigkeiten
* pytorch
* setuptools
* pandas
* matplotlib
* python >=3.5
* für die Beispiele: jupyter notebook

## Installation
Unter Linux: `sudo python3 setup.py install` aufrufen.

## Benutzung

### Beispiele
Um die Beispiele ausführen zu können, müssen zuerst ein paar Datensätze heruntergeladen werden. Dazu kann (unter Linux) `download_data.sh` ausgeführt werden. Die Datensätze werden in diesem Fall in dem Unterordner `./data` gespeichert. Unter Windows kann diese [.zip-Datei](https://cloudstore.zih.tu-dresden.de/index.php/s/cq9Wra6PERtFCKf/download) heruntergeladen und dann im `./data`-Ordner entpackt werden.

`examples/load_dataset.ipynb`:
* Laden eines Datensatzes
* Plotten der Punktwolke
* Plotten von Waveforms

`examples/train_classifier.ipynby`:
* Trainieren eines Klassifikators (Convolutional Neural Networks) auf einem gegebenen Datensatz
* Visualisieren von Trainingszwischenergebnissen
* Plotten der Confusion Matrix des Klassifikators über den Datensatz
* Speichern und Laden eines trainierten Modells
* Annotieren eines Datensatzes mit den vorhergesagten Klassen
* Plotten der Punktwolke mit den vorhergesagten Wahrscheinlichkeiten