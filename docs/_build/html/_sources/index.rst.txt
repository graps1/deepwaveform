#########################################
Deep Waveform Classification and Encoding
#########################################

Contact: hans.harder@mailbox.tu-dresden.de

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Models
======

.. autoclass:: deepwaveform.ConvNet
   :members: __init__, annotate_dataframe

.. autoclass:: deepwaveform.AutoEncoder
   :members: __init__, annotate_dataframe

Preprocessing
=============

.. automethod:: deepwaveform.preprocessing.load_dataset

.. automethod:: deepwaveform.preprocessing.waveform2matrix

Visualization
=============

.. automodule:: deepwaveform.visualization
   :members:


Dataset and Training
====================

.. autoclass:: deepwaveform.WaveFormDataset
   :members: __init__

.. autoclass:: deepwaveform.Trainer
   :members: __init__, train_classifier, train_autoencoder

