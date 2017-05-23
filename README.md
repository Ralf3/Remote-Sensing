# Remote-Sensing
A data set with hyper spectral data and methods to use this data for Machine Learning are provided. 

The Excel-Files contain the hyper spectral data for different crops.
For each date there are different samples called "Spectrum_n" (n=1,2,3,...).

This data set will be used to generate different collections of training data
(called modells). The Python program "gen_sample_all.py" helps to select
the training data for machine learning examples. The wavelength are restricted
to the wavelength of the satellite "Sentinel2" but can be adapted if necessary. 
The user can select a sampling date from a list and use this data for training
runs using machine learning algorithms. This example was the result of a
training course for Batu (Batunacun). She developed parts of the python code and
will add new code in the future.

# Crops

ALF = Alfalfa = Luzerne

COC = Cocks Grass = Knaulgras 

LUP = Lupin = Lupine

PEA = Pea = Erbse

POT = Potato = Kartoffel

SMA = Silo Maize = Silomais

TRI = Triticale = Triticale

WBAR = Winter Barley = Wintergerste

WRP = Winter Rape = Winterraps

WRY = Winter Rye = Winterroggen

WWH = Winter Wheat = Winterweizen

# Acknowledgment

We want to thanks Dr. Bernd Zbell who has collected the data and allowed us
to use it. This project is supported by the ZALF "http://www.zalf.de" . 
The software is based on Python and uses the packages "sklearn", "numpy",
"matplotlib" and "pandas" which should be installed.