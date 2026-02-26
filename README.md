# Dataset-and-Code-for-Dual-Wavelength-Fluorescence-Based-Chlorophyll-Sensor



This repository contains the complete dataset, machine learning models, and embedded deployment code supporting the manuscript:


### "Dual-Wavelength Laser-Induced Fluorescence Sensor with Embedded Machine Learning for Simultaneous Quantification of Chlorophyll-a and Chlorophyll-b"



The repository enables full reproducibility of:

* Spectrometer wavelength calibration
* Fluorescence spectral acquisition and preprocessing
* Machine learning model training and validation
* TensorFlow Lite model conversion
* Arduino Nano 33 BLE Sense edge deployment





### 01\_Spectrometer\_Calibration



Contains raw spectral data and calibration mapping used to convert CCD pixel index to wavelength.



**Files:**

* calibration\_spectra.csv

&nbsp;	Raw emission spectra for mercury lamp and 405 nm / 450 nm lasers.

* pixel\_wavelength\_mapping.csv

&nbsp;	Extracted calibration peaks and polynomial fit mapping pixels to wavelength (nm).

* calibration\_plot.png

&nbsp;	Calibration curve visualization.

* calibration\_spectra.png

&nbsp;	Raw calibration spectra visualization.



This calibration establishes the wavelength accuracy of the custom diffraction-based spectrometer.



### 02\_Fluorescence\_Spectra



Contains all fluorescence spectra used for model development.



##### Single\_Pigment\_Spectra/



* ChlA\_spectrum.csv
* ChlB\_spectrum.csv
* Fluorescence\_raw\_spectra.png
* Fluorescence\_smoothed\_spectra.png



**Spectral Preprocessing**



Smoothed spectra were generated using a Savitzky–Golay filter with:



* Window length = 50 points
* Polynomial order = 2



##### Concentration\_Series/



Contains fluorescence spectra across multiple concentration levels.



* ChlA\_concentration\_series\_spectra.csv
* ChlB\_concentration\_series\_spectra.csv
* Corresponding visualization PNG files.



These datasets were used for regression model development and validation.







