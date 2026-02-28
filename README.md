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

 	Raw emission spectra for mercury lamp and 405 nm / 450 nm lasers.

* pixel\_wavelength\_mapping.csv

 	Extracted calibration peaks and polynomial fit mapping pixels to wavelength (nm).

* calibration\_plot.png

 	Calibration curve visualization.

* calibration\_spectra.png

 	Raw calibration spectra visualization.



This calibration establishes the wavelength accuracy of the custom diffraction-based spectrometer.



### 02\_Fluorescence\_Spectra



Contains all fluorescence spectra used for model development.



#### Single\_Pigment\_Spectra/



* ChlA\_spectrum.csv
* ChlB\_spectrum.csv
* Fluorescence\_raw\_spectra.png
* Fluorescence\_smoothed\_spectra.png



**Spectral Preprocessing**



Smoothed spectra were generated using a Savitzky–Golay filter with:



* Window length = 50 points
* Polynomial order = 2



#### Concentration\_Series/



Contains fluorescence spectra across multiple concentration levels.



* ChlA\_concentration\_series\_spectra.csv : Raw and smoothed fluorescence spectra of chlorophyll-a measured at multiple concentrations for regression model development.
* ChlB\_concentration\_series\_spectra.csv : Raw and smoothed fluorescence spectra of chlorophyll-a measured at multiple concentrations for regression model development.
* Corresponding visualization PNG files. : Visualization of fluorescence emission spectra of Chl-a and Chl-b across different concentration levels.



These datasets were used for regression model development and validation.



### 03\_Machine\_Learning



Contains all scripts, processed datasets, trained models, and validation results.



#### Dataset/



Contains the final processed dataset used for training and validation.



#### Model\_testing/



Includes:



* Linear regression
* Polynomial regression
* Random forest regression
* Leave-One-Out Cross-Validation (LOO-CV) scripts
* Performance result CSV files
* Estimated vs actual plots



Polynomial regression (2nd order) demonstrated best performance.



#### TFLite\_Model\_Conversion/



Separate folders for Chl-a and Chl-b:



Includes:



* SVD transformation scripts
* Polynomial regression training scripts
* Model serialization files (.pkl)
* TensorFlow Lite models (.tflite)
* Conversion scripts



**Dimensionality Reduction:**



Singular Value Decomposition (SVD) was applied.

Number of components selected: 11



### 04\_Edge\_Deployment



Contains all files required for microcontroller deployment.



#### Arduino\_Code/



* Edge Impulse exported Arduino libraries
* Inference .ino files for Chl-a and Chl-b



Target hardware:



* Arduino Nano 33 BLE Sense



Inference type:



* Regression (TensorFlow Lite Micro)



#### EdgeImpulse\_Export/



Contains exported deployment packages generated via the Edge Impulse platform.



The .tflite models were uploaded to Edge Impulse to generate optimized Arduino inference libraries.



### Machine Learning Summary



Model Type : 2nd Order Polynomial Regression

Validation Method : Leave-One-Out Cross Validation (LOO-CV)

Dataset Size : n = 11

SVD Components : 11

Chl-a R² : 0.947

Chl-b R² : 0.968

Chl-a RMSE : 0.071 mg/L

Chl-b RMSE : 0.057 mg/L



### Software Requirements



* Python 3.x
* NumPy
* Pandas
* SciPy
* scikit-learn
* TensorFlow
* Matplotlib
* Joblib
* Arduino IDE
* TensorFlow Lite Micro



### Reproducibility Instructions

To retrain models:



1. Navigate to 03\_Machine\_Learning/
2. Run polynomial regression scripts.
3. Perform LOO-CV validation.
4. Convert trained model to TensorFlow Lite format.
5. Deploy to Arduino using provided .ino files.



All scripts use relative paths to access datasets.



### Notes on Limitations



* Calibration performed using extracted chlorophyll in acetone.
* Dataset size limited (n = 11).
* Validation performed using LOO-CV (no independent external dataset).
* Current validated concentration range: mg/L scale.
