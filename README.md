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

* calibration\_spectra.csv : Raw emission spectra for mercury lamp and 405 nm / 450 nm lasers.
* pixel\_wavelength\_mapping.csv : Extracted calibration peaks and polynomial fit mapping pixels to wavelength (nm).
* calibration\_plot.png : Calibration curve visualization.
* calibration\_spectra.png : Raw calibration spectra visualization.



This calibration establishes the wavelength accuracy of the custom diffraction-based spectrometer.



### 02\_Fluorescence\_Spectra



Contains all fluorescence spectra used for model development.



#### Single\_Pigment\_Spectra/



* ChlA\_spectrum.csv : Raw and smoothed fluorescence emission spectrum of Chl-a under selective laser excitation.
* ChlB\_spectrum.csv : Raw and smoothed fluorescence emission spectrum of Chl-b under selective laser excitation.
* Fluorescence\_raw\_spectra.png : Overlay plot of raw fluorescence spectra for Chl-a and Chl-b.
* Fluorescence\_smoothed\_spectra.png : Overlay plot of Savitzky–Golay smoothed fluorescence spectra for Chl-a and Chl-b with fluorescence peaks marked.



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



* chl\_2000\_3694.csv : Raw pixel intensity data (pixels 2000–3693) for all 11 samples, including corresponding Chl-a and Chl-b concentrations (mg/L), used as input for all machine learning model training and validation without baseline correction, smoothing, or normalization. Each row corresponds to one sample.



#### Model\_testing/



Includes:



* linear\_regression\_model.py : Python script for training and evaluating a linear regression model using spectral pixel features.



* polynomial\_regression\_model.py : Python script for training the second-order polynomial regression model using SVD-reduced spectral features.



* randomForest\_regression\_model.py : Python script for training and evaluating the random forest regression model using spectral input data.



* linear\_LOO\_CV.py : Python script implementing Leave-One-Out Cross-Validation for linear regression models.



* polynomial\_LOO\_CV.py : Python script performing Leave-One-Out Cross-Validation for the second-order polynomial regression model.



* randomForest\_LOO\_CV.py : Python script performing Leave-One-Out Cross-Validation for the random forest regression model.



* ChlA\_poynomial\_LOO\_CV\_results.csv : Leave-One-Out Cross-Validation performance metrics for the polynomial regression model predicting Chl-a concentration.



* ChlB\_poynomial\_LOO\_CV\_results.csv : Leave-One-Out Cross-Validation performance metrics for the polynomial regression model predicting Chl-b concentration.



* polynomial\_LOO\_CV\_estimated\_vs\_actual\_plot.png : Visualization comparing predicted versus actual chlorophyll concentrations for the polynomial regression model under LOO-CV.



Polynomial regression (2nd order) demonstrated best performance.



#### TFLite\_Model\_Conversion/



Separate folders for Chl-a and Chl-b:



Includes:

* Polynomial\_SVD.py : Python script applying Singular Value Decomposition (SVD) and polynomial feature transformation for Chl-a and Chl-b  model preparation.



* Poly\_to\_TFlite.py : Python script converting the trained Chl-a and Chl-b polynomial regression model into TensorFlow Lite format.



* SVD\_components\_to\_C\_array.py : Utility script exporting trained SVD components into C-array format for embedded deployment.



* chl\_**x**\_model.tflite : TensorFlow Lite regression model for real-time Chl-**x** concentration estimation.



* poly\_model\_chl**x**.pkl : Serialized scikit-learn polynomial regression model for Chl-**x**.



* poly\_transform.pkl : Serialized second-order polynomial feature transformer expanding 11 SVD components into 78 features for regression model training and deployment.



* svd\_chl**x**.pkl : Serialized TruncatedSVD model used to reduce raw spectral pixel inputs to 11 principal components for Chl-**x**.



**Dimensionality Reduction:**



Singular Value Decomposition (SVD) was applied.

Number of components selected: 11



### 04\_Edge\_Deployment



Contains all files required for microcontroller deployment.

EdgeImpulse\_Export/



Includes:

* ei-chl\_conc\_model\_a-arduino-1.0.2.zip
* ei-chl\_conc\_model\_b\_poly-arduino-1.0.1.zip

Edge Impulse–generated Arduino deployment package containing the optimized TensorFlow Lite Micro inference library for Chl concentration estimation.



The .tflite models were uploaded to Edge Impulse to generate optimized Arduino inference libraries.





#### Arduino\_Code/



Include:

* ChlA\_arduino\_inferencing\_library.zip : Packaged Arduino inference library for integrating the Chl-a TensorFlow Lite model into the Arduino Nano 33 BLE Sense environment.



* ChlA\_recieve\_inference.ino : Arduino sketch implementing real-time Chl-a concentration inference using the deployed TensorFlow Lite model.



* ChlB\_arduino\_inferencing\_library.zip : Packaged Arduino inference library for integrating the Chl-b TensorFlow Lite model into the Arduino Nano 33 BLE Sense environment.



* ChlB\_recieve\_inference.ino:Arduino sketch implementing real-time Chl-b concentration inference using the deployed TensorFlow Lite model.



Target hardware:



* Arduino Nano 33 BLE Sense



Inference type:



* Regression (TensorFlow Lite Micro)





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
