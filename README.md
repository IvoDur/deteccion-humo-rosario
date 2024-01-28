# Wildfire detection with Computer Vision and Machine Learning
Implementation of a software based on Computer Vision and artificial intelligence techniques. Its main objective is the detection of
smoke plumes generated in forest fires, with a specific focus on the application of this system in the detection of fires within wetland
ecosystems in order to be applied in the wetlands of the city of Rosario.
# The App
Built in **Python**, making use of two very popular libraries for AI development, **OpenCV** and **TensorFlow**.
The model is divided in two modules, one for detection during the day and another one during the night. Each module uses a different algorithm to ensure optimal detection in each of the different environments.
## Day Module
Makes use of a machine learning algorithm previously developed by a group of researchers in the United States. **https://github.com/abg3/Smoke-Detection-using-Tensorflow-2.2**

In this project, the algorithm is implemented using **TensorFlow**, being **OpenCV** the library in charge of handling the images so that the model can perform the detection, besides providing a visual detection for the user.

![day](https://github.com/IvoDur/deteccion-humo-rosario/assets/98555807/4b2e9db2-1629-453d-8c7d-bee25c8e79c2)

## Night Module
Due to the lack of light, it was necessary to implement a different detection model based on color in this module. **OpenCV** was used for this due to its high image handling capabilities.

![night](https://github.com/IvoDur/deteccion-humo-rosario/assets/98555807/1caae93b-2301-4db0-a4b5-508544950c8c)
