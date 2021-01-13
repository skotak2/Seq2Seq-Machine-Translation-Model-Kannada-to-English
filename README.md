# Seq2Seq-Machine-Translation-Model-Kannada-to-English
Develop REST API to perform machine translation using Seq2Seq  model. The model deployment is done using google could platform. 

# Kannada-to-English-Translator

### TABLE OF CONTENTS
* [Objective](#objective)
* [Technologies](#technologies)
* [Algorithms](#algorithms)
* [Data](#data)
* [Implementation](#implementation)
* [Results](#results)

## OBJECTIVE 
1. Built a REST API to convert Kannada sentences to English using GRU and pyTorch. 
2. Deploy the model on Google cloud platform

## TECHNOLOGIES
Project is created with: 
* Python - **pandas, pyTorch, numpy, seaborn, sklearn, pickle**
* Google Cloud Platform - **cloud functions**
* Flask

## ALGORITHMS
* Autoencoder - Decoder 
* Gated Recurrent Unit (GRU)
* Attention Mechanism
* GET - POST methodology

## DATA
The data for this project is available as text file on [Data Source](https://www.manythings.org/anki/), where each line has a sentence in kannada and translation of it in english with space delimiter. We manually verified randomly to ensure that each example made sense.

## IMPLEMENTATION

# Modeling
First we build the encoder decoder model, with attention mechanism using GRU RNN. The training was done using Python script available [Here](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Code/Kannada_to_English_Machine_Translation.ipynb)

# App development
Build a Flask application which can be access from local machine at the address http://127.0.0.1:5000/predict. 

# Deployment

* We will use the [Script](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Code/Kannada_to_English_Machine_Translation.ipynb) to train the model. After training the model, we will save the model weights in a .pt file and store in google cloud storage. We also build the vocabulary dictionary by indexing each word to a number and pickle them. These pickle files are also stored in storage file. You can access them [here](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/tree/main/Storage%20Files) Once these files are in place, the deployment can be done following the steps below


* We will upload the files on a storage bucket. To Create a bucket using following options as highlighted with following specifications

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture1.png)

* Use the highlighted options to upload the .pt and .pkl files and to configure permissions. Here we set the files to public access, however you could set it to your requirements.

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture2.png)

* Now we use the cloud functions capability to deploy the code the code for the Flask API and access the weights and vocabulary dictionary from the storage. 

For creating the cloud function, browse for it on the GCP platform and use the options highlighted to below to create a function,


![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture3.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture4.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture5.png)

*Allocation of 1 GiB memory is recommended. Once set, click on ‘Next’ and deploy the code on the cloud function console. 

To deploy the code, first configure the console with the below highlighted settings and prepare the environment using the requirements file (this is equivalent to pip install {library}) as described below, 

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture6.png)

* Once Requirement is set with the above libraries, prepare the main.py script for deployment. The script has the api_request(x) function defined for returning the desired output given the input – ‘x’, from an external source. The code is uploaded above with name “main.py”. Once the code is arranged click on deploy.

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture7.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture8.png)

* Once deployment is complete, click on the cloud function, using TESTING option to debug for deployment errors. Once the input is passed in the below format, test the function, and look for the desired output.

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture9.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture10.png)


## RESULTS
The deployed model can be accessed from the url from any system to translate kannada sentences to english. 

## REFERENCES
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://pytorch.org/tutorials/beginner/saving_loading_models.html

