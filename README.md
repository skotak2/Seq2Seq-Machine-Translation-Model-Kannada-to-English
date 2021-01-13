# Seq2Seq-Machine-Translation-Model-Kannada-to-English
Develop REST API to perform machine translation using Seq2Seq  model. The model deployment is done using google could platform. 

# English-to-Telugu-Translator

### TABLE OF CONTENTS
* [Objective](#objective)
* [Technologies](#technologies)
* [Algorithms](#algorithms)
* [Data](#data)
* [Implementation](#implementation)
* [Results](#results)

## OBJECTIVE 
1. Built a REST API to convert English sentences to Telugu using LSTM and Keras. 
2. Deploy the built model onto AWS while using Docker-container orchestration.

## TECHNOLOGIES
Project is created with: 
* Python - **pandas, keras, numpy, seaborn, sklearn, pickle**
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
Build a Flask application which can be access from local machine at the address http://127.0.0.1:5000/predict. To check if the application is working or not, we put "/" method where we are printing "Hello World!" in the address http://127.0.0.1:5000/.

![GitHub Logo](/images/Predictmethod.PNG)

The code file to build the Flask application is available here: https://github.com/VipanchiKatthula/English-to-Telugu-Translator/blob/master/code/main_program.py
![GitHub Logo](/images/Access.PNG)

Once the Flask application is up and running in the local PC. You can prepare to deploy it on AWS using Linux t2.micro server. We need to have the model weights, indexer pkl files as in this case, the end-users will provide text input which needs to be indexed before it is passed to the Keras model. The indexer pkl file could not be shared on github as it was occupying a space of 120 Mb. 

# AWS setup
* Create an AWS account if you don't already have one and you would have access to 650 hours of FREE access to Amazon's basic EC2 servers which are basically Virtual machines hosted on Amazon.
* Go to products, search for EC2

![GitHub Logo](/images/AWS1.png)

* Click on "Launch EC2 Instance"

![GitHub Logo](/images/AWS2Launch.png)

* Choose **Amazon Linux 2 AMI (HVM)**

![GitHub Logo](/images/AWS3AMI.png)

* Select the **t2.micro** instance in the shows AMIs

![GitHub Logo](/images/AWS4t2micro.png)

* Click on Review and Launch the server. 
![GitHub Logo](/images/AWS5ReviewLaunch.png)

* Add a new HTTP permission for the VM to be access from external traffic. You would be asked a Key-value pair which is like the password to access the Virtual Machine. Select the option to generate a new Key-value pair and keep it safe as you need it to access the VM from PC.

![GitHub Logo](/images/AWS6HTTP.png)

# Docker
The requirements.txt file is present [here](https://github.com/VipanchiKatthula/English-to-Telugu-Translator/blob/master/requirements.txt) which gives the list of requirements for the docker to be installed in the VM to host our Flask application.

Run the following commands in cmd prompt to connect to the VM:[docker_commands](https://github.com/VipanchiKatthula/English-to-Telugu-Translator/blob/master/docker_commands.txt) where the public-dns-name is in the format ec2–x–x–x–x.compute-1.amazonaws.com. 

You will be able to access the deployed model in the url: http://public-dns-name/predict
## RESULTS
The deployed model can be accessed from any python development tool like Jupyter Notebook or Spyder. As the data used for the model development was much smaller than the training data for Google translate, we were not able to achieve that level of accuracy. However, the deployed model can be improved upon and made perfect by adding advanced techniques like attention. 

## REFERENCES
* https://towardsdatascience.com/deploy-ml-models-at-scale-151204549f41
* https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf

