## Overview

Web app for [Udacity's Dog Breed Classifier Project](https://github.com/udacity/dog-project.git).

From a Convolutional Neural Network trained on 133 different dog breeds it: 

* If a dog image is selected: 
    * Determine which dog breed it most likely is. 
* If a human face image is selected: 
    * Determine which dog breed is most resembling.

In both cases the app displays:
* The five most likely dog breeds with associated probabilities
* An image of the dog breed with the highest probability alongside the selected image

![demo-1](/doc/img/index.png)

![demo-2](/doc/img/result-dog.png)

## Instructions

1. Clone the repository.
```	
git clone https://github.com/o1sa/dog-project-app.git
cd dog-project-app
```

2. Create and activate a new virtual environment.
```
python3 -m virtualenv venv
source venv/bin/activate
```

3. Download the dependencies using pip.
```
pip install -r requirements.txt
```

4. Start the app locally 

* either by

```
python dog-project.py
```

* or using the flask run command

```
export FLASK_DEBUG=1
export FLASK_ENV=development
export FLASK_APP=dog-project.py
flask run
```

5. Open a web browser and navigate to the specified url.