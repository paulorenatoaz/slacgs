# SLACGS

A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples in order to evaluate Trade Off Between Samples and Features sizes in Classification Problems on gaussian Samples.

* Reports with results will be stored in different Google Spreadsheet:  
    - Experiment Scenario
    - Custom Scenario
    - Custom Simulation.
* The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'                         
  owned by slacgs' google service account and shared with the user's Google Drive account.
* Also, images with data visualization will be exported to a local folder inside project's root folder ( slacgs/images/ )

Reports Exported (Google Spreadsheets):
    - Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
    - Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
    - Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

Images Exported (<user>/slacgs/images/ or /content/slacgs/images (for Gcolab) ):
    - Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
    - Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
    - Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

Loss Functions:
    - Theoretical Loss: estimated using probability theory
    - Empirical Loss with Train Data: estimated using empirical approach with train data
    - Empirical Loss with Test Data: estimated using empirical approach with test data

## Install cmd or terminal

'''bash
git clone https://github.com/paulorenatoaz/slacgs

pip install slacgs/

## Demo 

```python
from slacgs.demo import *







