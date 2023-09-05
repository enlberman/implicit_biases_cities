# Code for "City Population, Majority Group Size, Residential Segregation, and Implicit Racial Biases in U.S. Cities"
 1. download raw iat data in csv fromat from https://osf.io/52qxl/ and save into the IAT_data folder
 2. run iat_filter_only_geotagged.py (this may take a few hours to run)
 3. run iat_data.py and iat_data_individual.py (this make take a few hours to run)
 4. the rest of the analysis and figure producing files can now be run (the main iat_figures.py file should run in about an hour)
 
This analysis software was developed using python 3.11 and the following package versions:
numpy 1.24.3
pandas 1.5.3
matplotlib 3.7.1
statsmodels 0.14.1
scipy 1.10.1
