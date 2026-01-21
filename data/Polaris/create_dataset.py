import numpy as np
import pandas as pd
from seshatdatasetanalysis.Template import Template as Template
from seshatdatasetanalysis.TimeSeriesDataset import TimeSeriesDataset as TSD

# categories to include in the dataset (used if the dataset is downloaded from the api)
categories = ['sc', # social complexity
              'wf', # warfare
              'id', # moralizing supernatural religion (MSP)
              'rel' # religion
              ]

load_template = False

if not load_template:
    template = Template(categories = categories, keep_raw_data=True)

    # using API downloads the latest data from the Seshat dataset, takes longer
    # reading from Polaris2025.xlsx uses a local copy of the dataset, faster
    use_api = False

    if use_api: 
        template = Template(categories = ['sc','wf','id','rel'], keep_raw_data=True)
        template.download_all_categories(add_to_template=False)
    else: 
        template.read_polaris('Polaris2025.xlsx')

    # creates the template and saves it as a CSV file
    template.template_from_dataset(use_new_method = True)
    template.save_dataset("template.csv")

# create dataset from template
dt = 100 # time step in years
dataset = TSD(categories = categories, template_path = "template.csv")
dataset.initialize_dataset_grid(-10000,1900,dt)

sampling_interpolation = 'zero' # 'linear' or 'zero' 
# sampling_interpolation determines how to interpolate data between known values
# 'linear' uses linear interpolation between known values
# 'zero' uses zero-order hold (step function), keeps the last known value until a new value is known
sampling_ranges = 'mean' # 'uniform' or 'mean' 
# sampling_ranges determines how to handle variable ranges (in numerical data like population size)
# 'uniform' samples a value uniformly within the range, changes the value each time the dataset is created
# if a variable range extends over multiple time steps, the sampling only happens once
# 'mean' uses the mean value of the range

dataset.download_all_categories(sampling_interpolation = sampling_interpolation,
                               sampling_ranges = sampling_ranges)

percentage_missing_allowed = 0.5
# when constructing complexity characteristics, if less than this percentage of data is present,
# the characteristic is not created and value is set to NaN 
# expects values between 0 and 1
dataset.build_social_complexity(allow_missing = percentage_missing_allowed)
# currently warfare and MSP characteristics are always built, regardless of missing data percentage
# warfare variables follow the strong evidence rule, if the value is not known, it is set to 0
# only when no data is avaliable is the value set to NaN
dataset.build_warfare()
dataset.build_MSP()

imputation = 'together' # 'separately' or 'together'

if imputation == 'separately':
    # this is our current imputation method, where Scale variables and Comp
    # variables are imputed separately
    dataset.scv['Hierarchy_sq'] = dataset.scv['Hierarchy'] ** 2
    scale_cols = ['Pop','Terr','Cap','Hierarchy', 'Hierarchy_sq']
    non_scale_cols = ['Government', 'Infrastructure', 'Information', 'Money']
    dataset.impute_missing_values(columns = scale_cols, add_resid = False)
    dataset.impute_missing_values(columns = non_scale_cols, add_resid = False)
elif imputation == 'together':
    # this method imputes all complexity variables together, this method was used on Equinox
    all_complexity_cols = ['Pop','Terr','Cap','Hierarchy', 
                           'Government', 'Infrastructure', 'Information', 'Money']
    dataset.impute_missing_values(columns = all_complexity_cols, add_resid = False)

# compute PCA
pca_cols = ['Pop','Terr','Cap','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
dataset.compute_PCA(cols = pca_cols, col_name = 'PC', n_cols = 2,  n_PCA= len(pca_cols))
# creates columns PC_1 and PC_2 in dataset.scv_imputed for PC components
# col_name specifies the prefix for the new columns
# n_cols specifies how many principal components to save in the dataset
# n_PCA specifies how many principal components to compute

dataset.save_dataset(path='', name='polaris_final')