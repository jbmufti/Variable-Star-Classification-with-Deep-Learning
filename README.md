# Classification of Variable Stars with Machine Learning

Sage Santomenna and Jalal Mufti (The Multimodal Maestros)


Project Title: Classification of Variable Stars with Machine Learning


For our project, we are training a model that can classify different types of variable stars, which are stars whose brightness appears to change over time. To train this model, we’re using the ASASSN dataset, a collection of approximately 378,000 lightcurves (measurements of the star’s brightness over a period of time) categorized by 19 different classifications. To prepare this data for training, we downsampled overrepresented classes, normalized the data by distance, and dropped data points that were extreme outliers. Our ultimate goal is to have a model with solid performance across multiple varying datasets, which could be used to identify interesting candidates for further observation.

# Configuration
Edit the config.toml to match your setup. The report indicates which paper the ASAS-SN light curve dataset can be retrieved from.  

# Milestone 2
Running notebooks `deepl_big_model.ipynb` and `deepl_small_model.ipynb` will run the baseline small and large models, assuming you have data access. The outputs of the notebooks are visible in the repository.


# Notebooks
- `deepl_big_model.ipynb`: original MLP model with poor performance
- `deepl_in_between_model.ipynb`: incrementally-improved MLP model: half as many dense layers, reduced dropout rate, and smaller batch size than the original.
- `deepl_mlp_model_final.ipynb`: final MLP model.
- `rnn_model.ipynb`: ill-fated first attempt at RNN model
- `subseq_bidirectional_rnn_model.ipynb` - ill-fated second attempt at RNN model

# /data
Contains data for running the MLP and random forest models. Must be unzipped according to instructions in data/README.md

# /predictions
Contains examples of predictions made by the MLP model
