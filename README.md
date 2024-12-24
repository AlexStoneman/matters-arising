To set up the environment:
```
conda env create -f environment.yml
```

To train and evaluate AGILE models: 
```
cd Matters_Arising_AGILE
python run_HeLa_finetuning.py <run_name> 10 8 <device>
python run_RAW_finetuning.py <run_name> 10 8 <device>
```
Models and predictions will appear in the directory `finetune/<run_name>_<split>_<cell>`.
Analysis can be done via:
```
python run_finetuning_analysis.py <run_name>_<split>_<cell> 8 10 --<split>
```
Use `--savepreds` to save ensemble predictions for the Murcko split.

To train and evaluate baseline models:
```
cd Matters_Arising_baselines
python run_HeLa_baselines.py <run_name> 10 <device>
python run_RAW_baselines.py <run_name> 10 <device>
```
Models and predictions will appear in the directory `<run_name>/<model>/<features>/<cell>/<split>`.
Analysis can be done via:
```
python analyze_baselines.py <run_name> 10 
```

All model results from the manuscript are contained in the repository and summarized in `Matters_Arising_Data.xlsx`.
