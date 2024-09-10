# MelodyBot

### Deployment to Telegram as a ChatBot
```python
python AutoReplyBot.py
```

### Build Environment
```bash
conda env create -f environment.yml
```

### Build Dictionary for Lyrics & Melodies
Please run the jupyter notebook file "./0_build_dictionary/0_build_dict_octuple.ipynb" will generate the used dictionary as music_dict.pkl under binary folder. 

### Binarise Lyrics and Melody Data

Run ./1_binarise_data/1_octuplemidi.ipynb to binarise the lyrics and melody dataset

### Training

```python
python ./2_train_model/3_train_bart_octuple.py
```


### Inference
For batch inference, please use "4_infer_bart_octuple.ipynb"
For custom single-sample inference, please use "4_infer_bart_octuple_single.ipynb"