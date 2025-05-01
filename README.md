# dafx25_antialiasing_neural

#### Requirements

First, clone this repo and the initialise the OpenAmp submodule:
```angular2html
git clone --recurse-submodules git@github.com:a-carson/dafx25_antialiasing_neural.git 
```

Create conda environment:
```
conda env create -t env.yaml
```

(optional) Download Proteus Tone Packs collection of pre-trained models:

```
curl -LO https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip
tar -xf Proteus_Tone_Packs.zip
rm Proteus_Tone_Packs.zip
```
(optional) Download the Neural Amp Modeler (NAM) weights which we used from the links below, and extract into `NAM` directory:

- Marshall JCM: https://www.tone3000.com/tones/marshall-jcm-800-2203-1071

- Vox AC15: https://www.tone3000.com/tones/vox-ac15-1080

#### Run training
Run fine tuning on Proteus Mesa model (default, Proteus Tone Packs required):
```
python3 train.py --config 0
```
or run fine tuning on one of our custom pre-trainedd models, e.g. the Broadcast TCN model:
```angular2html
python3 train.py --config 6
```
