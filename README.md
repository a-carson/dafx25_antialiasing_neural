# dafx25_antialiasing_neural

! Minimum working version, more updates and documentation to follow !
#### Requirements

Create environment:
```
conda env create -t env.yaml
```


(optional) Download Proteus Tone Packs collection of pre-trained models:

```
curl -LO https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip
tar -xf Proteus_Tone_Packs.zip
rm Proteus_Tone_Packs.zip
```

#### Run training
Run fine tuning on Proteus Mesa model (default, Proteus Tone Packs required):
```
python3 train.py --config 0
```
or run fine tuning on one of our custom pre-trainedd models, e.g. the Broadcast TCN model:
```angular2html
python3 train.py --config 6
```
