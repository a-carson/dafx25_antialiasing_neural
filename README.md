# dafx25_antialiasing_neural
Accompanying code and weights for the DAFx25 paper "Anti-aliasing of neural distortion effects via model fine-tuning" by Alistair Carson, Alec Wright and Stefan Bilbao.
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
mv Proteus_Tone_Packs weights/
```
(optional) Download the Neural Amp Modeler (NAM) weights (TCN models) which we used from the links below, and extract into `weights/` directory:

- Marshall JCM: https://www.tone3000.com/tones/marshall-jcm-800-2203-1071

- Vox AC15: https://www.tone3000.com/tones/vox-ac15-1080

#### Run training
Run fine tuning anti-aliasing training:
```
python3 train.py --config <config_idx>
```
where `config_idx` can be chosen from the following list (default: 0)
```angular2html
config_idx      Target Pedal/Amp        Model          Initial weights dir
---------------------------------------------------------------------------------------------------
0               Mesa Mini Rectifier     LSTM           weights/Proteus_Tone_Packs/AmpPack1/MesaMiniRec_HighGain_DirectOut.json
1               Goat Pedal              LSTM           weights/Proteus_Tone_Packs/PedalPack1/GoatPedal_HighGain.json
2               Vox AC15                TCN            weights/NAM/Vox AC15/Vox AC15CH Overdriven Normal.nam
3               Marshall JCM            TCN            weights/NAM/Marshall JCM 800 2203/JCM800 2203 - P5 B5 M5 T5 MV7 G10 - AZG - 700.nam
4               Hudson Broadcast        LSTM           weights/pre-trained-by-us/broadcast_lstm_teacher.json
5               JHM8 Fuzz               LSTM           weights/pre-trained-by-us/jhm8_lstm_teacher.json 
6               Hudson Broadcast        TCN            weights/pre-trained-by-us/broadcast_tcn_teacher.nam
7               JHM8 Fuzz               TCN            weights/pre-trained-by-us/jhm8_tcn_teacher.nam
```

#### Citation
If using this work please use the following citation:
```angular2html
@inproceedings{Carson2025_antialiasing,
    author = {Carson, A. and Wright, A. and Bilbao, S.},
    title = {Anti-aliasing of neural distortion effects via model fine-tuning},
    booktitle = {Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25)},
    year = {2025},
    month = {Sept.},
    address = {Ancona, Italy}
}
```