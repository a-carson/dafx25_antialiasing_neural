import matplotlib.pyplot as plt
import pytorch_lightning as pl
import scipy.signal.windows
import torch
import torchaudio
from pytorch_lightning import LightningModule
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenAmp'))
from Open_Amp.amp_model import AmpModel
from torch import optim
from dataloader import SineToneDataset, SequenceDataset
from torch.utils.data import DataLoader
from spectral import cheb_fft, bandlimit_batch, PerceptualFIRFilter
import os
import matplotlib as mpl
import numpy as np
import wandb
import argparse
import time
from nmr import NMR
from config import get_config
#mpl.use('macosx')
from scipy.signal import freqz

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
parser.add_argument("--config", type=float, default=0)

args = parser.parse_args()

def dB10(x):
    return 10 * torch.log10(x).cpu().numpy()

def linear10(x):
    return 10 ** (x/10)

conf = get_config(args.config)
model = AmpModel(conf['model_json'], conf['model_name'])
print(model)
# w, H = freqz(self.aweight_fir.fir.weight[0,0,:].numpy())
# plt.semilogx(w, 20 * np.log10(np.abs(H)))
# plt.show()

class AARNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.model.double()
        self.aweight_fir = PerceptualFIRFilter(filter_type='aw')
        self.nmr = NMR(fs=conf['sample_rate'])

        if conf['pre_emph'] == 'lp':
            self.lpf = PerceptualFIRFilter(filter_type='lp')
        elif conf['pre_emph'] == 'aw':
            self.lpf = PerceptualFIRFilter(filter_type='aw')
        else:
            self.lpf = lambda a: a

        self.val_loss = []
        self.val_loss_epoch = self.new_val_loss_metrics()
        self.automatic_optimization = False
        self.batch_load_start_time = time.time()
        self.audio_log_counter = 0


    def new_val_loss_metrics(self):
        return {
            'mesr': {},
            'esr': {},
            'asr': {},
            'audio_esr': {},
            'esr_normal': {},
            'esr_lpf': {},
            'nmr': {}
        }

    def loss_function(self, targ, pred, f0):

        targ = self.lpf(targ)
        pred = self.lpf(pred)

        loss = pred.new_zeros(1)
        metrics = {
            'train/step': self.global_step,
        }

        if conf['loss_weights']['nmr'] > 0:
            nmr, nmr_dB = self.nmr(pred, targ)
            nmr_mean = linear10(nmr_dB.mean())
            loss += conf['loss_weights']['nmr'] * nmr_mean
            metrics['train/nmr'] = nmr_mean

        if conf['loss_weights']['esr_normal'] > 0:
            esr_normal = torch.sum((pred - targ) ** 2) / torch.sum(targ ** 2)
            loss += conf['loss_weights']['esr_normal'] * esr_normal
            metrics['train/esr_normal'] = esr_normal

        if conf['loss_weights']['dc'] > 0:
            dc = torch.mean(targ - pred) ** 2 / torch.mean(targ**2)
            loss += conf['loss_weights']['dc'] * dc
            metrics['train/dc'] = dc

        # BANDLIMIT
        bandlimit = conf['loss_weights']['mesr'] > 0 and conf['loss_weights']['esr'] > 0 and conf['loss_weights'][
            'asr'] > 0
        if bandlimit:
            y_pred_bl, aliases = bandlimit_batch(pred.squeeze(-1), f0, conf['sample_rate'])

            # MESR
            Y_bl = cheb_fft(targ).abs()
            Y_pred_bl = cheb_fft(y_pred_bl, dim=1).abs()
            mesr = torch.sum((Y_pred_bl - Y_bl) ** 2) / torch.sum(Y_bl ** 2)

            # ESR
            esr = torch.sum((y_pred_bl - targ) ** 2) / torch.sum(targ ** 2)

            # ASR
            asr = torch.sum(aliases ** 2) / torch.sum(y_pred_bl ** 2)

            loss += conf['loss_weights']['mesr'] * mesr \
                    + conf['loss_weights']['asr'] * asr \
                    + conf['loss_weights']['esr'] * esr

            metrics = metrics | {
                'train/mesr': mesr,
                'train/esr': esr,
                'train/asr': asr,
                'train/total': loss,
            }

        metrics['train/total'] = loss

        return loss, metrics

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        x, y, f0, dB = batch
        run.log({
                'batch_no': batch_idx + len(train_loader) * self.current_epoch,
                'batch_load_time': time.time() - self.batch_load_start_time})

        batch_train_start_time = time.time()
        y, _ = bandlimit_batch(y.squeeze(-1), f0, conf['sample_rate'])
        warmup_samples = x.shape[1] - y.shape[-1]

        if self.model.model_class == 'SimpleRNN':
            # warmup step
            self.model.model.reset_state()
            self.model(x[:, :warmup_samples, :])
            x = x[:, warmup_samples:, :]                   # remove warmup samples
            warmup_samples = 0

        # tbptt
        num_frames = int(np.floor(y.shape[-1] / conf['tbptt_steps']))
        for n in range(num_frames):
            opt.zero_grad()
            start = conf['tbptt_steps'] * n
            end = conf['tbptt_steps'] * (n + 1)
            x_frame = x[:, start:warmup_samples+end, :]    # keep warmup samples if WaveNet
            y_frame = y[:, start:end]

            start_time = time.time()
            y_pred = self.model(x_frame).squeeze(-1)
            y_pred = y_pred[:, -conf['tbptt_steps']:]       # ensure only one frame-size left
            model_time = time.time() - start_time

            # mpl.use('macosx')
            # Y_pred = np.fft.rfft(y_pred[0, :].detach().numpy() * scipy.signal.windows.chebwin(conf['tbptt_steps'], at=-120))
            # Y = np.fft.rfft(y_frame[0,:].detach().numpy() * scipy.signal.windows.chebwin(conf['tbptt_steps'], at=-120))
            # plt.plot(20*np.log10(np.abs(Y_pred))), plt.plot(20*np.log10(np.abs(Y))), plt.show()
            # plt.plot(y_pred[0, :].detach().numpy()), plt.plot(y_frame[0,:].detach().numpy()), plt.show()

            loss, metrics = self.loss_function(y_frame, y_pred, f0)

            metrics['model_time'] = model_time
            self.manual_backward(loss)
            opt.step()

            if self.model.model_class == 'SimpleRNN':
                self.model.model.detach_state()

            run.log(metrics)


        run.log({
            'batch_no': batch_idx + len(train_loader) * self.current_epoch,
            'batch_train_time': time.time() - batch_train_start_time})

        self.batch_load_start_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 0:
            x, y, f0, dB = batch
            y, _ = bandlimit_batch(y.squeeze(-1), f0, conf['sample_rate'])

            batch_size = x.shape[0]

            if self.model.model_class == 'WaveNet':
                y_pred = self.model(x).squeeze(-1)
            else:
                self.model.model.reset_state()
                y_pred = self.forward_frames(x, conf['tbptt_steps']).squeeze(-1)
            y_pred = y_pred[:, -conf['sample_rate']:]
            y_pred_bl, aliases = bandlimit_batch(y_pred, f0, conf['sample_rate'])


            Y_bl = cheb_fft(y).abs()
            Y_pred = cheb_fft(y_pred).abs()
            Y_pred_bl = cheb_fft(y_pred_bl, dim=1).abs()

            nmr, _ = self.nmr(y_pred, y)
            mesr = torch.sum((Y_pred_bl - Y_bl) ** 2, dim=-1) / torch.sum(Y_bl ** 2, dim=-1)

            esr = torch.sum((y_pred_bl - y) ** 2, dim=-1) / torch.sum(y ** 2, dim=-1)
            asr = torch.sum(aliases ** 2, dim=-1) / torch.sum(y_pred_bl ** 2, dim=-1)
            esr_normal = torch.sum((y_pred - y) ** 2, dim=-1) / torch.sum(y ** 2, dim=-1)
            esr_lpf = torch.sum(self.lpf(y_pred - y)**2, dim=-1) / torch.sum(self.lpf(y)**2, dim=-1)

            for b in range(batch_size):
                f0_idx = int(f0[b, ...].squeeze())
                self.val_loss_epoch['esr'][f0_idx] = esr[b]
                self.val_loss_epoch['mesr'][f0_idx] = mesr[b]
                self.val_loss_epoch['asr'][f0_idx] = asr[b]
                self.val_loss_epoch['esr_normal'][f0_idx] = esr_normal[b]
                self.val_loss_epoch['esr_lpf'][f0_idx] = esr_lpf[b]
                self.val_loss_epoch['nmr'][f0_idx] = nmr[b]



                if args.wandb and f0_idx > 3000:
                    plt.plot(2 * dB10(Y_pred[b, ...]))
                    plt.ylim([-60, 80])
                    run.log({f'spectra/{f0_idx}': plt})

        else:
            x, y = batch
            if self.model.model_class == 'WaveNet':
                y_pred = self.model(x)
            else:
                self.model.model.reset_state()
                y_pred = self.forward_frames(x, conf['tbptt_steps'])
            y = y[:, conf['tbptt_steps']:, :]
            y_pred = y_pred[:, conf['tbptt_steps']:, :]

            if args.wandb:
                if self.audio_log_counter % 4 == 0:
                    if batch_idx < 5:
                        key = f'Audio/clip_{batch_idx}'
                        wandb.log({key: wandb.Audio(y_pred[-1, :, 0].cpu().numpy(),
                                                                  sample_rate=conf['sample_rate']),
                                   'epoch': -1 if self.global_step == 0 else self.current_epoch},
                                  )
                    self.audio_log_counter += 1

            # compute loss
            y = self.aweight_fir(y)
            y_pred = self.aweight_fir(y_pred)
            esr = torch.sum((y_pred - y) ** 2) / torch.sum(y ** 2)

            self.val_loss_epoch['audio_esr'][batch_idx] = esr

    def forward_frames(self, x, frame_size):
        num_frames = int(np.floor(x.shape[1] / frame_size))
        y_pred = []
        for n in range(num_frames):
            start = frame_size * n
            end = frame_size * (n + 1)
            x_frame = x[:, start:end, :]
            y_pred_frame = self.model(x_frame)
            y_pred.append(y_pred_frame)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred

    def on_validation_end(self) -> None:

        metrics = {
            'epoch': -1 if self.global_step == 0 else self.current_epoch,
        }
        f0_values = np.array(list(self.val_loss_epoch['esr'].keys()))

        for metric_name, metric_dict in self.val_loss_epoch.items():
            values_T = torch.stack(list(metric_dict.values()))

            metrics['val/' + metric_name + '_mean'] = values_T.mean()
            metrics['val/' + metric_name + '_max'] = values_T.max()
            metrics['val/' + metric_name + '_mean_dB'] = dB10(values_T.mean())
            metrics['val/' + metric_name + '_max_dB'] = dB10(values_T.max())

            if metric_name in ['nmr', 'asr', 'esr_normal']:
                plt.semilogx(f0_values, dB10(values_T), label=metric_name)

        plt.ylim([-120, 0])
        plt.xlabel('Freq')
        plt.ylabel('[dB]')
        plt.legend()
        metrics['val/metrics_vs_f0'] = plt

        print(metrics)
        run.log(metrics)

        if args.wandb:
            epoch = metrics['epoch']
            json_path = os.path.join(run.dir, 'json')
            os.makedirs(json_path, exist_ok=True)
            self.model.export(dir=json_path, to_append=f'_epoch={epoch}')

        self.val_loss_epoch = self.new_val_loss_metrics()


    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=conf['lr'])
        return optimizer

if __name__ == '__main__':
    print(f'cpu cores - {os.cpu_count()}')
    pl_model = AARNN()


    if torch.cuda.is_available():
        accelerator = 'gpu'
        num_workers = 10
        persistent_workers = True
        pin_memory = True
    else:
        accelerator = 'cpu'
        num_workers = 0
        persistent_workers = False
        pin_memory = False

    torch.manual_seed(0)
    train_loader = DataLoader(
        SineToneDataset(device=conf['model_json'],
                        sample_rate=conf['sample_rate'],
                        **conf['train_data']),
        batch_size=conf['batch_size']['train'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory)

    val_loader = DataLoader(
        SineToneDataset(device=conf['model_json'],
                        sample_rate=conf['sample_rate'],
                        **conf['val_data']),
        batch_size=conf['batch_size']['val'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=False
    )

    if args.wandb:
        run = wandb.init(project='aa_rnn', config=conf)
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("plot/*", step_metric="epoch")
        run.define_metric("spectra/*", step_metric="epoch")
    else:
        class DummyLogger():
            def log(self, to_log):
                return
        run = DummyLogger()


    in_audio, audio_sample_rate = torchaudio.load('../../audio_datasets/dist_fx_192k/44k/val/input.wav')

    if audio_sample_rate != conf['sample_rate']:
        from scipy.signal import resample
        in_audio = torch.from_numpy(resample(in_audio.numpy().squeeze(),
                                    int(conf['sample_rate']/audio_sample_rate * in_audio.shape[-1]))).unsqueeze(0)

    gain = 10 ** (conf['audio_val_data']['dB']/20)
    in_audio = gain * in_audio.to(torch.double)


    audio_loader = DataLoader(
        SequenceDataset(input=in_audio,
                        device=conf['model_json'],
                        sequence_length=int(conf['audio_val_data']['dur'] * conf['sample_rate'])),
        batch_size=conf['batch_size']['audio_val']
    )
    trainer = pl.Trainer(max_epochs=conf['max_epochs'], accelerator=accelerator,
                         num_sanity_val_steps=0)
    trainer.validate(model=pl_model, dataloaders=[val_loader, audio_loader])
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=[val_loader, audio_loader])
    trainer.test(model=pl_model, dataloaders=[val_loader, audio_loader])
