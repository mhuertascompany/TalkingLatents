import torch

import yaml
from utils import *
from data.transforms import *
from data.dataset import SpectraDataset, SimulationDataset
import pandas as pd
from nn.models import *
from nn.train import *
from visualization import *
from nn.optim import CQR
import datetime


def get_lamost_dfs():
    lamost_catalog = pd.read_csv(r"C:\Users\Ilay\projects\kepler\data\lamost\lamost_local_catalog.csv")
    lamost_catalog = lamost_catalog.drop_duplicates(subset=['obsid'])
    lamost_catalog = lamost_catalog[lamost_catalog['snrg'] > 0]
    lamost_catalog = lamost_catalog.dropna(subset=['teff', 'logg', 'feh'])
    print("values ranges: ")
    for c in ['teff', 'logg', 'feh']:
        if c == 'teff':
            lamost_catalog[c] = lamost_catalog[c] / 5778  # match the pretraining normalization
        print(c, lamost_catalog[c].min(), lamost_catalog[c].max())
    return lamost_catalog

def load_model(config):
    model = MultiTaskRegressor(Container(**config['model_args']), Container(**config['conformer_args']))
    return model

def predict(test_dl, device):
    loss_fn = CQR(quantiles=[0.1,0.25,0.5,0.75,0.9], reduction='none')
    ssl_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,
                                 weight_decay=1e-6)

    trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                                     criterion=loss_fn, ssl_criterion=ssl_loss_fn,
                                     output_dim=3, scaler=None,
                                     scheduler=None, train_dataloader=None,
                                     val_dataloader=None, device=device,
                                     num_quantiles=5,
                                     exp_num='local', log_path=None, range_update=None,
                                     accumulation_step=1, max_iter=1000, w_name='snrg',
                                     w_init_val=1, exp_name=f"predict")

    return trainer.predict(test_dl, device=device)


def test_predictions(log_dir):
    info_df = pd.read_csv(os.path.join(log_dir, 'info.csv'))
    preds = np.load(os.path.join(log_dir, 'preds.npy'))
    targets = np.load(os.path.join(log_dir, 'targets.npy'))
    features = np.load(os.path.join(log_dir, 'features.npy'))
    xs = np.load(os.path.join(log_dir, 'xs.npy'))
    decodes = np.load(os.path.join(log_dir, 'decodes.npy'))


    plot_quantiles(targets, preds, [0.1,0.25,0.5,0.75,0.9], [5778, 1, 1], ['Teff', 'Logg', 'FeH'])
    plot_decode(xs, decodes, info_df, num_sampels=10)
    plot_umap(features, info_df, ['Teff', 'Logg', 'FeH', 'snrg', 'RV'])

if __name__ == '__main__':
    cur_time = datetime.date.today().strftime("%Y-%m-%d")
    os.makedirs(f'logs/{cur_time}', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = 'pretrained_models/MultiTaskRegressor_spectra__decode_1_complete_config.yaml'
    weights_path = 'pretrained_models/spectra.pth'
    lamost_data_dir = r"C:\Users\Ilay\projects\kepler\data\lamost\data"
    cfg = yaml.safe_load(open(config_path, 'r'))
    model = load_model(cfg)
    model = load_checkpoints_ddp(model, weights_path)
    model.to(device)
    model = model.eval()
    train_df = get_lamost_dfs()
    transf = Compose([GeneralSpectrumPreprocessor(), ToTensor()])
    train_ds = SpectraDataset(lamost_data_dir,transf, train_df, max_len=4096, labels=['teff', 'logg', 'feh'])
    train_dl = DataLoader(train_ds,
                          batch_size=16,
                          collate_fn=kepler_collate_fn,
                          )
    preds, targets, features, xs, decodes, info = predict(train_dl, device)
    print(preds.shape, targets.shape, features.shape, xs.shape, decodes.shape, info.keys())
    if info:
        max_length = max(len(v) for v in info.values())
        for key in info:
            current_length = len(info[key])
            if current_length < max_length:
                # Pad with NaN for numeric data, or None for mixed data
                info[key].extend([np.nan] * (max_length - current_length))

    info_df = pd.DataFrame(info)
    info_df.to_csv(f'logs/{cur_time}/info.csv', index=False)
    np.save(f'logs/{cur_time}/preds.npy', preds)
    np.save(f'logs/{cur_time}/targets.npy', targets)
    np.save(f'logs/{cur_time}/features.npy', features)
    np.save(f'logs/{cur_time}/xs.npy', xs)
    np.save(f'logs/{cur_time}/decodes.npy', decodes)

    test_predictions(f'logs/{cur_time}')