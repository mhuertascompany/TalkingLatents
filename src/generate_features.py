import yaml
from data.transforms import *
from data.dataset import SpectraDataset
import pandas as pd
from nn.models import *
from nn.train import *
from util.visualization import *
from util.utils import *
from nn.optim import CQR
from torch.utils.data import DataLoader
import datetime
import argparse

# these are just for my convenient. The actual paths are given as arguments
config_path = 'pretrained_models/MultiTaskRegressor_spectra__decode_1_complete_config.yaml'
weights_path = 'pretrained_models/spectra.pth'
lamost_data_dir = r"C:\Users\Ilay\projects\kepler\data\lamost\data"
simulation_data_dir = "data/dataset_noiseless/lamost"
simulation_df_path = "data/dataset_noiseless/simulation_properties.csv"
lamost_df_path = r"C:\Users\Ilay\projects\kepler\data\lamost\lamost_local_catalog.csv"


def get_lamost_df(df_path):
    lamost_catalog = pd.read_csv(df_path)
    lamost_catalog = lamost_catalog.drop_duplicates(subset=['obsid'])
    lamost_catalog = lamost_catalog[lamost_catalog['snrg'] > 0]
    lamost_catalog.rename(columns={'teff': 'Teff', 'feh':'FeH'}, inplace=True)
    lamost_catalog = lamost_catalog.dropna(subset=['Teff', 'logg', 'FeH'])
    print("values ranges: ")
    for c in ['Teff', 'logg', 'FeH']:
        if c == 'Teff':
            lamost_catalog[c] = lamost_catalog[c] / 5778  # match the pretraining normalization
        print(c, lamost_catalog[c].min(), lamost_catalog[c].max())
    return lamost_catalog

def get_simulation_df(df_path):
    sim_df = pd.read_csv(df_path)
    for c in ['Teff', 'logg', 'FeH']:
        if c == 'Teff':
            sim_df[c] = sim_df[c] / 5778  # match the pretraining normalization
        print(c, sim_df[c].min(), sim_df[c].max())
    return sim_df

def load_model(config):
    model = MultiTaskRegressor(Container(**config['model_args']), Container(**config['conformer_args']))
    return model

def predict(model, test_dl, device, max_iter=100):
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
                                     accumulation_step=1, max_iter=max_iter, w_name=None,
                                     w_init_val=1, exp_name=f"predict")

    return trainer.predict(test_dl, device=device)


def test_predictions(log_dir, quantile_labels=['Teff', 'logg', 'FeH'],
                     umap_labels=['Teff', 'logg', 'FeH'],
                     savedir='figs'
                     ):
    info_df = pd.read_csv(os.path.join(log_dir, 'info.csv'))
    preds = np.load(os.path.join(log_dir, 'preds.npy'))
    targets = np.load(os.path.join(log_dir, 'targets.npy'))
    features = np.load(os.path.join(log_dir, 'features.npy'))
    xs = np.load(os.path.join(log_dir, 'xs.npy'))
    decodes = np.load(os.path.join(log_dir, 'decodes.npy'))

    plot_quantiles(targets, preds, [0.1,0.25,0.5,0.75,0.9], [5778, 1, 1], quantile_labels, savedir=savedir)
    plot_decode(xs, decodes, info_df, num_sampels=10, savedir=savedir)
    plot_umap(features, info_df, umap_labels, savedir=savedir)


def run(config_path, weights_path, data_dir, df_path, simulation=True, max_iter=100):
    cur_time = datetime.date.today().strftime("%Y-%m-%d")
    if simulation:
        cur_time = cur_time + "-simulation"
    savedir = f'logs/{cur_time}/figs'
    os.makedirs(f'logs/{cur_time}', exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = yaml.safe_load(open(config_path, 'r'))
    model = load_model(cfg)
    model = load_checkpoints_ddp(model, weights_path)
    model.to(device)
    model = model.eval()
    train_df = get_simulation_df(df_path) if simulation else get_lamost_df(df_path)
    rv_norm = not simulation
    file_type = 'pqt' if simulation else 'fits'
    id = 'Simulation Number' if simulation else 'obsid'
    labels = ['Teff', 'logg', 'FeH']
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=rv_norm), ToTensor()])
    train_ds = SpectraDataset(data_dir,
                              transf, train_df,
                              max_len=4096,
                              id=id,
                              labels=labels,
                              file_type=file_type)
    train_dl = DataLoader(train_ds,
                          batch_size=16,
                          collate_fn=kepler_collate_fn,
                          )
    preds, targets, features, xs, decodes, info = predict(model, train_dl, device, max_iter=max_iter)
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

    test_predictions(f'logs/{cur_time}', savedir=savedir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specta Feature extraction.')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to config file of the model')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to weights file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data folder')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='Path to csv with meta data (stellar parameters, ids, etc.)')
    parser.add_argument('--simulation', type=bool, default=False,
                        help='Run on simulated data (default: False)')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Number of iterations (default: 100)')

    args = parser.parse_args()


    run(args.model_config, args.weights_path, args.data_dir, args.meta_file,
        simulation=args.simulation, max_iter=args.max_iter)