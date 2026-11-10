from pathlib import Path
import csv
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)


def write_csv(name, columns, records):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(records)


def metrics(y, pred):
    residual = y - pred
    rmse = np.sqrt(np.mean(residual ** 2))
    return [
        np.corrcoef(y, pred)[0, 1],
        1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2),
        rmse,
        np.mean(np.abs(residual)),
        100 * np.mean(np.abs(residual / y)),
        np.std(y, ddof=1) / rmse,
    ]


observations, predictions = {}, {}
summary = []
for model_name in ('Direct', 'WS-CB', 'DO-Sum', 'DO-CBC'):
    with np.load(ROOT / 'inputs' / f'{model_name}.npz') as data:
        x_seq = torch.from_numpy(data['seq'])
        x_static = torch.from_numpy(data['static'])
        obs = data['observed'][126:].astype(float)
        if model_name == 'DO-CBC':
            observed_pools = data['observed_organ_pools'][126:].astype(float)
            observed_traits = data['observed_traits'][126:].astype(float)
    model = torch.jit.load(str(ROOT / 'models' / f'{model_name}.pt'), map_location='cpu').eval()
    with torch.inference_mode():
        output = model(x_seq, x_static).numpy()[126:].astype(float)
    pred = output[:, -1]
    observations[model_name] = obs
    predictions[model_name] = pred
    summary.append([model_name, *metrics(obs, pred)])

write_csv('Table2_spectral_texture.csv',
          ['Model', 'R', 'R2', 'RMSE', 'MAE', 'MAPE_percent', 'RPD'], summary)

docbc = torch.jit.load(str(ROOT / 'models' / 'DO-CBC.pt'), map_location='cpu').eval()
with np.load(ROOT / 'inputs' / 'DO-CBC.npz') as data:
    seq = torch.from_numpy(data['seq'])
    static = torch.from_numpy(data['static'])
with torch.inference_mode():
    docbc_pred = docbc(seq, static).numpy()[126:].astype(float)
organ_pred = docbc_pred[:, :3]
trait_pred = docbc_pred[:, 3:9]

errors = observed_pools - organ_pred
abs_shares = np.sum(np.abs(errors), axis=0)
abs_shares = abs_shares / abs_shares.sum()
squared_shares = np.sum(errors ** 2, axis=0)
squared_shares = squared_shares / squared_shares.sum()
pool_shares = observed_pools.sum(axis=0)
pool_shares = pool_shares / pool_shares.sum()

conc_err = (observed_traits[:, 0::2] - trait_pred[:, 0::2]) * observed_traits[:, 1::2]
biom_err = trait_pred[:, 0::2] * (observed_traits[:, 1::2] - trait_pred[:, 1::2])
component_biomass_share = np.abs(biom_err).sum(axis=0) / (np.abs(conc_err).sum(axis=0) + np.abs(biom_err).sum(axis=0))

write_csv('Organ_pathway_errors.csv',
          ['Organ', 'Observed_TSPU_share', 'Absolute_error_share',
           'Squared_error_share', 'Abs_error_share_divided_by_contribution',
           'Biomass_share_of_absolute_component_error'],
          [[name, pool_shares[i], abs_shares[i], squared_shares[i],
            abs_shares[i] / pool_shares[i], component_biomass_share[i]]
           for i, name in enumerate(('Leaf', 'Stem', 'Grain'))])

cancellation = 1 - np.abs(errors.sum(axis=1)) / np.abs(errors).sum(axis=1)
paired_improvement = np.abs(observations['Direct'] - predictions['Direct']) - \
                     np.abs(observations['DO-Sum'] - predictions['DO-Sum'])
print('Table 2 (spectral + texture):')
for name, r, r2, rmse, mae, mape, rpd in summary:
    print(f'{name:7s} R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}')
print(f'Grain absolute-pathway error share: {100 * abs_shares[2]:.2f}%')
print(f'Grain biomass-related component error share: {100 * component_biomass_share[2]:.1f}%')
print(f'DO-Sum improvement over Direct: {np.count_nonzero(paired_improvement > 0)}/54 plots')
print(f'Opposite-signed organ errors: {np.count_nonzero((errors.min(axis=1) < 0) & (errors.max(axis=1) > 0))}/54 plots')
print(f'At least 20% cancellation: {np.count_nonzero(cancellation >= 0.2)}/54 plots')
print(f'Median cancellation: {100 * np.median(cancellation):.1f}%')
