import os, yaml
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_nematic import ResidualsNematic

name = 'run_1'
wandb_track = False # set to True to track training with wandb
os.chdir('./PIDM-Nematics')
load_model_flag = False # set to True to load a model
if load_model_flag:
    name = 'your_pretrained_model'
    load_path = './trained_models/' + name
    load_model_step = 0
    config = yaml.safe_load(Path(load_path, 'model', 'model.yaml').read_text())
else:
    config = yaml.safe_load(Path('model.yaml').read_text())

# diffusion parameters
if config['x0_estimation'] == 'mean':
    use_ddim_x0 = False
elif config['x0_estimation'] == 'sample':
    use_ddim_x0 = True
ddim_steps = config['ddim_steps']
residual_grad_guidance = config['residual_grad_guidance'] # gradient guidance scale as in https://www.sciencedirect.com/science/article/pii/S0021999123000670
# residual corrections (can be changed after training since only affects inference) similar to https://arxiv.org/abs/2312.10527
gov_eqs = config['gov_eqs']
fd_acc = config['fd_acc'] # finite difference accuracy
c_data = config['c_data']
c_residual = config['c_residual']
diff_steps = config['diff_steps']
use_dynamic_threshold = False
self_condition = False

# evaluation params
test_eval_freq = 500
sample_freq = 20000
full_sample_freq = 100000
ema_start = 1000
ema = EMA(0.99)
use_double = False
no_samples = 8
save_output = True
eval_residuals = True
create_gif = False

# training parameters and datasets
data_paths = None
if gov_eqs == 'Nematic':
    # [xi_1,xi_2] -> [p,K]
    input_dim = 2
    output_dim = 2
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = False # this is to be consistent with ascending coordinates in the figures
    data_paths = ('./data/Nematic/train/Qxx_data.csv', './data/Nematic/train/Qxy_data.csv')
    label_paths = ('./data/Nematic/train/Qxx_label.csv', './data/Nematic/train/Qxy_label.csv')
    bcs = 'none' # 'none', 'periodic'
    pixels_per_dim = 64

    ds = Dataset(data_paths, label_paths, use_double=use_double)
    # ds_valid = Dataset(data_paths_valid, use_double=use_double)
    if use_ddim_x0:
        train_batch_size = 16
    else:
        train_batch_size = 32
    sigmoid_last_channel = False
    train_iterations = 300000
else:
    raise ValueError('Unknown governing equations.')

if use_double:
    torch.set_default_dtype(torch.float64)

dl = cycle(DataLoader(ds, batch_size = train_batch_size, shuffle=False))
# dl_valid = cycle(DataLoader(ds_valid, batch_size = train_batch_size, shuffle=False))

# diffusion utils
diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

# model 
if gov_eqs == 'Nematic':
    model = Unet3D(dim = 32, channels = output_dim, sigmoid_last_channel = sigmoid_last_channel).to(device)
else:
    raise ValueError('Unknown governing equations, cannot create model.')
if load_model_flag:
    load_model(Path(load_path, 'model', 'checkpoint_' + str(load_model_step) + '.pt'), model)
ema.register(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# residual computation based on governing equations
if gov_eqs == 'Nematic':
    residuals = ResidualsNematic(model = model, fd_acc = fd_acc, pixels_per_dim = pixels_per_dim, pixels_at_boundary = pixels_at_boundary, reverse_d1 = reverse_d1, device = device, bcs = bcs, domain_length = domain_length, residual_grad_guidance= residual_grad_guidance, use_ddim_x0 = use_ddim_x0, ddim_steps = ddim_steps)
else:
    raise ValueError('Unknown residuals mode.')

optimizer = optim.Adam(model.parameters(), lr=1.e-4)

if wandb_track:
    import wandb
    wandb.init(project='pi_diffusion', name=name)
    log_fn = wandb.log
else:
    log_fn = noop
log_freq = 20
    
output_save_dir = f'./trained_models/{name}'
os.makedirs(output_save_dir, exist_ok=True)

pbar = tqdm(range(train_iterations+1))
for iteration in pbar:
    model.train()
    cur_batch = next(dl).to(device)
    loss, data_loss, residual_loss = diffusion_utils.model_estimation_loss(
                cur_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual)    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()        
    # logging
    if iteration % log_freq == 0:
        pbar.set_description(f'training loss: {loss.item():.3e}')
        log_fn({'loss': loss.item()}, step=iteration)
        log_fn({'loss_data': data_loss}, step=iteration)
        log_fn({'residual_mean_abs': residual_loss}, step=iteration)
    # ema update
    if iteration > ema_start:
        ema.update(model)

    # evaluation on validation set
    model.eval()
    ema.ema(residuals.model)
    # if iteration % test_eval_freq == 0 and exists(dl_valid):
    #     cur_test_batch = next(dl_valid).to(device)
    #     # NOTE: we do not use torch.no_grad() since we may require residual gradient for classifier-free guidance
    #     loss_test, data_loss_test, residual_loss_test = diffusion_utils.model_estimation_loss(
    #                 cur_test_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual)
        
    #     print(f'test loss at iteration {iteration}: {loss_test:.3e}')
    #     log_fn({'loss_test': loss_test.item()}, step=iteration)
    #     log_fn({'loss_data_test': data_loss_test}, step=iteration)
    #     log_fn({'residual_mean_abs_test': residual_loss_test}, step=iteration)

    # generate and evaluate samples
    if (iteration % sample_freq == 0) or (iteration == train_iterations):        
        if gov_eqs == 'Nematic':
            conditioning_input = None
            sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
        
        output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape, 
                                save_output=save_output, surpress_noise=True, 
                                use_dynamic_threshold=use_dynamic_threshold, 
                                residual_func=residuals, eval_residuals = eval_residuals)
        
        if eval_residuals:
            seqs = output[0]
            residual = output[1]['residual']
            residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
        else:
            seqs = output
            
        output_save_dir_step = output_save_dir + f'/training/step_{iteration}/'
        os.makedirs(output_save_dir_step, exist_ok=True)
                
        labels = ['sample', 'model_output']
        for seq_idx, seq in enumerate(seqs):

            # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
            if seq_idx == 1:
                continue

            seq = torch.stack(seq, dim=0)

            if len(seq.shape) == 6:
                seq = seq.squeeze(-3)
                
            last_preds = seq[-1].numpy()
            sel_samples = np.arange(len(last_preds))
            channels = np.arange(output_dim)

            for sel_sample in sel_samples:
                for sel_channel in channels:
                    last_pred = last_preds[sel_sample, sel_channel]
                    last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min()) # normalize to [0,1]

                    image = np.uint8(last_pred_normalized * 255)
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    if eval_residuals:
                        title = f'eq: {residual[sel_sample]:.2e}'
                        plt.title(title, color='green')
                    filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
                    plt.savefig(output_save_dir_step + filename, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    os.makedirs(output_save_dir_step + f'/sample_{sel_sample}/', exist_ok=True)
                    np.savetxt(output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')

                    if create_gif:
                        sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                        image_array_to_gif(sel_seq, output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.gif')


        if eval_residuals:
            residuals_array = residual.detach().cpu().numpy()
            # logging
            log_fn({'residual_mean_abs_samples': np.nanmean(residuals_array)}, step=iteration)
            log_fn({'residual_median_abs_samples': np.nanmedian(residuals_array)}, step=iteration)
            df_data = {'Sample Index': list(range(no_samples)),
                    'Residuals (abs)': list(residuals_array)}
            print(df_data)
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_save_dir_step, 'sample_statistics.csv')
            df.to_csv(csv_path, index=False)

        
        if iteration > 0:
            save_model(config, model, iteration, output_save_dir)

    ema.restore(residuals.model)

if wandb_track:
    wandb.finish()