import os, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from einops import reduce, rearrange
from src.residuals_nematic import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fancy plots
def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')
hdr_plot_style()

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, x, y] to [batch, x*y, channels].
    """
    assert len(tensor.shape) == 4, 'Input tensor must have shape [batch, channels, x, y].'
    batch_size, channels, pixels_x, pixels_y = tensor.shape
    return torch.permute(tensor, (0,2,3,1)).view(batch_size, pixels_x*pixels_y, channels)

def b_xy_c_to_image(tensor, pixels_x = None, pixels_y = None):
    """
    Transpose the tensor from [batch, x*y, channels] to [batch, channels, x, y].
    """
    assert len(tensor.shape) == 3, 'Input tensor must have shape [batch, x*y, channels].'
    batch_size, pixels_x_times_y, channels = tensor.shape
    if pixels_x is None and pixels_y is None:
        assert np.sqrt(pixels_x_times_y) % 1 == 0, 'Number of pixels must be a perfect square.'
        pixels_x = pixels_y = int(np.sqrt(pixels_x_times_y))
    else:
        assert pixels_x*pixels_y == pixels_x_times_y, 'Number of given pixels must match dim 1 of input tensor.'
    return torch.permute(tensor.view(batch_size, pixels_x, pixels_y, channels), (0,3,1,2))

def resize_image(tensor, target_size):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    assert len(tensor.shape) > 3, f"Expected image, got {tensor.shape}"
    original_shape = tensor.shape
    batch_size = original_shape[0]
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b' + ' (' + ' '.join([f'c{i}' for i in range(num_dims)]) + ') ' + 'x y'
    tensor = rearrange(tensor, pattern)
    tensor = transforms.Resize((target_size, target_size), antialias=False)(tensor).view(batch_size, *original_shape[1:-2], target_size, target_size)
    return tensor


def noop(*args, **kwargs):
    pass

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def plot_data(data):
    plt.figure(figsize=(16, 12))
    plt.scatter(*data, s=10)
    plt.title('Ground truth $q(\mathbf{x}_{0})$')
    plt.show()
    exit()

def sample_zeros(size):
    return np.zeros((size, 2))

def sample_gaussian(size, dim=2):
    return np.random.randn(size, dim)

def sample_circle(size):
    # sample points from a circle
    theta = np.random.uniform(0, 2*np.pi, size)
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return x

def sample_hypersphere(size, dim):
    # sample points from a hypersphere surface in `dim` dimensions
    x = np.random.normal(0, 1, (size, dim))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / norm    
    return x_normalized

def sample_two_points(size):
    # two points in 2D
    x = np.array([[-0.5, -0.5], [0.5, 0.5]])
    # random selection of these two points
    return x[np.random.randint(2, size=size)]

def sample_four_points(size):
    # four points in 2D
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    # random selection of these four points
    return x[np.random.randint(4, size=size)]

def sample_images_with_squares(no_points, pixels_per_dim, dim, frame_dim = False, use_double = True):

    if use_double:
        dtype = np.float64
    else:
        dtype = np.float32

    # Define the size of the square (e.g., a quarter of the image dimension)
    square_size = pixels_per_dim // 4

    # Initialize an array to store the images
    # Shape: (no_points, pixels_per_dim, pixels_per_dim, dim)
    if frame_dim:
        images = np.zeros((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=dtype)
    else:
        images = np.zeros((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=dtype)

    for i in range(no_points):
        # Randomly choose the top-left corner of the square
        x_start = np.random.randint(0, pixels_per_dim - square_size)
        y_start = np.random.randint(0, pixels_per_dim - square_size)

        for j in range(dim):
            # Draw the square in each channel of the image
            # You can modify the pattern per channel as needed
            if frame_dim:
                images[i, j, :, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
            else:
                images[i, j, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
                
    return images

def sample_ones(no_points, pixels_per_dim, dim, frame_dim = False):
    if frame_dim:
        return np.ones((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=float)
    else:
        return np.ones((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=float)

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module, backup=True):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if backup:
                    self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, module):
        assert hasattr(self, 'backup')
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def remove_outliers(data, percentile = 0.01, also_lower_bound = False):    
    percentile *= 100    
    if data.size == 0:
        return data  # Return the empty array as is
    
    norms = np.linalg.norm(data, axis=1)
    
    # compute the lower and upper bounds for filtering based on norms
    lower_bound = np.percentile(norms, percentile) if also_lower_bound else 0.
    upper_bound = np.percentile(norms, 100 - percentile)
    mask = (norms > lower_bound) & (norms < upper_bound)
    return data[mask]

# tensor of shape (channels, frames, height, width) -> gif
def array_to_gif(data, output_save_dir, x_lim, y_lim, label = None, duration = 0.05):    
    # Create a GIF writer object
    with imageio.get_writer(output_save_dir, mode='I', duration = duration, loop=1) as writer:
        for step in range(data.shape[0]):
            fig, ax = plt.subplots()
            ax.scatter(data[step, :, 0], data[step, :, 1], s=10)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            if label is not None:
                if label == 'sampled':
                    ax.set_title('$p(\mathbf{x}_{' + str(len(data)-step-1)+'})$')
                elif label == 'pred':
                    ax.set_title('Model pred. step ' + str(len(data)-step-1))
                else:
                    ax.set_title(label)
            # Save the current figure directly to the GIF, without an intermediate file
            gif_frame_path = output_save_dir[:-4] + f'_gif_frame_{step}.png'
            plt.savefig(gif_frame_path)
            plt.close(fig)
            writer.append_data(imageio.imread(gif_frame_path))
            # remove the temporary image file after adding to the GIF
            os.remove(gif_frame_path)

def image_array_to_gif(Q, output_file, frame_duration=0.05, normalization_mode='final_pred', given_min_max=None):
    """
    Create a GIF from a numpy array of images with different normalization modes.

    :param image_array: Numpy array of shape (frames, pixel, pixel, channels)
    :param output_file: Output file path for the GIF
    :param frame_duration: Duration of each frame in the GIF
    :param normalization_mode: Mode of normalization ('given', 'global', 'individual', 'none')
    :param given_min_max: Tuple of (min, max) values for 'given' normalization mode
    """
    Q = np.array(Q)
    b = Q.shape
    with imageio.get_writer(output_file, mode='I', duration=frame_duration) as writer:
        for frame in Q:
            Qxx = Q[frame,:,:,0]
            Qxy = Q[frame,:,:,1]
            a = Qxx.shape
            N2, N1 = Qxx.shape[1], Qxx.shape[0]
            nx = np.zeros((N2,N1))
            ny = np.zeros((N2,N1))
            S = np.zeros((N2,N1))
            x, y = np.meshgrid(np.arange(0, N1, 1), np.arange(0, N2, 1))
            for i in range(N2):
                for j in range(N1):
                    matrix = np.array([[Qxx[i, j], Qxy[i, j]],
                                    [Qxy[i, j], -Qxx[i, j]]])
                    vals, vecs = np.linalg.eigh(matrix)
                    index = np.argmax(vals)
                    nx[i,j] = vecs[0, index]
                    ny[i,j] = vecs[1, index]
                    S[i,j] = vals[index]*2
            # plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(S, cmap='hot', interpolation='nearest', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=0.707)
            plt.colorbar(im)
            plt.xlabel('x')
            plt.ylabel('y')
            ax.quiver(x, y, nx, ny, color='black', scale=100, headlength=0, headaxislength=0, headwidth=0, pivot='middle')
            frame = plt.gcf()
            writer.append_data(frame)

def save_model(config, model, train_iterations, output_save_dir):

    os.makedirs(Path(output_save_dir, 'model/'), exist_ok=True)

    # save yaml file for later runs
    with open(output_save_dir + '/model/model.yaml', 'w') as yaml_file:
        yaml.dump(dict(config), yaml_file, default_flow_style=False)    
    save_dir_model = output_save_dir + '/model/checkpoint_' + str(train_iterations) + '.pt'
    save_obj = dict(
        model = model.state_dict()
    )
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    print(f'\ncheckpoint saved to {output_save_dir}/.')

def load_model(path, model, strict = True):

    # to avoid extra GPU memory usage in main process when using Accelerate
    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    try:
        model.load_state_dict(loaded_obj['model'], strict = strict)
    except RuntimeError:
        print('Failed loading state dict.')
    print('\nCheckpoint loaded from {}'.format(path))

    return model

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class DenoisingDiffusion(nn.Module):
    def __init__(self, n_steps, device, residual_grad_guidance = False):
        self.n_steps = n_steps
        self.device = device
        self.diff_dict = self.create_diff_dict()
        self.residual_grad_guidance = residual_grad_guidance

    def create_diff_dict(self): 
        diff_dict = {
            'betas': self.make_beta_schedule(schedule='cosine', n_timesteps=self.n_steps, start=1e-5, end=1e-2).to(self.device),
        }
        diff_dict['alphas'] = 1. - diff_dict['betas']
        diff_dict['sqrt_recip_alphas'] = torch.sqrt(1. / diff_dict['alphas'])
        diff_dict['alphas_prod'] = torch.cumprod(diff_dict['alphas'], 0)
        diff_dict['alphas_prod_p'] = torch.cat([torch.tensor([1], device=device).float(), diff_dict['alphas_prod'][:-1]], 0)
        diff_dict['alphas_bar_sqrt'] = torch.sqrt(diff_dict['alphas_prod'])
        diff_dict['sqrt_recip_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'])
        diff_dict['sqrt_recipm1_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'] - 1)
        diff_dict['one_minus_alphas_bar_log'] = torch.log(1 - diff_dict['alphas_prod'])
        diff_dict['one_minus_alphas_bar_sqrt'] = torch.sqrt(1 - diff_dict['alphas_prod'])
        diff_dict['alphas_prod_prev'] = F.pad(diff_dict['alphas_prod'][:-1], (1, 0), value=1.)
        diff_dict['posterior_mean_coef1'] = diff_dict['betas'] * torch.sqrt(diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        # posterior mean_coef1 = betas * sqrt(a_bar_t-1) / (1 - a_bar_t)
        diff_dict['posterior_mean_coef2'] = (1. - diff_dict['alphas_prod_prev']) * torch.sqrt(diff_dict['alphas']) / (1. - diff_dict['alphas_prod'])
        # posterior mean_coef2 = (1 - a_bar_t-1) * sqrt(a_t) / (1 - a_bar_t)
        diff_dict['noise_mean_coeff'] = torch.sqrt(1. / diff_dict['alphas']) * (1. - diff_dict['alphas']) / torch.sqrt(1. - diff_dict['alphas_prod'])
        # noise_mean_coeff = sqrt(1/a) * (1 - a) / sqrt(1 - a_bar)
        # posterior variance
        diff_dict['posterior_variance'] = diff_dict['betas'] * (1. - diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        # posterior variance = betas * (1 - a_bar_t-1) / (1 - a_bar_t)
        # clip this since it is 0 at the beginning
        diff_dict['posterior_variance_clipped'] = diff_dict['posterior_variance'].clone()
        diff_dict['posterior_variance_clipped'][0] = diff_dict['posterior_variance'][1]

        # NOTE Ho et al. also have a version that clips the log to 1.e-20.
        diff_dict['posterior_log_variance_clipped'] = torch.log(diff_dict['posterior_variance_clipped'])

        use_constant_p2_weight = False
        if use_constant_p2_weight:
            p2_loss_weight_gamma = 1.0  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1.0
            diff_dict['p2_loss_weight'] = (p2_loss_weight_k + diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])) ** -p2_loss_weight_gamma
        else:
            snr = diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])
            diff_dict['p2_loss_weight'] = torch.minimum(snr, torch.ones_like(snr) * 5.0) # from https://arxiv.org/pdf/2303.09556.pdf

        return diff_dict

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == 'cosine':
            s = 0.008
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        return betas

    # Sampling function
    def q_sample(self, x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_t = extract(alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    def plot_diffusion(self, dataset, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
        fig, axs = plt.subplots(1, 10, figsize=(18, 2))
        for i in range(10):
            q_i = self.q_sample(dataset, torch.tensor([i * 10]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
            axs[i].set_axis_off(); axs[i].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$', fontsize=10)
        plt.show()

    def p_sample(self, x, label_input, conditioning_input, t, 
                save_output = False, surpress_noise = False, 
                use_dynamic_threshold = False, residual_func = None, eval_residuals = False):
        '''
        Function for backward denoising step, input is x_t & Mask & t, output is x_{t-1}
        '''

        x_init = x.clone().detach()        
        if conditioning_input is not None:
            conditioning, bcs, solution = conditioning_input 
            x = torch.cat((x, conditioning), dim = 1)
        batch_size = len(x)
        
        t = torch.tensor([t], device=x.device)
        model_input = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model
        model_input = (model_input, t.repeat(batch_size))
        # model input is x_t and t
        model_intermediate = None
        
        # model output
        # evaluate residuals at last timestep if required
        if residual_func.gov_eqs == 'Nematic':
            residual_input = (model_input, )
            # sample = True
            sample = False
        # MAKE SURE ALL THE INPUT AT THE SAME DEVICE
        if not isinstance(label_input, torch.Tensor):
            label_input = torch.tensor(label_input, device=x.device)
        else:
            label_input = label_input.clone().detach().to(x.device)
        out_dict = residual_func.compute_residual(  residual_input,
                                                    label_input,
                                                    reduce='per-batch',
                                                    return_model_out = True,
                                                    sample = sample,
                                                    ddim_func = self.ddim_sample_x0)
        
        output, residual = out_dict['model_out'], out_dict['residual']
        model_out = output
        if len(model_out.shape) == 3:
            # convert to image [batch_size, channels, pixels, pixels]
            model_out = generalized_b_xy_c_to_image(model_out)

        if save_output:
            model_intermediate = model_out.clone().detach()
        x0_pred = model_out
        mean = (
            extract(self.diff_dict['posterior_mean_coef1'], t, x_init) * x0_pred +
            extract(self.diff_dict['posterior_mean_coef2'], t, x_init) * x_init
        )
        # gaussian mean mu = coef1 * x0_pred + coef2 * x_t

        # Generate z
        z = torch.randn_like(x_init, device=x.device)
        # Fixed sigma
        sigma_t = extract(self.diff_dict['betas'], t, x_init).sqrt()
        # no noise when t == 0
        if surpress_noise:
            nonzero_mask = (1. - (t == 0).float())
        else:
            nonzero_mask = 1.
        sample = mean + nonzero_mask * sigma_t * z
        # sample = mean + sigma_t * N(0,1)
        dynamic_thres_percentile = 0.9
        if use_dynamic_threshold:
            def maybe_clip(x):
                s = torch.quantile(
                    rearrange(x.float(), "b ... -> b (...)").abs(),
                    dynamic_thres_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = right_pad_dims_to(x, s)
                x = x.clamp(-s, s) / s
                return x
            sample = maybe_clip(sample)
        
        if (t[0] == 0 and eval_residuals):
            aux_out = {}
            aux_out['residual'] = residual
            return (sample, model_intermediate), aux_out
        else:
            return (sample, model_intermediate), None

    # NOTE we do not use @torch.inference_mode() since we need gradients to obtain residual
    # to free up memory, we manually call .detach() where appropriate
    def p_sample_loop(self,
                    conditioning_input,
                    label_input,
                    shape,
                    save_output = False, 
                    surpress_noise = True,
                    use_dynamic_threshold = False,
                    residual_func = None,
                    eval_residuals = False):
        '''
        Backward denoising loop from t=n_steps to t=0
        '''
        # random initial condition at t=n_steps
        cur_x = torch.randn(shape, device=self.diff_dict['alphas'].device)
        x_seq = [cur_x.detach().cpu()]

        if save_output:
            interm_imgs = [torch.zeros(shape)]
        else:
            interm_imgs = []

        interm_img = None

        # backward denoising loop from t=n_steps to t=0
        for i in reversed(range(self.n_steps)):
            

            eval_residuals = True
                
            output = self.p_sample(cur_x.detach(), label_input, conditioning_input, i, save_output, surpress_noise, use_dynamic_threshold,
                                        residual_func = residual_func, eval_residuals = eval_residuals)
            cur_x, interm_img = output[0]

            x_seq.append(cur_x.detach().cpu())
            interm_imgs.append(interm_img.detach().cpu())
            
        if eval_residuals:
            return (x_seq, interm_imgs), output[1]
        else:
            return x_seq, interm_imgs

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        KL divergence between normal distributions parameterized by mean and log-variance.
        """
        kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
        return kl

    def gaussian_log_likelihood(self, x, means, variance):
        centered_x = x - means    
        squared_diffs = (centered_x ** 2) / variance
        log_probs = -0.5 * squared_diffs
        return log_probs
    # log p(x|mu,var) = -1/2 * (x - mu)^2 / var

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t -
            extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t) * noise
        )
    # x0_pred = 1/\sqrt(a_bar) * x_t - \sqrt(1-1/a_bar) * N(0,1)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t - x0
        ) / extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t)
    # noise_pred = 1/\sqrt(a_bar) * x_t - x_0 / \sqrt(1-1/a_bar) 

    def predict_noise_from_mean(self, x_t, t, mean_t):
        return (
            extract(self.diff_dict['sqrt_recip_alphas'], t, mean_t) * x_t - mean_t
        ) / extract(self.diff_dict['noise_mean_coeff'], t, mean_t)
    # noise_pred = 1/\sqrt(a) * x_t - mean_t / noise_mean_coeff

    def loss_variational(self, output, x_0, x_t, t, base_2 = False):    
        batch_size = x_0.shape[0]

        # Compute the true mean and variance
        true_mean = (
        extract(self.diff_dict['posterior_mean_coef1'], t, x_t) * x_0 +
        extract(self.diff_dict['posterior_mean_coef2'], t, x_t) * x_t
        )
        # posterior mean_coef1 = betas * sqrt(a_bar_t-1) / (1 - a_bar_t)
        # posterior mean_coef2 = (1 - a_bar_t-1) * sqrt(a_t) / (1 - a_bar_t)
        # true mean = beta_t * sqrt(a_bar_t-1) / (1 - a_bar_t) * x_0 + (1 - a_bar_t-1) * sqrt(a_t) / (1 - a_bar_t) * x_t
    
        true_var = extract(self.diff_dict['posterior_variance_clipped'], t, x_t)
        model_var = true_var
        # model_var same as true var

        # Infer the mean and variance with our model
        model_mean = output

        # Compute the KL loss
        true_var_log = torch.log(true_var)
        model_var_log = torch.log(model_var)
        kl = self.normal_kl(true_mean, true_var_log, model_mean, model_var_log)
        # kl DIV between true and model distributions
        kl = torch.mean(kl.view(batch_size, -1), dim=1)
        if base_2:
            kl = kl / np.log(2.)

        # define p(x_0|x_1) simply as a gaussian
        log_likelihood = self.gaussian_log_likelihood(x_0, means=model_mean, variance=model_var)
        log_likelihood = torch.mean(log_likelihood.view(batch_size, -1), dim=1) 
        if base_2:
            log_likelihood = log_likelihood / np.log(2.)

        # At the first timestep return the log likelihood, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # BUG (imo) nan/inf values in tensor that is not considered in torch.where() still affects gradients. Thus check for this before.
        assert log_likelihood.isnan().any() == False, 'Log likelihood is nan.'
        assert log_likelihood.isinf().any() == False, 'Log likelihood is inf.'

        loss_log_likelihood = -1. * log_likelihood # since we minimize loss (instead of maximizing likelihood)

        loss = torch.where(t == 0, loss_log_likelihood, kl)
        # if t==0, return log likelihood, otherwise return KL

        return loss.mean(-1)

    def model_estimation_loss(self,
                              data_input,
                              label_input,
                              residual_func = None, 
                              c_data = 1.,
                              c_residual = 0.):

        batch_size = len(data_input)
        t = torch.randint(0, self.n_steps, size=(batch_size,), device=data_input.device)
        # random t from 0 to n_steps
        if residual_func.gov_eqs == 'Nematic':
            x_0 = data_input

        # x0 multiplier
        a = extract(self.diff_dict['alphas_bar_sqrt'], t, x_0) # sqrt(a_bar)
        # eps multiplier
        am1 = extract(self.diff_dict['one_minus_alphas_bar_sqrt'], t, x_0) # sqrt(1-a_bar)
        e = torch.randn_like(x_0, device=x_0.device) # N(0,1)
        # model input
        x = x_0 * a + e * am1 # previous x_t 
        # forward diffusion x_t= sqrt(a_bar)*x_0 + sqrt(1-a_bar)*N(0,1)
        noisy_in1 = x[:, 0, :, :]  # [batch_size,1, pixels_per_dim, pixels_per_dim]
        noisy_in2 = x[:, 1, :, :]  # [batch_size,1, pixels_per_dim, pixels_per_dim]
        Azi = label_input[:,0, :, :] # [batch_size, pixels_per_dim, pixels_per_dim] anchoring
        Mask = label_input[:,1,:, :] # [batch_size, pixels_per_dim, pixels_per_dim] index 0 --> no need
        Seq = torch.sqrt(torch.tensor(2.0, device=device)) / 2
        Qxx_anchoring = Seq * torch.cos(Azi*2)
        Qxy_anchoring = Seq * torch.sin(Azi*2)
        # 创建新的张量而不是修改原有张量
        # noisy_in1 = torch.where(Mask == 0, Qxx_anchoring, noisy_in1)
        # noisy_in2 = torch.where(Mask == 0, Qxy_anchoring, noisy_in2)
        noisy_in1[Mask == 0] = Qxx_anchoring[Mask == 0]
        noisy_in2[Mask == 0] = Qxy_anchoring[Mask == 0]
        x = torch.stack([noisy_in1, noisy_in2], dim=1)  # [batch_size, 2, pixels_per_dim, pixels_per_dim]
        x = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model

        model_input = (x, t)


        if residual_func.gov_eqs == 'Nematic':
            residual_input = (model_input, )
            label_input = (label_input)
        out_dict = residual_func.compute_residual(residual_input,
                                                  label_input,
                                                  reduce='per-batch', 
                                                  return_model_out = True,
                                                  sample = False, 
                                                  ddim_func = self.ddim_sample_x0)
        
        residual, output = out_dict['residual'], out_dict['model_out']
        # get residual of PDE and model output as x0_pred

        Mask = label_input[:,1]
        # reshape output to image (batch_size, channels, pixels, pixels)
        if len(output.shape) == 3:
            output = b_xy_c_to_image(output)
            Mask = b_xy_c_to_image(Mask)
        # 使用非破坏性mask操作
        Mask = Mask.unsqueeze(1)  # 增加通道维度
        # copy Mask_expanded to [batch_size, 2, pixels_per_dim, pixels_per_dim]  
        Mask_expanded = Mask.repeat(1, 2, 1, 1)  # 重复通道维度
        # # 创建新的目标张量而不是修改原有张量
        target = x_0.clone()
        # target[Mask_expanded == 0] = 0
        # # target = torch.where(Mask_expanded == 0, torch.zeros_like(target), target)
        
        # # 同样处理输出
        # # output = torch.where(Mask_expanded == 0, torch.zeros_like(output), output)
        # output[Mask_expanded == 0] = 0
        
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(target, output)
        # print('loss shape:', loss.shape)
        loss = torch.where(Mask_expanded == 0, torch.zeros_like(loss), loss)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.diff_dict['p2_loss_weight'], t, loss)
        loss = loss.mean()

        # adjust data-driven loss term
        data_loss = c_data * loss
        data_loss_track = data_loss.item()
        loss = data_loss

        # add negative residual log-likelihood, i.e., - log p(r|x_0_pred(x_0))
        var = extract(self.diff_dict['posterior_variance_clipped'], t, residual)

        #posterior variance = betas * (1 - a_bar_t-1) / (1 - a_bar_t)
        # var_residual = betas * (1 - a_bar_t-1) / (1 - a_bar_t)
        # residual_loss_track = residual.mean().item()
        Mask = generalized_image_to_b_xy_c(Mask)
        Mask_expanded = Mask.repeat(1, 1, 2)  # 重复通道维度
        residual = torch.where(Mask_expanded == 0, torch.zeros_like(residual), residual)
        

        residual_log_likelihood = self.gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
        #log p(r,residual,var) = -1/2 * (residual)^2 / var
        residual_loss = c_residual * -1. * residual_log_likelihood
        residual_loss_track = residual_loss.abs().mean().item()
        # residual_loss = c/2var * (residual)^2
        loss += residual_loss.mean()

        return loss, data_loss_track, residual_loss_track
    
    def ddim_sample_x0(self, xt, t, model, shape, reduced_n_steps, ddim_sampling_eta, self_cond = None):

        batch, device, sample_timesteps, eta = shape[0], self.diff_dict['alphas'].device, reduced_n_steps, ddim_sampling_eta

        if len(t) == 1:
            batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
        else:
            batch_t = t

        batch_t = batch_t.cpu().numpy()
        seqs = []
        seqs_next = []
        for t_idx, t in enumerate(batch_t):
            seq = list(map(int, np.linspace(0, batch_t[t_idx], sample_timesteps+2, endpoint=True, dtype=float))) # evenly spread from 0 to current t
            seqs.append(list(reversed(seq)))
            seq_next = [-1] + list(seq[:-1])
            seqs_next.append(list(reversed(seq_next)))
            seq = None

        # tranpose to have time as first dimension
        cur_times = torch.tensor(seqs, device=device).T
        next_times = torch.tensor(seqs_next, device=device).T

        time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if len(xt.shape) == 3:            
            xt = generalized_b_xy_c_to_image(xt)


        model_input = xt
        cur_x = xt
        x0_pred = None

        model_output = None
        for fwd_idx, (t, t_next) in enumerate(time_pairs):

            # create mask for those timesteps that are equal
            mask = (t == t_next).float().view(-1, 1, 1, 1)
            model_output = model(model_input, t, self_cond)
            x0_pred = model_output            
            mean = (
                extract(self.diff_dict['posterior_mean_coef1'], t, cur_x) * x0_pred +
                extract(self.diff_dict['posterior_mean_coef2'], t, cur_x) * cur_x
            )
            # noise estimate
            eps_theta = self.predict_noise_from_mean(cur_x, t, mean)

            if fwd_idx == 0:
                model_out = model_output

            if t_next[0] < 0: # this happens when we predict x0, should never happen during training
                # assert that all next timesteps are equal
                assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
                cur_x = x0_pred
                continue

            alpha = extract(self.diff_dict['alphas_prod'], t, cur_x)
            alpha_next = extract(self.diff_dict['alphas_prod'], t_next, cur_x)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(cur_x)

            # only update cur_x where t != t_next
            # NOTE: this is the standard update (not using larger variance)
            cur_x_ = x0_pred * alpha_next.sqrt() + \
                     c * eps_theta + \
                     sigma * noise
            
            cur_x = mask * cur_x + (1 - mask) * cur_x_

        assert model_out is not None, 'Model output not given.'
        return cur_x, model_out
    
    def generate_mask(self, N1,N2,icx,icy,R,anchoring_type,inclusion,anchoring_direction):
        x,y = np.meshgrid(np.arange(N1), np.arange(N2))
        mask = np.zeros((2,N1,N2))
        # find the normal vector of the boundary
        r = np.sqrt((x-icx)**2 + (y-icy)**2) 
        r[r==0] = 1e-4
        vx = (x-icx)/r
        vy = (y-icy)/r
        azi = np.arctan2(vy,vx)
        if inclusion:
            mask[1,:,:][ r < R] = 1
        else:
            mask[1,:,:][ r > R] = 1

        mask[1,:,:] = 1 - mask[1,:,:]
        assert not np.any(np.isnan(azi)), 'azi has nan values'
        if anchoring_type==1: #homeotropic
            mask[0,:,:] = azi
        elif anchoring_type==2: #planar
            mask[0,:,:] = 0.5*np.pi+azi
        elif anchoring_type==3: #arbitrary
            mask[0,:,:] = anchoring_direction*np.pi+azi
        elif anchoring_type==4: # +1/2
            mask[0,:,:] = 0.5*azi+anchoring_direction*np.pi
        elif anchoring_type==5:
            mask[0,:,:] = -0.5*azi+anchoring_direction*np.pi
        return mask
    
    def generate_mask_triangle(self, Ax,Ay,Bx,By,Cx,Cy,N1,N2,inclusion,anchoring_type):
        x,y = np.meshgrid(np.arange(N1), np.arange(N2))
        mask = np.zeros((2,N1,N2))
        #vector AB
        ABx = Bx - Ax
        ABy = By - Ay
        #vector BC
        BCx = Cx - Bx
        BCy = Cy - By
        #vector CA
        CAx = Ax - Cx
        CAy = Ay - Cy
        #vector PA
        PAx = x - Ax
        PAy = y - Ay
        #vector PB
        PBx = x - Bx
        PBy = y - By
        #vector PC
        PCx = x - Cx
        PCy = y - Cy
        #cross product AB and AP
        cross_AB_AP = ABx*PAy - ABy*PAx
        #cross product BC and BP
        cross_BC_BP = BCx*PBy - BCy*PBx
        #cross product CA and CP
        cross_CA_CP = CAx*PCy - CAy*PCx

        mask[1,:,:][(cross_AB_AP > 0) & (cross_BC_BP>0) & (cross_CA_CP>0)] = 1 
        mask[1,:,:][(cross_AB_AP < 0) & (cross_BC_BP<0) & (cross_CA_CP<0)] = 1 
        mask[1,:,:][(cross_AB_AP == 0) & (cross_BC_BP>0) & (cross_CA_CP>0)] = 1 
        mask[1,:,:][(cross_AB_AP == 0) & (cross_BC_BP<0) & (cross_CA_CP<0)] = 1 
        mask[1,:,:][(cross_BC_BP == 0) & (cross_CA_CP>0) & (cross_AB_AP>0)] = 1 
        mask[1,:,:][(cross_BC_BP == 0) & (cross_CA_CP<0) & (cross_AB_AP<0)] = 1
        mask[1,:,:][(cross_CA_CP == 0) & (cross_AB_AP>0) & (cross_BC_BP>0)] = 1
        mask[1,:,:][(cross_CA_CP == 0) & (cross_AB_AP<0) & (cross_BC_BP<0)] = 1
        if inclusion:
            mask[1,:,:] = 1-mask[1,:,:]

        length_AB = np.sqrt(ABx**2 + ABy**2)
        length_BC = np.sqrt(BCx**2 + BCy**2)
        length_CA = np.sqrt(CAx**2 + CAy**2)
        #distance from P to AB
        d_AB = np.abs(cross_AB_AP)/length_AB
        #distance from P to BC
        d_BC = np.abs(cross_BC_BP)/length_BC
        #distance from P to CA
        d_CA = np.abs(cross_CA_CP)/length_CA
        # pick the minimum distance to set the anchoring
        d = np.stack([d_AB,d_BC,d_CA],axis=-1)
        index = np.argmin(d,axis=-1)
        # anchoring 0: along AB, find the angle of the normal vector of AB
        # find the normal vector of AB
        n_ABx = -ABy/length_AB
        n_ABy = ABx/length_AB
        Azi_n_AB = np.arctan2(n_ABy,n_ABx)
        # find the angle of the normal vector of BC
        n_BCx = -BCy/length_BC
        n_BCy = BCx/length_BC
        Azi_n_BC = np.arctan2(n_BCy,n_BCx)
        # find the angle of the normal vector of CA
        n_CAx = -CAy/length_CA
        n_CAy = CAx/length_CA
        Azi_n_CA = np.arctan2(n_CAy,n_CAx)
        # find the centroid of the triangle
        cx = (Ax + Bx + Cx)/3
        cy = (Ay + By + Cy)/3
        vx = x - cx
        vy = y - cy
        Azi = np.arctan2(vy,vx)
        if anchoring_type==1:
            # homeotropic anchoring
            mask[0,:,:][index==0] = Azi_n_AB
            mask[0,:,:][index==1] = Azi_n_BC
            mask[0,:,:][index==2] = Azi_n_CA
        elif anchoring_type==2:
            # planar anchoring
            mask[0,:,:][index==0] = 0.5*np.pi + Azi_n_AB
            mask[0,:,:][index==1] = 0.5*np.pi + Azi_n_BC
            mask[0,:,:][index==2] = 0.5*np.pi + Azi_n_CA
        elif anchoring_type==3:
            # arbitrary anchoring
            mask[0,:,:] = Azi 
        elif anchoring_type==4:
            mask[0,:,:] = Azi + 0.5*np.pi
        return mask
    
    def generate_mask_pentagon(self,Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Ex,Ey,N1,N2,inclusion,anchoring_type):
        x,y = np.meshgrid(np.arange(N1), np.arange(N2))
        mask = np.zeros((2,N1,N2))
        #vector AB
        ABx = Bx - Ax
        ABy = By - Ay
        #vector BC
        BCx = Cx - Bx
        BCy = Cy - By
        #vector CD
        CDx = Dx - Cx
        CDy = Dy - Cy
        #vector DE
        DEx = Ex - Dx
        DEy = Ey - Dy
        #vector EA
        EAx = Ax - Ex
        EAy = Ay - Ey
        #vector PA
        PAx = x - Ax
        PAy = y - Ay
        #vector PB
        PBx = x - Bx
        PBy = y - By
        #vector PC
        PCx = x - Cx
        PCy = y - Cy
        #vector PD
        PDx = x - Dx
        PDy = y - Dy
        #vector PE
        PEx = x - Ex
        PEy = y - Ey
        #cross product AB and AP
        cross_AB_AP = ABx*PAy - ABy*PAx
        #cross product BC and BP
        cross_BC_BP = BCx*PBy - BCy*PBx
        #cross product CD and CP
        cross_CD_CP = CDx*PCy - CDy*PCx
        #cross product DE and DP
        cross_DE_DP = DEx*PDy - DEy*PDx
        #cross product EA and EP
        cross_EA_EP = EAx*PEy - EAy*PEx
        #check if the point is inside the triangle
        # if the point is inside the triangle, the cross product should have same sign for all three edges
        # generate a boolean of size (N1,N2) for cross_AB_AP > 0 & cross_BC_BP > 0 & cross_CA_CP > 0


        mask[1,:,:][(cross_AB_AP > 0) & (cross_BC_BP>0) & (cross_CD_CP>0) & (cross_DE_DP>0) & (cross_EA_EP>0)] = 1
        mask[1,:,:][(cross_AB_AP < 0) & (cross_BC_BP<0) & (cross_CD_CP<0) & (cross_DE_DP<0) & (cross_EA_EP<0)] = 1
        mask[1,:,:][(cross_AB_AP == 0) & (cross_BC_BP>0) & (cross_CD_CP>0) & (cross_DE_DP>0) & (cross_EA_EP>0)] = 1
        mask[1,:,:][(cross_AB_AP == 0) & (cross_BC_BP<0) & (cross_CD_CP<0) & (cross_DE_DP<0) & (cross_EA_EP<0)] = 1
        mask[1,:,:][(cross_BC_BP == 0) & (cross_CD_CP>0) & (cross_DE_DP>0) & (cross_EA_EP>0) & (cross_AB_AP>0)] = 1
        mask[1,:,:][(cross_BC_BP == 0) & (cross_CD_CP<0) & (cross_DE_DP<0) & (cross_EA_EP<0) & (cross_AB_AP<0)] = 1
        mask[1,:,:][(cross_CD_CP == 0) & (cross_DE_DP>0) & (cross_EA_EP>0) & (cross_AB_AP>0) & (cross_BC_BP>0)] = 1
        mask[1,:,:][(cross_CD_CP == 0) & (cross_DE_DP<0) & (cross_EA_EP<0) & (cross_AB_AP<0) & (cross_BC_BP<0)] = 1
        mask[1,:,:][(cross_DE_DP == 0) & (cross_EA_EP>0) & (cross_AB_AP>0) & (cross_BC_BP>0) & (cross_CD_CP>0)] = 1
        mask[1,:,:][(cross_DE_DP == 0) & (cross_EA_EP<0) & (cross_AB_AP<0) & (cross_BC_BP<0) & (cross_CD_CP<0)] = 1
        mask[1,:,:][(cross_EA_EP == 0) & (cross_AB_AP>0) & (cross_BC_BP>0) & (cross_CD_CP>0) & (cross_DE_DP>0)] = 1
        mask[1,:,:][(cross_EA_EP == 0) & (cross_AB_AP<0) & (cross_BC_BP<0) & (cross_CD_CP<0) & (cross_DE_DP<0)] = 1

        if inclusion:
            mask[1,:,:] = 1-mask[1,:,:]
        # find the distance from P to the line AB
        # cross product of AB and AP = |AB|*|AP|*sin(theta)
        # distance = |cross product|/|AB|
        length_AB = np.sqrt(ABx**2 + ABy**2)
        length_BC = np.sqrt(BCx**2 + BCy**2)
        length_CD = np.sqrt(CDx**2 + CDy**2)
        length_DE = np.sqrt(DEx**2 + DEy**2)
        length_EA = np.sqrt(EAx**2 + EAy**2)
        #distance from P to AB
        d_AB = np.abs(cross_AB_AP)/length_AB
        #distance from P to BC
        d_BC = np.abs(cross_BC_BP)/length_BC
        #distance from P to CD
        d_CD = np.abs(cross_CD_CP)/length_CD
        #distance from P to DE
        d_DE = np.abs(cross_DE_DP)/length_DE
        #distance from P to EA
        d_EA = np.abs(cross_EA_EP)/length_EA
        # pick the minimum distance to set the anchoring
        d = np.stack([d_AB,d_BC,d_CD,d_DE,d_EA],axis=-1)
        index = np.argmin(d,axis=-1)
        # anchoring 0: along AB, find the angle of the normal vector of AB
        # find the normal vector of AB
        n_ABx = -ABy/length_AB
        n_ABy = ABx/length_AB
        Azi_n_AB = np.arctan2(n_ABy,n_ABx)
        # find the angle of the normal vector of BC
        n_BCx = -BCy/length_BC
        n_BCy = BCx/length_BC
        Azi_n_BC = np.arctan2(n_BCy,n_BCx)
        # find the angle of the normal vector of CD
        n_CDx = -CDy/length_CD
        n_CDy = CDx/length_CD
        Azi_n_CD = np.arctan2(n_CDy,n_CDx)
        # find the angle of the normal vector of DE
        n_DEx = -DEy/length_DE
        n_DEy = DEx/length_DE
        Azi_n_DE = np.arctan2(n_DEy,n_DEx)
        # find the angle of the normal vector of EA
        n_EAx = -EAy/length_EA
        n_EAy = EAx/length_EA
        Azi_n_EA = np.arctan2(n_EAy,n_EAx)
        # find the centroid of the triangle
        cx = (Ax + Bx + Cx + Dx + Ex)/5
        cy = (Ay + By + Cy + Dy + Ey)/5
        vx = x - cx
        vy = y - cy
        Azi = np.arctan2(vy,vx)
        if anchoring_type==1:
            # homeotropic anchoring
            mask[0,:,:][index==0] = Azi_n_AB
            mask[0,:,:][index==1] = Azi_n_BC
            mask[0,:,:][index==2] = Azi_n_CD
            mask[0,:,:][index==3] = Azi_n_DE
            mask[0,:,:][index==4] = Azi_n_EA
        elif anchoring_type==2:
            # planar anchoring
            mask[0,:,:][index==0] = 0.5*np.pi + Azi_n_AB
            mask[0,:,:][index==1] = 0.5*np.pi + Azi_n_BC
            mask[0,:,:][index==2] = 0.5*np.pi + Azi_n_CD
            mask[0,:,:][index==3] = 0.5*np.pi + Azi_n_DE
            mask[0,:,:][index==4] = 0.5*np.pi + Azi_n_EA
        elif anchoring_type==3:
            # arbitrary anchoring
            mask[0,:,:] = Azi 
        elif anchoring_type==4:
            mask[0,:,:] = Azi + 0.5*np.pi



        return mask