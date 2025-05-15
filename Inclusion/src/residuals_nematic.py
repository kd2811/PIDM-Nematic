import torch
from src.grad_utils import *
import einops as ein
        
class ResidualsNematic:
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary, reverse_d1=None, device = 'cpu', bcs = 'none', domain_length = 1., residual_grad_guidance = False, use_ddim_x0 = False, ddim_steps = 0, K = 1, gamma = 2, W = 20., Wp = 20.):
        """
        Initialize the residual evaluation.

        :param model: The neural network model to compute the residuals for.
        :param n_steps: Number of steps for time discretization.
        """
        self.gov_eqs = 'Nematic'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2
        self.K = K
        self.gamma = gamma
        self.W = W
        self.Wp = Wp
        
        if bcs == 'periodic':
            self.periodic = True


        


        self.grads = GradientsHelper(d0=1, d1=1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.relu = torch.nn.ReLU()

        self.pixels_per_dim = pixels_per_dim


        self.device = device

        self.residual_grad_guidance = residual_grad_guidance

        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps



    def compute_residual(self, input, label_input, reduce = 'none', return_model_out = False, sample = False, ddim_func = None, pass_through = False):
        Azi = label_input[:,0] # [batch_size, pixels_per_dim, pixels_per_dim] anchoring
        Mask = label_input[:,1] # [batch_size, pixels_per_dim, pixels_per_dim] index 1 --> no need

        Seq = torch.sqrt(torch.tensor(2.0)) / self.gamma
        Qxx_anchoring = Seq * torch.cos(Azi*2)
        Qxy_anchoring = Seq * torch.sin(Azi*2)
        Qxx_0 = Seq * torch.ones_like(Qxx_anchoring) 
        Qxy_0 = Seq * torch.zeros_like(Qxy_anchoring)
        Qxx_anchoring[:,:,[0,-1]] = Qxx_0[:,:,[0,-1]]
        Qxx_anchoring[:,[0,-1],:] = Qxx_0[:,[0,-1],:]

        Qxy_anchoring[:,:,[0,-1]] = Qxy_0[:,:,[0,-1]]
        Qxy_anchoring[:,[0,-1],:] = Qxy_0[:,[0,-1],:]

        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
        else:
            assert len(input[0]) == 2 and isinstance(input[0], tuple), 'Input[0] must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(input[0])
            cond = torch.stack([Qxx_anchoring, Qxy_anchoring], dim=1) # [batch_size, 2, pixels_per_dim, pixels_per_dim]
            kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=torch.float32).to(self.device)
            index = nn.functional.conv2d(label_input[:,1].unsqueeze(1), kernel, stride=1, padding=1).repeat(1,2,1,1) #[B,2,H,W]
            # index *= label_input[:,1]
            cond = torch.where((index<4)&(index>0), cond, torch.zeros_like(cond)) # [B,2,H,W]
            if self.residual_grad_guidance:
                # assert not self.use_ddim_x0, 'Residual gradient guidance is not implemented with sample estimation for residual.'
                # noisy_in.requires_grad = True
                # residual_noisy_in = self.compute_residual(generalized_b_xy_c_to_image(noisy_in),label_input, pass_through = True)['residual']
                # dr_dx = torch.autograd.grad(residual_noisy_in.abs().mean(), noisy_in)[0]
                # cond *= label_input[:,1]

                if sample:
                    x0_pred = self.model.forward_with_guidance_scale(noisy_in, time, Mask_input = label_input[:,1], cond = cond, guidance_scale = 3.) # There is no mentioning of value for the guidance scale in the paper and repo?!?
                    model_out = x0_pred
                else:
                    x0_pred = self.model(noisy_in, time, Mask_input = label_input[:,1], cond = cond, null_cond_prob = 1)
                    model_out = x0_pred
            else:
                if self.use_ddim_x0:
                    x0_pred, model_out = ddim_func(noisy_in, time, self.model, noisy_in.shape, self.ddim_steps, 0.)
                else:
                    # print('noisy_in shape',noisy_in.shape)
                    # print('time shape',time.shape)
                    # print('label_input shape',label_input.shape)
                    x0_pred = self.model(noisy_in, time, Mask_input = label_input[:,1], cond = None)
                    model_out = x0_pred

        assert len(x0_pred.shape) == 4, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape
        # print(f'x0_pred shape at time {time}',x0_pred.shape)
        # assert x0_pred.shape == [batch_size, 2, pixels_per_dim, pixels_per_dim]
        batch_size, label_dim, pixels_per_dim, pixels_per_dim = label_input.shape
        Qxx = x0_pred[:, 0]  # [batch_size,1, pixels_per_dim, pixels_per_dim]
        Qxy = x0_pred[:, 1]  # [batch_size,1, pixels_per_dim, pixels_per_dim]
        Qxx = torch.where(Mask == 0, Qxx_anchoring, Qxx)
        Qxy = torch.where(Mask == 0, Qxy_anchoring, Qxy)

        Qxx_d0 = self.grads.stencil_gradients(Qxx, mode='d_d0') # dQxx/dx
        Qxx_d1 = self.grads.stencil_gradients(Qxx, mode='d_d1') # dQxx/dy
        Qxy_d0 = self.grads.stencil_gradients(Qxy, mode='d_d0') # dQxy/dx
        Qxy_d1 = self.grads.stencil_gradients(Qxy, mode='d_d1') # dQxy/dy

        thermo = (-0.5*1/self.gamma**2) * (Qxx**2+Qxy**2) + 0.25 * ((Qxx**2) + (Qxy**2))**2
        elastic = self.K * (Qxx_d0**2 + Qxx_d1**2+ Qxy_d0**2 + Qxy_d1**2)
        penalty = (Seq**2)/4-(Seq**4)/16
        free_energy = (thermo + elastic + penalty)*Mask
        free_energy = free_energy.unsqueeze(1)

        x0_pred = generalized_image_to_b_xy_c(x0_pred)  
        # from [batch_size, 2, pixels_per_dim, pixels_per_dim] to [batch_size, pixels_per_dim*pixels_per_dim, 2]
        free_energy = generalized_image_to_b_xy_c(free_energy)
        # from [batch_size, 1, pixels_per_dim, pixels_per_dim] to [batch_size, pixels_per_dim*pixels_per_dim, 1]

        # obtain equilibrium equations for residual
        eq_0 = free_energy
        residual = eq_0  
        
        # manually add BCs
        # reshape output to match image shape
        # grad_p_img = generalized_b_xy_c_to_image(grad_Q)
        # grad_Q.shape = [batch_size, 4, pixels_per_dim, pixels_per_dim]

        ### NOTE: BC constraint
        Qxx_0 = Seq * torch.cos(torch.tensor(np.pi*2)) * torch.ones_like(Qxx) 
        Qxy_0 = Seq * torch.sin(torch.tensor(np.pi*2)) * torch.ones_like(Qxy)
        kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=torch.float32).to(self.device)
        index = nn.functional.conv2d(label_input[:,1].unsqueeze(1), kernel, stride=1, padding=1).squeeze(1)
        Qxx_0 = torch.where((index<4)&(index>1), Qxx_anchoring, Qxx_0)
        Qxy_0 = torch.where((index<4)&(index>1), Qxy_anchoring, Qxy_0)
        deviate_xx = (- Qxx + Qxx_0)**2
        deviate_xy = (- Qxy + Qxy_0)**2
        free_energy_bc = torch.zeros_like(Qxx)
        free_energy_bc_particle = torch.zeros_like(Qxx)
        free_energy_bc_particle = self.Wp*0.5*self.W*(deviate_xx+deviate_xy) # ymin ybottom
        free_energy_bc[:,0,:] = 0.5*self.W*(deviate_xx[:,0,:]+deviate_xy[:,0,:]) # ymin ybottom
        free_energy_bc[:,-1,:] = 0.5*self.W*(deviate_xx[:,-1,:]+deviate_xy[:,-1,:]) # ymax ytop
        free_energy_bc[:,:,0] = 0.5*self.W*(deviate_xx[:,:,0]+deviate_xy[:,:,0])  # xmax xleft
        free_energy_bc[:,:,-1] = 0.5*self.W*(deviate_xx[:,:,-1]+deviate_xy[:,:,-1]) # ymin xright
        free_energy_bc = torch.where((index<4)&(index>0), free_energy_bc_particle, free_energy_bc)
        # print('shape before mask',free_energy_bc.shape)
        free_energy_bc *= Mask
        # print('shape before unsqueeze',free_energy_bc.shape)
        free_energy_bc = free_energy_bc.unsqueeze(1)
        # print('shape after unsqueeze',free_energy_bc.shape)
        free_energy_bc = generalized_image_to_b_xy_c(free_energy_bc)
        # print('shape after generalized_image_to_b_xy_c',free_energy_bc.shape)
        # from residual_bc.shape = [batch_size, 1, pixels_per_dim, pixels_per_dim] to [batch_size, pixels_per_dim*pixels_per_dim, 1]
        # eq_0.shape = [batch_size, pixels_per_dim*pixels_per_dim, 1]     
        residual = torch.cat([eq_0, free_energy_bc], dim=-1)
        # from residual.shape [batch_size, pixels_per_dim*pixels_per_dim, 2]
        output = {}
        output['residual'] = residual

        if return_model_out:
            output['model_out'] = model_out

        if reduce == 'full':
            # mean over all items in dict
            return {k: v.mean() for k, v in output.items()}
        elif reduce == 'per-batch':
            # mean over all but first dimension (batch dimension) 
            # only if tensor has more than one dimension and key is not 'model_out'
            return {k: v.mean(dim=tuple(range(1, v.ndim))) if v.ndim > 1 and (k != 'model_out' and k != 'residual') else v for k, v in output.items()}
        elif reduce == 'none':
            # return as-is
            return output
        else:
            raise ValueError('Unknown reduction method.')