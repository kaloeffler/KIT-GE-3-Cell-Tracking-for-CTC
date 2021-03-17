import torch
from torch import nn
from scipy.ndimage import gaussian_filter
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GaussianKernel(nn.Module):
    def __init__(self, sigma, n_dims, padding=False):
        super().__init__()
        self.padding = padding
        self.n_dims = n_dims
        self.n_sigma = 3
        width = max(3, np.ceil(self.n_sigma*sigma))
        if width % 2 == 0:
            width += 1
        width = int(width)
        if self.padding:
            self.pad_w = width // 2
        else:
            self.pad_w = 0
        f = np.zeros([width for _ in range(self.n_dims)])  #
        if self.n_dims == 3:
            f[len(f) // 2, len(f) // 2, len(f) // 2] = 1
            self.conv = torch.conv3d
        else:
            f[len(f) // 2, len(f) // 2] = 1
            self.conv = torch.conv2d

        self.kernel_weights = torch.tensor(gaussian_filter(f, sigma=sigma, truncate=self.n_sigma),
                                           device=device).float().unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # todo: change to reflect mode padding
        return self.conv(x, self.kernel_weights, padding=self.pad_w)


class GaussianPyramid(nn.Module):
    def __init__(self, n_dims, ratio, n_levels):
        super().__init__()
        self.ratio = 0.75 if ratio > 0.98 or ratio < 0.4 else ratio
        self.n_levels = n_levels
        self.sigma_base = 1 / self.ratio - 1
        self.n = int(np.ceil(np.log(0.25) / np.log(self.ratio)))
        self.gaussian_pyramid = [GaussianKernel(self.sigma_base * (i + 1), n_dims, True) for i in range(self.n)]

    def forward(self, x):
        img_pyramid = []
        for i in range(self.n_levels):
            if (i+1) <= self.n:
                o = self.gaussian_pyramid[i](x)
                rescale_size = (torch.tensor(o.shape[2:]) * self.ratio**(i+1)).int()
                rescale_size[rescale_size < 1] = 1
                o = nn.functional.interpolate(o, tuple(rescale_size))
            else:
                o = self.gaussian_pyramid[self.n-1](img_pyramid[i-self.n+1])
                # todo: padd to small axis
                rescale_size = (torch.tensor(o.shape[2:]) * self.ratio ** (self.n)).int()
                rescale_size[rescale_size < 1] = 1
                o = nn.functional.interpolate(o, tuple(rescale_size))
            img_pyramid.append(o)
        img_pyramid = img_pyramid[::-1]
        img_pyramid.append(x)
        return img_pyramid # so coarsest scale is in front


class BronxOpticalFlow(nn.Module):
    def __init__(self, n_dims, ratio, n_levels, n_outer_iter, n_inner_iter=10, n_sor_iterations=10,alpha=0.3):
        super().__init__()
        self.n_dims = n_dims
        self.ratio = ratio
        self.n_levels = n_levels
        self.n_outer_iter = n_outer_iter
        self.n_inner_iter = n_inner_iter
        self.n_sor_iterations = n_sor_iterations
        self.alpha = alpha
        self.gaussian_pyramid = GaussianPyramid(self.n_dims, self.ratio, self.n_levels)

        # set up derivative and smoothing kernels
        s = torch.tensor([0.02, 0.11, 0.74, 0.11, 0.02])
        diff = -1/12 * torch.tensor([-1, 8, 0, -8, 1])

        if self.n_dims == 3:
            spatial_weights = [diff.reshape(-1, 1, 1), diff.reshape(1, -1, 1), diff.reshape(1, 1, -1)]
        else:
            spatial_weights = [diff.reshape(-1, 1), diff.reshape(1, -1)]

        self.spatial_kernel_t = spatial_weights[0].unsqueeze(0).unsqueeze(0).to(device)
        self.spatial_kernel_x = spatial_weights[0].unsqueeze(0).unsqueeze(0).to(device)
        self.spatial_kernel_y = spatial_weights[1].unsqueeze(0).unsqueeze(0).to(device)
        if self.n_dims == 3:
            self.spatial_kernel_z = spatial_weights[2].unsqueeze(0).unsqueeze(0).to(device)
            self.smoothing_kernel = torch.tensor(s.reshape(-1, 1, 1) * s.reshape(1, -1, 1) * s.reshape(1, 1, -1),
                                                 device=device).float().unsqueeze(0).unsqueeze(0)
            self.conv = nn.functional.conv3d
        else:
            self.smoothing_kernel = torch.tensor(s.reshape(-1, 1) * s.reshape(1, -1),
                                                 device=device).float().unsqueeze(0).unsqueeze(0)
            self.conv = nn.functional.conv2d
            self.spatial_kernel_z = None
        self.smoothing_kernel /= self.smoothing_kernel.sum()

    def calc_derivative_2D(self, image_1, image_2):
        img_1 = self.conv(image_1, self.smoothing_kernel, padding=2)
        img_2 = self.conv(image_2, self.smoothing_kernel, padding=2)

        dt = img_2 - img_1

        fused = img_1*0.4 + img_2*0.6
        dx = self.conv(fused, self.spatial_kernel_x, padding=2)
        dy = self.conv(fused, self.spatial_kernel_y, padding=2)
        return dx, dy, dt

    def forward(self, x):
        pass

    def calc_derivatives_2D(self, img1,img2):
        pass
        #dx
        #dy
        #dt
       # dxx
        #dyy
        #dxt
        #dyt =

    def calc_flow_2d(self, image_1, image_2, u, v):
        pyramid_img_1 = self.gaussian_pyramid(image_1)
        pyramid_img_2 = self.gaussian_pyramid(image_2)

        u, v = torch.zeros_like(image_1, device=device)
        for i, images in zip(pyramid_img_1, pyramid_img_2):
            im1, im2 = images
            # todo: upscale u, v -> interpolate

            #init du,dv again as variables
            du = torch.autograd.Variable(torch.zeros_like(u, device=device), requires_gtad=True)
            dv = torch.autograd.Variable(torch.zeros_like(v, device=device), requires_gtad=True)

            derivatives = self.calc_derivative_2D(im1, im2)
            # psi'data

            # psi' smooth

            # equations

            #do n LFBGS steps (== innerloop)
            optim = torch.optim.LBFGS(params=[du, dv])
            optim.zero_grad()
            for _ in range(self.n_inner_iter):
                loss = torch.nn.functional.l1_loss(..., 0)
                loss.backward()
                optim.step()
            # update u, v
            du = du.detach()
            dv = dv.detach()
            u += du
            v += dv

@torch.jit.script
def psi(x, epsilon=0.001):
    return torch.sqrt(x**2 + epsilon**2)


@torch.jit.script
def psi_derivative(x, epsilon):
    return torch.pow(x**2 + epsilon**2, -0.5) * 2 * x
