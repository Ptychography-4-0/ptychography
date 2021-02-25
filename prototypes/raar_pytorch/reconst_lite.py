import torch
import numpy as np
from util import torch_angle, torch_fftshift, torch_ifftshift
from math import pi
import warnings
warnings.filterwarnings("ignore")

class CDI_Reconst():
    def __init__(self, device='cpu:0', dtype=torch.float64):
        self.device = torch.device(device)  # speicify if cpu or gpu use
        self.float = dtype

        self.beta = 0.95 # beta value of the RAAR algorithm ref. [1, 2]
        self.AC_threshold = 0.04  # threshold of the autocorrelation to start with ref. [3]
        self.iter_total = 300  # total number of iterations
        self.iter_update = 25  # iterations before updating support ref. [3]
        self.obj_threshold = 0.22  # threshold on the current object to update the suport ref. [3]
        self.sigma = 3  # standard deviation of the Gaussian function used to convolve with the object ref. [3]
        self.iter_cycle = 250  # length of one cycle if we want to mix error reduction and RAAR (or RAAR)
        self.RAAR_num = 250  # length of RAAR (or RAAR) per cycle

        self.single_zero = torch.zeros(1, dtype=self.float, device=self.device)  # parameter for comparing
        self.single_one = torch.ones(1, dtype=self.float, device=self.device)  # parameter for comparing
        self.global_iter = 0

        # Gauss Smoothing Filter Parameters
        self.x_t = torch.arange(1, 26, 1, dtype=self.float, device=self.device)
        self.y_t = torch.arange(1, 26, 1, dtype=self.float, device=self.device)
        self.x0_t = torch.tensor([(len(self.x_t) // 2) + 1], dtype=self.float, device=self.device)
        self.y0_t = torch.tensor([(len(self.y_t) // 2) + 1], dtype=self.float, device=self.device)
        self.w_t = torch.from_numpy(np.array([2.3548 * self.sigma],dtype=np.float32)).to(self.device)


    def forward(self, P_filtered_t):
        S_t_1 = torch.stack((torch_ifftshift(P_filtered_t[:, :, 0]), torch_ifftshift(P_filtered_t[:, :, 1])), -1)
        S_t_2 = torch.ifft(S_t_1, signal_ndim=2)
        S_t_3 = torch.stack((torch_ifftshift(S_t_2[:, :, 0]), torch_ifftshift(S_t_2[:, :, 1])), -1)
        S_t = S_t_3
        return S_t

    def backward(self, S_filtered_t):
        P_t_1 = torch.stack((torch_fftshift(S_filtered_t[:, :, 0]), torch_fftshift(S_filtered_t[:, :, 1])), -1)
        P_t_2 = torch.fft(P_t_1, signal_ndim=2)
        P_t_3 = torch.stack((torch_fftshift(P_t_2[:, :, 0]), torch_fftshift(P_t_2[:, :, 1])), -1)
        P_t = P_t_3
        return P_t

    def get_support(self, S, threshold):
        m = torch.max(S)
        in_support_t = torch.where(torch.abs(S) > threshold * m, self.single_one, self.single_zero)
        out_of_support_t = torch.where(in_support_t == 1.0, self.single_zero, self.single_one)
        pixels_out_supp_t = torch.sum(out_of_support_t)
        return in_support_t, out_of_support_t, pixels_out_supp_t

    def get_obj_AC(self, sample):
        obj_AC_t = torch_ifftshift(torch.ifft(torch_ifftshift(sample), signal_ndim=2))
        obj_AC_t = torch.sqrt(obj_AC_t[:, :, 0] ** 2 + obj_AC_t[:, :, 1] ** 2)
        return obj_AC_t

    def euler_form(self, diff_int_t, phase_rand_t):
        magn = torch.sqrt(diff_int_t)
        sin = torch.sin(phase_rand_t)
        cos = torch.cos(phase_rand_t)

        full_diff_field_t = torch.stack((magn * cos, magn * sin), -1)
        diff_amp_t = torch.sqrt(((magn * cos) ** 2 + (magn * sin) ** 2))
        return full_diff_field_t, diff_amp_t

    def gauss_filter(self, S_t):
        S_abs_t = torch.sqrt(S_t[:, :, 0] ** 2 + S_t[:, :, 1] ** 2)
        if self.w_t > 0.5 * self.sigma * 2.3548:  # ref. [3]
            self.w_t = self.w_t * 0.99
        else:
            w_t = 1.5 * 2.3548

        v_1_t = torch.exp(
            torch.mul(
                torch.mul(
                    torch.Tensor([-4]).to(self.device),
                    torch.log(torch.tensor([2], dtype=self.float, device=self.device))),
                ((self.y_t - self.y0_t) / self.w_t) ** 2))
        v_2_t = torch.exp(
            torch.mul(
                torch.mul(
                    torch.Tensor([-4]).to(self.device),
                    torch.log(torch.tensor([2], dtype=self.float, device=self.device))),
                ((self.x_t - self.x0_t) / self.w_t) ** 2))
        G_t = torch.ger(v_1_t, v_2_t).unsqueeze(0).unsqueeze(0)

        Gauss_Smooth_Filter = torch.nn.Conv2d(S_abs_t.size(1),
                                              G_t.size(1),
                                              kernel_size=25,
                                              padding=(12, 12),
                                              stride=(1, 1),
                                              bias=False)
        Gauss_Smooth_Filter.weight = torch.nn.Parameter(G_t, requires_grad=False)
        Supp_Conv_t = Gauss_Smooth_Filter(S_abs_t.unsqueeze(0).unsqueeze(0))
        Supp_Conv_t = Supp_Conv_t.reshape(S_t.size()[0], S_t.size()[1])
        return Supp_Conv_t

    def fourier_constraint_I(self, P_t, diff_amp_t):
        P_filtered_t_real = diff_amp_t * torch.cos(torch_angle(P_t))
        P_filtered_t_imag = diff_amp_t * torch.sin(torch_angle(P_t))
        P_filtered_t = torch.stack((P_filtered_t_real, P_filtered_t_imag), -1)
        return P_filtered_t

    def RAAR(self, S_t, S_prev_t, in_support_t, out_of_support_t, beta):
        """
        Update function of the reconstruction.
        Preprared to use NN for Updating and also to save images before and after updating.
        It is necessary to do some pre runs before using NN.
        """
        

        obj_in_supp_t = S_t * torch.stack((in_support_t, in_support_t), -1)
        obj_out_supp_t = (beta * S_prev_t + (1 - 2 * beta) * S_t) * \
        torch.stack((out_of_support_t, out_of_support_t), -1)
        S_filtered_t = obj_in_supp_t + obj_out_supp_t
        
        return S_filtered_t


    def object_error(self, S_t, out_of_support_t):
        sum_total_t = torch.sum(torch.flatten(torch.abs(torch.mul(S_t, S_t))))
        out_of_support_t = out_of_support_t.float()
        S_t = S_t.float()
        sum_out_t = torch.sum((S_t[:, :, 0] * out_of_support_t) * (S_t[:, :, 0] * out_of_support_t) + \
                              (S_t[:, :, 1] * out_of_support_t) * (S_t[:, :, 1] * out_of_support_t))

        object_error_t = torch.sqrt(sum_out_t / sum_total_t)
        return object_error_t

    def reconst(self,diff_int_t, phi_rand=None):
        diff_int_t = torch.tensor(diff_int_t, dtype=self.float, device=self.device)
        
        diff_int_t = torch.max(diff_int_t, self.single_zero)
        diff_int_t = torch.transpose(diff_int_t, dim0=0, dim1=1)
        
        sqrInt_t = torch.max(self.single_zero, diff_int_t)
        sqrInt_t = torch.sqrt(sqrInt_t)
        sqrInt_t = torch.transpose(sqrInt_t, dim0=0, dim1=1)
        
        if phi_rand is None:
            phi_rand = -pi + 2 * pi * torch.rand((diff_int_t.size()[0], diff_int_t.size()[1]))
        
        phase_rand_t = torch.tensor(phi_rand, dtype=self.float, device=self.device)
        
        real = torch.zeros((diff_int_t.size()[0], diff_int_t.size()[1]), dtype=self.float, device=self.device)
        sample = torch.stack((real, diff_int_t), dim=-1, out=real)
        
        with torch.no_grad():            
            fourier_error_t = torch.zeros(self.iter_total, dtype=self.float, device=self.device)
            object_error_t = torch.zeros(self.iter_total, dtype=self.float, device=self.device)

            num_of_updates = 0

            full_diff_field_t, diff_amp_t = self.euler_form(diff_int_t=diff_int_t, phase_rand_t=phase_rand_t)
            obj_AC_t = self.get_obj_AC(sample=sample)
            in_support_t, out_of_support_t, _ = self.get_support(obj_AC_t, self.AC_threshold)
            in_support_t = torch.tensor(in_support_t, dtype=self.float, device=self.device)
            out_of_support_t = torch.tensor(out_of_support_t, dtype=self.float, device=self.device)
            
            P_filtered_t = torch.tensor(full_diff_field_t, dtype=self.float, device=self.device)

            for i in range(1, self.iter_total + 1):
                S_t = self.forward(P_filtered_t) # to real space
                S_t = torch.tensor(S_t, dtype=self.float, device=self.device)
                object_error_t[i - 1] = self.object_error(S_t, out_of_support_t)

                
                if i % self.iter_update == 0:
                    num_of_updates += 1
                    Supp_Conv_t = self.gauss_filter(S_t)
                    in_support_t, out_of_support_t, pixels_out_supp_t = self.get_support(Supp_Conv_t, self.obj_threshold)
                    in_support_t = torch.tensor(in_support_t, dtype=self.float, device=self.device)
                    out_of_support_t = torch.tensor(out_of_support_t, dtype=self.float, device=self.device)

                if (i % self.iter_cycle) < self.RAAR_num:
                    if i == 1:
                        S_filtered_t = S_t
                    else:
                        S_filtered_t = self.RAAR(S_t, S_prev_t, in_support_t, out_of_support_t, self.beta)
                        
                    S_prev_t = S_filtered_t
                    S_prev_t = torch.tensor(S_prev_t, dtype=self.float, device=self.device)
                    
                P_t = self.backward(S_filtered_t) # to fourier space
                P_filtered_t = self.fourier_constraint_I(P_t, sqrInt_t) # get new P
                P_filtered_t = torch.tensor(P_filtered_t, dtype=self.float, device=self.device)
                final_object = S_t
                self.global_iter += 1
            return final_object