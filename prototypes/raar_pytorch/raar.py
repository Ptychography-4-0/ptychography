from reconst_lite import CDI_Reconst
from math import pi

import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for RAAR Reconstruction")
    parser.add_argument("--input", action="store", type=str, default='intensity.txt',help="Path for intensity txt file")
    parser.add_argument("--output",action="store", type=str, default='reconstruction.txt',help="Path for image output file")
    parser.add_argument("--gpu",action="store", type=int, default=1,help="1 - Use GPU, 0 - Use CPU")
    parser.add_argument("--seed",action="store", type=int, default=0,help="Seed for initial phase")
    
    parser.add_argument("--beta", action="store", type=float, default=0.95,help="Relaxation Parameter [0,1]")
    parser.add_argument("--AC_threshold",action="store", type=float, default=0.04, help="Threshold for autocorrelation" )
    parser.add_argument("--obj_threshold",action="store", type=float, default=0.22, help="Threshold for support updates")
    
    parser.add_argument("--iter_total", action="store", type=int, default=300, help="Number of total iterations")
    parser.add_argument("--iter_update", action="store", type=int, default=25, help="Number of iteration for support update")
    parser.add_argument("--iter_cycle", action="store", type=int, default=250)
    parser.add_argument("--RAAR_num", action="store", type=int, default=250, help="Number of RAAR iterations")
    

args = parser.parse_args()

intensity = torch.from_numpy(np.loadtxt(args.input))

if torch.cuda.is_available() and args.gpu == 1:
    device = 'cuda:0'
else:
    device = 'cpu:0'
    
ReconstructionClass = CDI_Reconst(device=device,dtype=torch.float32)

ReconstructionClass.beta = args.beta  # beta value of the RAAR algorithm ref. [1, 2]
ReconstructionClass.AC_threshold = args.AC_threshold  # threshold of the autocorrelation to start with ref. [3]
ReconstructionClass.iter_total = args.iter_total  # total number of iterations
ReconstructionClass.iter_update = args.iter_update  # iterations before updating support ref. [3]
ReconstructionClass.obj_threshold = args.obj_threshold  # threshold on the current object to update the suport ref. [3]
ReconstructionClass.iter_cycle = args.iter_cycle  # length of one cycle if we want to mix error reduction and RAAR (or RAAR)
ReconstructionClass.RAAR_num = args.RAAR_num  # length of RAAR (or RAAR) per cycle 

np.random.seed(seed=args.seed)
phi_rand = -pi + 2 * pi * np.random.rand(intensity.shape[0],intensity.shape[1])
phi_rand = torch.from_numpy(phi_rand)
phi_rand = torch.tensor(phi_rand,device=ReconstructionClass.device,dtype=ReconstructionClass.float)

final_object = ReconstructionClass.reconst(diff_int_t=intensity.to(ReconstructionClass.device),
                                           phi_rand=phi_rand)

reconstruction = np.sqrt(final_object.cpu().numpy()[:,:,0]**2 + final_object.cpu().numpy()[:,:,1]**2)
plt.imshow(reconstruction,"gray")
plt.axis("off")
plt.savefig(args.output)