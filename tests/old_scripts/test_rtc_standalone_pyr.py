"""script to test rtc_standalone feature

Usage:
  test_rtc_standalone <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  -d, --devices devices      Specify the devices
"""

from docopt import docopt
from tqdm import tqdm

from carmaWrap import threadSync
import numpy as np
import shesha.sim
import shesha.init
from shesha.constants import CONST
import time
import matplotlib.pyplot as plt
plt.ion()

arguments = docopt(__doc__)
param_file = arguments["<parameters_filename>"]

# Get parameters from file
sim = shesha_sim.Simulator(param_file)
sim.config.p_loop.set_niter(100)

if arguments["--devices"]:
    devices = []
    for k in range(len(arguments["--devices"])):
        devices.append(int(arguments["--devices"][k]))
    sim.config.p_loop.set_devices(devices)

sim.init_sim()
nactu = sim.config.p_controller0.nactu
nvalid = sim.config.p_controller0.nvalid
offset = 0
p_wfs = sim.config.p_wfs0
p_centroider = sim.config.p_centroider0
scale = (p_wfs.Lambda * 1e-6 / sim.config.p_tel.diam) * \
            p_wfs.pyr_ampl * CONST.RAD2ARCSEC

rtc_standalone = shesha_init.rtc_standalone(sim.c, len(sim.config.p_wfss), nvalid, nactu,
                                            sim.config.p_centroider0.type,
                                            sim.config.p_controller0.delay, offset,
                                            scale)
rtc_standalone.set_cmat(0, sim.rtc.get_cmat(0))
rtc_standalone.set_decayFactor(0, np.ones(nactu, dtype=np.float32))
rtc_standalone.set_matE(0, np.identity(nactu, dtype=np.float32))
rtc_standalone.set_modal_gains(
        0,
        np.ones(nactu, dtype=np.float32) * sim.config.p_controller0.gain)

s_ref = np.zeros((sim.config.p_loop.niter, 2 * nvalid.sum()), dtype=np.float32)
s = s_ref.copy()
c = np.zeros((sim.config.p_loop.niter, nactu), dtype=np.float32)
c_ref = c.copy()
img = sim.wfs.get_pyrimg(0)
img = np.zeros((sim.config.p_loop.niter, img.shape[0], img.shape[1]), dtype=np.float32)

for k in tqdm(range(sim.config.p_loop.niter)):
    sim.next()
    img[k, :, :] = sim.wfs.get_pyrimg(0)
    s_ref[k, :] = sim.rtc.get_centroids(0)
    c_ref[k, :] = sim.rtc.get_com(0)

rtc_standalone.load_rtc_validpos(0, sim.config.p_wfs0._validsubsy,
                                 sim.config.p_wfs0._validsubsx)
rtc_standalone.set_open_loop(0, 1)
rtc_standalone.set_open_loop(0, 0)

rtc_time = 0
for k in tqdm(range(sim.config.p_loop.niter)):
    rtc_standalone.load_rtc_pyrimg(0, img[k, :, :].copy())
    a = time.time()
    rtc_standalone.do_centroids(0)
    threadSync()
    rtc_standalone.do_control(0)
    threadSync()
    rtc_standalone.save_com(0)
    rtc_time += (time.time() - a)
    s[k, :] = rtc_standalone.get_centroids(0)
    c[k, :] = rtc_standalone.get_com(0)

print("RTC speed : ", 1 / (rtc_time / sim.config.p_loop.niter), " fps")
