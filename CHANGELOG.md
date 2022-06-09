# COMPASS Change logs

- [COMPASS Change logs](#compass-change-logs)
  - [Release v5.3.0](#release-v530)
  - [Release v5.2.1](#release-v521)
  - [Release v5.2](#release-v52)
  - [Release v5.1](#release-v51)
  - [Release v5.0](#release-v50)
  - [Release v4.4.2](#release-v442)
  - [Release v4.4.1](#release-v441)
  - [Release v4.4.0](#release-v440)
  - [Release v4.3.2](#release-v432)
  - [Release v4.3.1](#release-v431)
  - [Release v4.3.0](#release-v430)
  - [Release v4.2.0](#release-v420)
  - [Release v4.1.0](#release-v410)
  - [Release v4.0.0](#release-v400)
  - [Release v3.4](#release-v34)
  - [Release v3.3](#release-v33)
  - [Release v3.2](#release-v32)
  - [Release v3.0](#release-v30)
  - [Release v2.0](#release-v20)
  - [Release v1.1](#release-v11)

## Release v5.3.0

- **New feature**: User-defined phase screens circular buffer in the telescope. Allows to put a cube of phase screens as an additional input of the telescope.
- **New feature**: Field stop can now be used with SHWFS. Does not simulate flux reduction.
- Modification of the custom DM construction: see the [new tutorial dedicated to this topic](../tutorials/2022-06-09-custom-dm)
- Support for clang compiler
- Fix a bug of the KL basis computation due to the move from MAGMA to CUSOLVER library
- Minor fixes

## Release v5.2.1

COMPASS v5.2.1 release notes :

- Add new geometric WFS method which takes into account pupil masks
- Bug fixes:
    - Wind interpolation
    - Slope-based pyramid centroider "pyr"
    - Pupil transpose
    - Generic linear controller

## Release v5.2

COMPASS v5.2 release notes :

- Add generic linear controller, see documentation
- Remove unused `nvalid` argument from controller signatures
- Debug WFS NCPA that were applied twice
- Debug RTC standalone
- Debug P2P GPU access
- Debug roket script in guardians
- Pytests debug
- Debug centroider to make it right with any number of sub-aperture and pixel per subap
- Debug geometric slope computation (edited)
- Rework CMakeFiles and conan dependencies

## Release v5.1

COMPASS v5.1 release notes :

- New class ParamConfig to handle parameters configuration : supervisor constructor requires an instance of it now. This change
- Multi GPU controller generic improvements
- Standalone RTC debug + pytests
- Add leaky factor in the generic controller
- Add [CLOSE](https://arxiv.org/abs/2103.09921) algorithm implementation
- Multi controllers support in the supervisor
- Sub-pixels move for phase screens
- GuARDIANS package updated
- Code documentation update

## Release v5.0

The goal of this release is to improve the code quality.
This is a major (and disruptive) update, there are changes in most supervisor codes : scripts need to be updated according to these changes.

COMPASS is more and more used by the AO community. With time, the amount of contributions external from the core team had become significant, especially in the user layer defined by the shesha package.
With its legacy architecture, the readability of the source code became messy, and the long term maintanability was penalized.

With this release, we propose a new architecture for the user top-level interface, i.e. supervisors. The goal is to become fully modular while increasing the level of documentation and code readability.

The architecture export the abstraction level from the supervisor to so called "components" : a supervisor is now a set of components in charge of the implementation of each AO module, such as Wfs, Dm, Rtc, etc...
Then, the supervisor defines the sequence of execution of each component. This approach aims to reach "plug-and-play" components, where third party library could be used to implements some components and still interface it with compass ones.

For this, we also define code guidelines that must be followed by each contribution to ensure self consistence. Usage of unit tests through pytest also becomes mandatory : each newly added feature accessible at user level has to come with its set of unit tests.

IMPORTANT NOTE : due to the very short available development time of the core COMPASS developers, this version does not fulfill all our expectations yet. This is a first step, but full modularity also requires to rework core software layers, and this is time consuming. Version 5 will be updated through time and the final version before next major release will reach the goals.

- PEP 8 application in `shesha.supervisor`
- PEP 8 style in `carma` / `sutra`
- Add CUDA 11 support
- Make MAGMA an optional dependency
- New supervisor architecture using components and optimizers
- Remove `get_centroids` --> becomes `get_slopes`
- Old behavior of `get_slopes` was to call `compute_slopes`, which compute and return. Directly use `compute_slopes` instead
- Remove `get_all_data_loop` functions from `abstractSupervisor` and `AoSupervisor` : unused
- Remove `computeImatModal` from `AoSupervisor` : not used and not implemented
- `set_gain` was able to set mgain also depending on the parameter given. Change the function to be more explicit : `set_gain only` set scalar loop gain while `set_modal_gain` set the modal gain vector
- Rename `set_mgain` to `set_modal_gain`
- Rename get_mgain to `get_modal_gain`
- Remove `write_config_on_file` from `AoSupervisor`, and rename it `getConfigFab` in `canapassSupervisor`
- Rename `set_global_r0` to `set_r0`
- Rename `getIFsparse` to `get_influ_basis` (to make difference with `get_influ`)
- Rename `getIFtt` to `get_tt_influ_basis`
- Rename `getIFdm` to `compute_influ_basis`
- Remove `getTarAmpliPup` (unused)
- Remove `reset` function, use `reset_simu` instead
- Remove `setModalBasis` : unused
- Remove `computePh2ModesFits` : unused
- Rename `setPyrSourceArray` in `set_pyr_modulation_points`
- `set_pyr_modulation` becomes `set_pyr_modulation_ampli`
- Signature changes in `setPyr*Source` : wfs_index first is mandatory
- Add new parameter in PWFS : `p_wfs._pyr_scale_pos` to store the scale applied to `pyr_cx` and `pyr_cy` before upload. Useful for Milan functions
- Rename recordCB in `record_ao_circular_buffer`
- Signature changes for `set_fourier_mask`, `set_noise`, `set_gs_mask` : wfs_index as first argument and mandatory
- Remove `compute_wfs_images` : not used
- Rename `set_dm_shape_from` into `set_command`
- Add new parameter in PDMS : `p_dms[0]._dim_screen` to store the dimension of the DM shape screen
- Add components module : this module defines classes to handle implementation of AO components such as Wfs, Dm, Rtc and so on. For now, only compass implementations are coded, but an abstraction for each component will be developed to allow third party library implementation
- Add optimizers module : this module defines classes to operate on the supervisor components for AO optimization. It could include many algorithms, define in some "thematic" class. For now, it includes `ModalBasis` class (for modal basis computations) and `Calibration` class (for interaction and command matrices). User defined algorithms that do not fit into one of those classes should be written in an other new class with an explicit name to be used by the supervisor.
- Remove `abstractSupervisor`
- Remove `aoSupervisor`
- Remove the simulator module : methods have been moved into the right component
- Add `CONTRIBUTING.md` file to define code guidelines
- Add unit tests for each method accessible from a `compassSupervisor` using pytest. Each contribution should define a new unit test
- Add templates for issue and merge request

## Release v4.4.2

- Fix COMPASS Ray tracing with LGS bug
- New feature : change wind on the fly
- Debug modal opti with Btt
- Add custom DM feature
- Add `carma_magma_gemv` method in libcarma

## Release v4.4.1

- handle different shape for raw images and cal images (using the LUT feature)
- possibility to attach a stream to centroiders
- opimization of pyramid HR wfs
- debug ```do_centroids_ref``` method
- debug ```init_custom_dm``` method
- add ```requirements*.txt``` file to install python dependencies
- add ```Jenkinsfile```
- update Doxyfile script and documentation rst
- generate gcov trace in debug

## Release v4.4.0

- Debug issue with Kepler architecture
- Multi GPU controller reworked
- Update pages-doc
- Add useful keyworks in ```rtc_cacao``` loopframe
- Add ```reset_coms``` function in ```sutra_controller```
- Update Jenkinsfile

## Release v4.3.2

- Add Jenkinsfile
- Debug compilation without octopus
- Improvement of multi-GPU controller : only run on P2P enabled GPU available in the context
- Bug fix in getSlopeGeom; chekc dims setPyrModulation when p_wfs._halfxy is 3D
- Cleanup + refactor of sutra_target raytracing

## Release v4.3.1

- Add spider rotation and circular obstruction for ELT-like pupils
- New feature : image with the selected pixels of the maskedpix centroider
- Debug maskedpix to divide the image by the mean value of the pixels instead of the sum
- Fix maskedpix get_type method
- Add cone effect for the altitude conjugated DM in case of LGS WFS

## Release v4.3.0

- change license to GNU LGPL-v3
- add Turing support
- add AoSupervisor class on top of CompassSupervisor and BenchSupervisor
- SH WFS can handle big subapertures (before it was limited to 20x20)
- add LUTpix in calibration process to reorder pixels
- possibility to compute target SR fitted on a0 sinc function
- modification of pyramid centroider to use CUB

## Release v4.2.0

- add pyramid focal plane visualization
- add md report generation in check.py
- update documentation generation using doxygen
- add natural integration of FP16 and complex in python
- add KECK aperture type
- better ELT Pupil generation
- drop BRAHMA, use CACAO interface instead
- computes PSF in every iteration
- add frame calibration
- remove KALMAN support
- add pyramid modulation weight
- add ```filter_TT``` parameter in centroider
- add modal integrator command law

for Internal developments:

- add custom pyr modulation script to handle natural guide dual-stars and resolved NGS
- add hunter integration optional

## Release v4.1.0

- Add multiple input/computation/output type for RTC module
- uniformize axis in widget display
- better ELT Pupil generation
- add unit tests of rtc module
- add DM petal
- add fake camera input
- debug ```load_config_from_file```
- debug ```wfs_init```

for Internal developments:

- add compile script

## Release v4.0.0

- change centroid computation using CUB library

for Internal developments:

- rename internal variables

## Release v3.4

- Naga anf Shesha are pur-python module
- CarmaWrap and SheshaWrap are pur-wrapper module using pybind
- minor debug

for Internal developments:

- rename internal variables

## Release v3.3

minor changes

## Release v3.2

- Re-up the database feature that allows to skip initialisation phase by re-using results of previous similar simulation runs

## Release v3.0

- Binding based on pyBind11
- N-faces pyramid WFS
- Introduce masked pixels centroider
- Introduce GUARDIANS package
- Introduce a way to check installation
- Debug pupil alignment (for non circular pupil)
- Shesha supervisor module improvement
- Remove the use of bytes (string instead)

## Release v2.0

- code restructuration
- Debug SH spots and PSF moves
- Debug Widget
- Fix build failure with cython 0.28
- Other minor debug

## Release v1.1

- update parameter files
- add pyr_misalignments
- add rtc_standalone
- add dm_standalone
- add supervisor between simulation and widget
