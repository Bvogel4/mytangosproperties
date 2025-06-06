# Standard library imports
import gc
import logging
import multiprocessing as mp
import sys
import traceback
import warnings

# Third-party imports
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pynbody
from pynbody import array, units
from pynbody.plot.sph import image
import pymp
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from skimage.measure import moments, moments_central
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model

# Local application imports
from tangos.properties import LivePropertyCalculation, LivePropertyCalculationInheritingMetaProperties
from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties.pynbody.centring import centred_calculation

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsophoteAnalysis")


def myprint(text, clear=False):
    """Custom print function for progress display"""
    import sys
    if clear:
        sys.stdout.write("\r" + " " * 100 + "\r")  # Clear line
    sys.stdout.write(f"\r{text}")
    sys.stdout.flush()


# logger = logging.getLogger('pynbody.analysis.halo')
logger = logging.getLogger('tangos')

#modifed from pynbody.analysis.halo to add parallelization within halos
def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=100, tol=1e-3, justify=False):
    """Calculates the shape of the provided particles in homeoidal shells, over a range of nbins radii.

    Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3). Note that all provided particles are used in
    calculating the shape, so e.g. to measure dark matter halo shape from a halo with baryons, you should pass
    only the dark matter particles.

    The simulation must be pre-centred, e.g. using :func:`center`.

    The algorithm is sensitive to substructure, which should ideally be removed.

    Caution is advised when assigning large number of bins and radial ranges with many particles, as the
    algorithm becomes very slow.

    Parameters
    ----------

      nbins : int
          The number of homeoidal shells to consider. Shells with few particles will take longer to fit.

      rmin : float
          The minimum radial bin in units of sim['pos']. By default this is taken as rout/1000.
          Note that this applies to axis a, so particles within this radius may still be included within
          homeoidal shells.

      rmax : float
          The maximum radial bin in units of sim['pos']. By default this is taken as the largest radial value
          in the halo particle distribution.

      bins : str
          The spacing scheme for the homeoidal shell bins. 'equal' initialises radial bins with equal numbers
          of particles, with the exception of the final bin which will accomodate remainders. This
          number is not necessarily maintained during fitting. 'log' and 'lin' initialise bins
          with logarithmic and linear radial spacing.

      ndim : int
          The number of dimensions to consider; either 2 or 3 (default). If ndim=2, the shape is calculated
          in the x-y plane. If using ndim=2, you may wish to make a cut in the z direction before
          passing the particles to this routine (e.g. using :class:`pynbody.filt.BandPass`).

      max_iterations : int
          The maximum number of shape calculations (default 10). Fewer iterations will result in a speed-up,
          but with a bias towards spheroidal results.

      tol : float
          Convergence criterion for the shape calculation. Convergence is achieved when the axial ratios have
          a fractional change <=tol between iterations.

      justify : bool
          Align the rotation matrix directions such that they point in a single consistent direction
          aligned with the overall halo shape. This can be useful if working with slerps.

    Returns
    -------

      rbin : SimArray
          The radial bins used for the fitting

      axis_lengths : SimArray
          A nbins x ndim array containing the axis lengths of the ellipsoids in each shell

      num_particles : np.ndarray
          The number of particles within each bin

      rotation_matrices : np.ndarray
          The rotation matrices for each shell

    """

    # Sanitise inputs:
    if (rmax == None): rmax = sim['r'].max()
    if (rmin == None): rmin = rmax / 1E3
    assert ndim in [2, 3]
    assert max_iterations > 0
    assert tol > 0
    assert rmin >= 0
    assert rmax > rmin
    assert nbins > 0
    if ndim == 2:
        assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins * 2
    elif ndim == 3:
        assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins * 2
    if bins not in ['equal', 'log', 'lin']: bins = 'equal'

    # Handy 90 degree rotation matrices:
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # -----------------------------FUNCTIONS-----------------------------
    sn = lambda r, N: np.append([r[i * int(len(r) / N):(1 + i) * int(len(r) / N)][0] \
                                 for i in range(N)], r[-1])

    # General equation for an ellipse/ellipsoid:
    def Ellipsoid(pos, a, R):
        x = np.dot(R.T, pos.T)
        return np.sum(np.divide(x.T, a) ** 2, axis=1)

    # Define moment of inertia tensor:
    def MoI(r, m, ndim=3):
        return np.array([[np.sum(m * r[:, i] * r[:, j]) for j in range(ndim)] for i in range(ndim)])



    # Calculate the shape in a single shell:
    def shell_shape(r, pos, mass, a, R, r_range, ndim=3):

        # Find contents of homoeoidal shell:
        mult = r_range / np.mean(a)
        in_shell = (r > min(a) * mult[0]) & (r < max(a) * mult[1])
        pos, mass = pos[in_shell], mass[in_shell]
        inner = Ellipsoid(pos, a * mult[0], R)
        outer = Ellipsoid(pos, a * mult[1], R)
        in_ellipse = (inner > 1) & (outer < 1)
        ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

        # End if there is no data in range:
        if not len(ellipse_mass):
            return a, R, np.sum(in_ellipse)

        # Calculate shape tensor & diagonalise:
        D = list(np.linalg.eigh(MoI(ellipse_pos, ellipse_mass, ndim) / np.sum(ellipse_mass)))

        # Rescale axis ratios to maintain constant ellipsoidal volume:
        R2 = np.array(D[1])
        a2 = np.sqrt(abs(D[0]) * ndim)
        div = (np.prod(a) / np.prod(a2)) ** (1 / float(ndim))
        a2 *= div

        return a2, R2, np.sum(in_ellipse)

    # Re-align rotation matrix:
    def realign(R, a, ndim):
        if ndim == 3:
            if a[0] > a[1] > a[2] < a[0]:
                pass  # abc
            elif a[0] > a[1] < a[2] < a[0]:
                R = np.dot(R, Rx)  # acb
            elif a[0] < a[1] > a[2] < a[0]:
                R = np.dot(R, Rz)  # bac
            elif a[0] < a[1] > a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Ry)  # bca
            elif a[0] > a[1] < a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Rz)  # cab
            elif a[0] < a[1] < a[2] > a[0]:
                R = np.dot(R, Ry)  # cba
        elif ndim == 2:
            if a[0] > a[1]:
                pass  # ab
            elif a[0] < a[1]:
                R = np.dot(R, Rz[:2, :2])  # ba
        return R

    # Calculate the angle between two vectors:
    def angle(a, b):
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Flip x,y,z axes of R2 if they provide a better alignment with R1.
    def flip_axes(R1, R2):
        for i in range(len(R1)):
            if angle(R1[:, i], -R2[:, i]) < angle(R1[:, i], R2[:, i]):
                R2[:, i] *= -1
        return R2

    def process_bin(i, r, pos, mass, rbins, bin_edges, ndim):
        a = np.ones(ndim) * rbins[i]
        R = np.identity(ndim)

        for j in range(max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim)

            convergence_criterion = np.all(np.isclose(np.sort(a), np.sort(a2), rtol=tol))
            if convergence_criterion:
                R = realign(R, a, ndim)
                if np.sign(np.linalg.det(R)) == -1:
                    R[:, 1] *= -1
                a = np.flip(np.sort(a))
                return i, a, R, N
        return i, np.ones(ndim) * np.nan, np.identity(ndim) * np.nan, 0


    # -----------------------------FUNCTIONS-----------------------------

    # Set up binning:
    r = np.array(sim['r']) if ndim == 3 else np.array(sim['rxy'])
    pos = np.array(sim['pos'])[:, :ndim]
    mass = np.array(sim['mass'])

    if (bins == 'equal'):  # Bins contain equal number of particles
        full_bins = sn(np.sort(r[(r >= rmin) & (r <= rmax)]), nbins * 2)
        bin_edges = full_bins[0:nbins * 2 + 1:2]
        rbins = full_bins[1:nbins * 2 + 1:2]
    elif (bins == 'log'):  # Bins are logarithmically spaced
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif (bins == 'lin'):  # Bins are linearly spaced
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialise the shape arrays:
    rbins = array.SimArray(rbins, sim['pos'].units)
    axis_lengths = array.SimArray(np.zeros([nbins, ndim]), sim['pos'].units) * np.nan
    N_in_bin = np.zeros(nbins).astype('int')
    V_in_bin = np.zeros(nbins)
    #create an array with n R = np.identity(ndim)
    rotations = np.array([np.identity(ndim)] * nbins) * np.nan


    # Calculate the shape in each bin:
    threads = 40
    #create shared objects axis_lengths, N_in_bin, rotations
    shared_results = pymp.shared.dict()


    with pymp.Parallel(threads) as p:
        for i in p.range(nbins):
            #p.print(f'Processing bin {i}')
            i, a, R, N = process_bin(i, r, pos, mass, rbins, bin_edges, ndim)
            #store in results
            with p.lock:
                shared_results[i] = (a, R, N)
    #unpack results
    results = dict(shared_results)
    for i in range(nbins):
        a, R, N = results[i]
        axis_lengths[i] = a
        N_in_bin[i] = N
        rotations[i] = R

    # for each bin, calculate the shape sensitivity
    # eq 7 from https://www.aanda.org/articles/aa/abs/2023/02/aa45031-22/aa45031-22.html
    # sensitivity = V_i / sqrt(N_i) * abs(N_(i-1)/V_(i-1) - N_(i+1)/V_(i+1))
    # where V is the volume of the shell and N is the number of particles in the shell




    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])
    #print(rotations[0])
    axis_lengths = np.squeeze(axis_lengths.T).T
    rotations = np.squeeze(rotations)

    return rbins, axis_lengths, N_in_bin, rotations

def get_bins(n):
    if n < 1e4:
        return int(20 * n/1e4 /2 + 10)
    elif n >= 1e4:
        return int( (np.log10(n) - 3) ** 2 * 20)


class ImageHalo(PynbodyPropertyCalculation):
    """Generates V-band luminosity density images at different orientations and calculates
    effective radii for each projection to study galaxy morphology"""

    # Store the key properties we need for shape analysis
    names = ['halo_images', 'image_reffs', 'image_orientations', 'Rhalf',
             'profile_sb_v', 'profile_v_lum_den', 'profile_rbins',
             'profile_lum_den', 'profile_mags_v', 'profile_binarea']

    @staticmethod
    def fit_sersic_profile(prof):
        """Fits a Sérsic profile to determine the effective radius (Reff) for each projection.

        The Sérsic profile describes how galaxy brightness varies with radius:
        μ(r) = μeff + 2.5(0.868n - 0.142)((r/reff)^(1/n) - 1)
        where μeff is surface brightness at effective radius, n is Sérsic index
        """

        def sersic(r, mueff, reff, n):
            return mueff + 2.5 * (0.868 * n - 0.142) * (
                    (r / reff) ** (1. / n) - 1)

        # Smooth the V-band surface brightness profile to reduce noise
        vband = prof['sb,V']
        smooth = np.nanmean(
            np.pad(vband.astype(float), (0, 3 - vband.size % 3),
                   mode='constant', constant_values=np.nan).reshape(-1, 3),
            axis=1)

        # Set up radial coordinates for fitting
        x = np.arange(
            len(smooth)) * 0.3 + 0.15  # Convert to physical units (kpc)
        x[0] = .05  # Avoid r=0 singularity

        # Remove any NaN values before fitting
        y = smooth[~np.isnan(smooth)]
        x = x[~np.isnan(smooth)]

        # Initial guesses for fit parameters
        r0 = x[int(len(x) / 2)]  # Initial Reff guess is middle of radial range
        m0 = np.mean(
            y[:3])  # Initial surface brightness guess from central region

        # Fit Sérsic profile with reasonable bounds for galaxy parameters
        par, _ = curve_fit(sersic, x, y, p0=(m0, r0, 1),
                           bounds=([10, 0, 0.5], [40, 100, 16.5]))
        return par[1]  # Return fitted Reff

    @staticmethod
    def generate_image(stars, width):
        f = plt.figure(frameon=False)
        f.set_size_inches(10, 10)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)
        im = pynbody.plot.sph.image(stars, qty='V_lum_den', width=width, subplot=ax, units='kpc^-2', resolution=1000,
                                    show_cbar=False, ret_im=True)
        data = im.get_array()  # Get the numpy array
        plt.close(f)
        return data


    def process_orientation(self,halo, width, Rhalf):
        orientation_data = {'Rhalf': Rhalf.view(np.ndarray)}
        prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                                nbins=int((5 * Rhalf) / 0.1))

        orientation_data.update({
            'sb,v': prof['sb,V'].copy().view(np.ndarray),
            'v_lum_den': prof['V_lum_den'].copy().view(np.ndarray),
            'rbins': prof['rbins'].copy().view(np.ndarray),
            'lum_den': (10.0 ** (-0.4 * prof['magnitudes,V']) / prof._binsize.in_units('pc^2')).copy().view(np.ndarray),
            'mags,v': prof['magnitudes,V'].copy().view(np.ndarray),
            'binarea': prof._binsize.in_units('pc^2').copy().view(np.ndarray)
        })
        orientation_data['Reff'] = self.fit_sersic_profile(prof)

        orientation_data['image'] = self.generate_image(halo.s, width)


        return orientation_data

    def process_halo(self, halo, existing_properties):
        """Generate images and measure Reff across different viewing angles.

        We sample viewing angles by rotating the galaxy through θ and φ to create
        a set of 2D projections that mimic observational studies.
        """
        dx, dy = 30, 30  # Rotation increments (default: 30,30 for finer sampling)
        halo.physical_units()  # Ensure physical units are used

        # Orient galaxy face-on initially using gas angular momentum
        pynbody.analysis.angmom.faceon(halo)
        # make sure halo has stars
        Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        width = 9 * Rhalf  # Image width captures extended structure

        # Select stars within a sphere that contains full projection at any angle
        ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)

        xrotations = np.arange(0, 180, dx)
        yrotations = np.arange(0, 360, dy)

        # Create a list of all orientation combinations
        all_orientations = [(x, y) for x in xrotations for y in yrotations]

        # Shared dictionary to store results
        shared_dict = pymp.shared.dict()

        # Process all orientations in parallel
        with pymp.Parallel(1) as p:  # Adjust number of processes as needed
            for i in p.range(len(all_orientations)):
                xrotation, yrotation = all_orientations[i]
                # Apply rotations
                with halo.rotate_x(xrotation).rotate_y(yrotation):
                    key = f'x{xrotation:03d}y{yrotation:03d}'
                    sb_dict = self.process_orientation(halo.s[ImageSpace], width, Rhalf)

                # Store result with lock to avoid conflicts
                with p.lock:
                    shared_dict[key] = sb_dict

        shared_dict = dict(shared_dict)
        # sort the dictionary by key
        shared_dict = dict(sorted(shared_dict.items()))

        # Unpack results from shared dictionary
        orientations = list(shared_dict.keys())
        #initialize lists
        images = []
        reff_values = []
        profile_sb_v = []
        profile_v_lum_den = []
        profile_rbins = []
        profile_lum_den = []
        profile_mags_v = []
        profile_binarea = []
        for key in orientations:
            images.append(shared_dict[key]['image'])
            reff_values.append(shared_dict[key]['Reff'])
            profile_sb_v.append(shared_dict[key]['sb,v'])
            profile_v_lum_den.append(shared_dict[key]['v_lum_den'])
            profile_rbins.append(shared_dict[key]['rbins'])
            profile_lum_den.append(shared_dict[key]['lum_den'])
            profile_mags_v.append(shared_dict[key]['mags,v'])
            profile_binarea.append(shared_dict[key]['binarea'])


        return images, reff_values, orientations, Rhalf, profile_sb_v, profile_v_lum_den, profile_rbins, profile_lum_den, profile_mags_v, profile_binarea
    def calculate(self,halo,existing_properties):
        return self.process_halo(halo, existing_properties)





class IsophoteAnalysis(LivePropertyCalculation):
    """Analyzes isophotes to measure projected galaxy shapes at different radii.

    For each projection, we measure ellipticity and position angle at 2-4 Reff
    to track how galaxy shape varies with radius.
    """
    names = 'isophote_parameters'

    def requires_property(self):
        return ['halo_images', 'image_reffs', 'image_orientations', 'Rhalf']

    def __init__(self, simulation):
        super().__init__(simulation)
        self.visualization_enabled = True  # Set to True to enable visualizations

    def estimate_initial_params(self, image, reff, kpc_per_pixel, plot=False):
        """Estimate ellipse parameters using image moments with visualization"""
        #logger.info("Estimating initial parameters from image moments")

        # Threshold image to separate galaxy from background
        threshold = np.median(image)

        # get binary image of values near the threshold
        binary_image = np.where((image > 5*threshold) & (image <20*threshold), 1, 0)

        # Show the original image
        if plot:
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(np.log10(image), origin='lower', cmap='viridis', vmin=-3)
            plt.colorbar(label='Intensity')
            plt.title('Original Image')

            plt.subplot(2, 2, 2)
            plt.imshow(binary_image, origin='lower', cmap='gray')
            plt.title(f'Thresholded Image (threshold={threshold:.2f})')


        # Calculate moments
        m = moments(binary_image)
        #logger.info(f"Zero-order moment (total mass): {m[0, 0]}")

        if m[0, 0] == 0:  # No pixels above threshold
            logger.warning("No pixels above threshold, using image center")
            return image.shape[0] / 2, image.shape[1] / 2, 0.1, 0, 3.0

        # Calculate centroid
        #centroid_y, centroid_x = m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]
        #set centroid to be center of image
        centroid_y, centroid_x = image.shape[0] / 2, image.shape[1] / 2
        #logger.info(f"Centroid: x={centroid_x:.2f}, y={centroid_y:.2f}")

        # Central moments for shape estimation
        mu = moments_central(binary_image, (centroid_y, centroid_x))

        # Covariance matrix
        cov = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]]) / mu[0, 0]
        #logger.info(f"Covariance matrix:\n{cov}")

        # Eigenvalues give us axis lengths
        evals, evecs = np.linalg.eig(cov)
        #logger.info(f"Eigenvalues: {evals}")
        #logger.info(f"Eigenvectors:\n{evecs}")

        # Check for negative eigenvalues (numerical instability)
        if np.any(evals <= 0):
            logger.warning("Negative eigenvalues detected, using default values")
            return centroid_x, centroid_y, 0.1, 0, 3.0

        a = np.sqrt(evals[0])  # Major axis
        b = np.sqrt(evals[1])  # Minor axis

        # compute angle
        #logger.info(f"Vector for angle calculation: y={evecs[0, 0]}, x={evecs[0, 1]}")
        angle = np.arctan2(evecs[0, 0], evecs[0, 1])
        # check that a > b
        if a < b:
            #logger.warning("Major axis is smaller than minor axis, swapping values")
            a, b = b, a
            # rotate angle by 90 degrees
            angle += np.pi / 2

        #logger.info(f"Major axis: {a:.2f}, Minor axis: {b:.2f}")

        # Ellipticity and position angle
        eps = 1.0 - (b / a)
        eps = min(0.85, max(0.01, eps))  # Constrain ellipticity
        #logger.info(f"Initial ellipticity: {eps:.2f}")
        #logger.info(f"Initial position angle: {angle:.2f} radians")

        # Estimate a good starting radius from moments
        estimated_radius = np.sqrt(np.mean([mu[2, 0], mu[0, 2]]) / mu[0, 0])
        # Convert to a factor of the effective radius
        radius_factor = estimated_radius / (reff / kpc_per_pixel)
        # Constrain to reasonable values
        radius_factor = min(4.5, max(2.5, radius_factor))
        #logger.info(f"Estimated radius: {estimated_radius:.2f} pixels")
        #logger.info(f"Radius factor (x Reff): {radius_factor:.2f}")

        # Visualize the estimated ellipse
        if plot:
            plt.subplot(2, 2, 3)
            plt.imshow(np.log10(image), origin='lower', cmap='viridis', vmin=-3)

            # Convert position angle to degrees for ellipse plotting
            angle_deg = np.degrees(angle)

            # Draw ellipse
            from matplotlib.patches import Ellipse as MplEllipse
            ell = MplEllipse((centroid_x, centroid_y),
                             width=2 * a, height=2 * b,
                             angle=angle_deg,
                             edgecolor='red', facecolor='none', linewidth=2)
            plt.gca().add_patch(ell)

            # Draw axes
            plt.plot([centroid_x, centroid_x + a * np.cos(angle)],
                     [centroid_y, centroid_y + a * np.sin(angle)],
                     'r-', linewidth=1)
            plt.plot([centroid_x, centroid_x + b * np.cos(angle + np.pi / 2)],
                     [centroid_y, centroid_y + b * np.sin(angle + np.pi / 2)],
                     'r-', linewidth=1)

            plt.title(f'Estimated Ellipse (e={eps:.2f}, PA={angle_deg:.2f}°)')
            plt.tight_layout()
            plt.show()

        return centroid_x, centroid_y, eps, angle, radius_factor

    def fit_single_image(self, image_data, radius, step_size_factors, center, eps, pa, sma_factor,
                         kpc_per_pixel, plot=False):
        """Fit ellipse with moment-based parameters and return results with visualization"""
        #logger.info("Starting isophote fitting")
        #logger.info(f"Effective radius: {radius:.2f} kpc")
        #logger.info(f"Scale: {kpc_per_pixel:.4f} kpc/pixel")
        #logger.info(f"Initial parameters: center={center}, eps={eps:.2f}, pa={pa:.2f}, sma_factor={sma_factor:.2f}")

        radius_pixels = radius / kpc_per_pixel
        #logger.info(f"Effective radius: {radius_pixels:.2f} pixels")

        # Track good isophotes
        good_isophotes = {}  # Will store successful fits indexed by sma in pixels

        # Track which target radii (2, 3, 4 * Reff) have been successfully measured
        target_multipliers = [2.0, 3.0, 4.0]
        target_radii = {mult: mult * radius for mult in target_multipliers}
        target_radii_met = {mult: False for mult in target_multipliers}

        #logger.info(f"Target radii: {target_radii}")

        # Initial ellipse geometry with original guess
        geometry = EllipseGeometry(
            x0=center[0], y0=center[1],
            sma=sma_factor * radius_pixels,
            eps=eps,
            pa=pa
        )
        #logger.info(f"Initial geometry: {geometry}")

        # Start with the full range
        min_sma = 0.75 * radius_pixels
        max_sma = 4.5 * radius_pixels
        #logger.info(f"SMA range: {min_sma:.2f} to {max_sma:.2f} pixels")

        # Base step size
        base_step = 1/8

        all_fitting_results = []  # Store results from each step size for visualization

        # Try progressively smaller step sizes until we get good fits at all target radii
        for step_idx, step_factor in enumerate(step_size_factors):
            step = base_step * radius_pixels * step_factor
            #logger.info(f"\n--- Using step size factor {step_factor} (step={step:.2f} pixels) ---")

            # Check if we've already met all targets
            if all(target_radii_met.values()):
                #logger.info("All target radii have been met, breaking loop")
                break

            # Calculate which ranges we still need to fit
            # Convert good_isophotes keys to kpc for comparison with target_radii
            good_smas_kpc = np.array([sma * kpc_per_pixel for sma in good_isophotes.keys()])

            # Determine remaining ranges to fit
            ranges_to_fit = []

            if not good_smas_kpc.size:
                # No good fits yet, fit the full range
                ranges_to_fit.append((min_sma, max_sma))
                #logger.info("No good fits yet, fitting full range")
            else:
                # Check which target radii haven't been met
                for mult in target_multipliers:
                    if not target_radii_met[mult]:
                        target = target_radii[mult]
                        # Find closest good fits below and above the target
                        below_idx = np.where(good_smas_kpc < target)[0]
                        above_idx = np.where(good_smas_kpc > target)[0]

                        below_sma = good_smas_kpc[below_idx[-1]] / kpc_per_pixel if below_idx.size else min_sma
                        above_sma = good_smas_kpc[above_idx[0]] / kpc_per_pixel if above_idx.size else max_sma

                        # Only add range if there's a significant gap
                        gap_threshold = 0.2 * radius_pixels
                        if above_sma - below_sma > gap_threshold:
                            ranges_to_fit.append((below_sma, above_sma))
                            #logger.info(
                                #f"Adding range: {below_sma:.2f} to {above_sma:.2f} pixels for target {mult}×Reff")

            # If no more ranges to fit, break
            if not ranges_to_fit:
                #logger.info("No more ranges to fit, breaking loop")
                break

            # Fit each remaining range
            for range_idx, (range_min, range_max) in enumerate(ranges_to_fit):
                #logger.info(
                    #f"Fitting range {range_idx + 1}/{len(ranges_to_fit)}: {range_min:.2f} to {range_max:.2f} pixels")

                # Get the closest good isophote to use as a starting point
                if good_isophotes:
                    # Find closest good isophote to the midpoint of our range
                    midpoint = (range_min + range_max) / 2
                    closest_sma = min(good_isophotes.keys(), key=lambda x: abs(x - midpoint))
                    closest_iso = good_isophotes[closest_sma]

                    #logger.info(f"Using closest good isophote at SMA={closest_sma:.2f} pixels as starting point")

                    # Update geometry with values from the closest good isophote
                    geometry = EllipseGeometry(
                        x0=closest_iso['x0'],
                        y0=closest_iso['y0'],
                        sma=closest_sma,
                        eps=closest_iso['eps'],
                        pa=closest_iso['pa']
                    )
                    #logger.info(f"Updated geometry: {geometry}")

                # Fit ellipses for this range
                #logger.info(f"Fitting with SMA range: {range_min:.2f} to {range_max:.2f} pixels")

                try:
                    ellipse = Ellipse(image_data, geometry)
                    isolist = ellipse.fit_image(
                        minsma=range_min,
                        maxsma=range_max,
                        sma0=geometry.sma,
                        linear=True,
                        step=step,
                        maxit=100,
                        minit=20,
                        fix_center=False,
                        fix_eps=False,
                        fix_pa=False,
                        sclip=2.,
                        nclip=3
                    )

                    # Store this fitting result for visualization
                    all_fitting_results.append({
                        'step_factor': step_factor,
                        'range': (range_min, range_max),
                        'isolist': isolist
                    })

                    # Check if we got any valid isophotes
                    if len(isolist.sma) == 0:
                        #logger.info("No valid isophotes found in this range")
                        #if it's the first step, let's retry our initial guess with a larger radius
                        if step_factor == step_size_factors[0]:
                            sma = geometry.sma
                            for r_factor in [1.5,0.75,2.0,0.5,3.0,0.25]:
                                try:
                                    #logger.info("Trying initial guess with radius factor: {}".format(r_factor))
                                    ellipse = Ellipse(image_data, geometry)
                                    range_min = 2.5 * radius_pixels
                                    range_max = 4.5 * radius_pixels
                                    #ensure the range includes sma*r_factor
                                    range_min = min(range_min, sma * r_factor)
                                    range_max = max(range_max, sma * r_factor)

                                    isolist = ellipse.fit_image(
                                        minsma=range_min,
                                        maxsma=range_max,
                                        sma0=sma * r_factor,
                                        linear=True,
                                        step=step,
                                        maxit=100,
                                        minit=20,
                                        fix_center=False,
                                        fix_eps=False,
                                        fix_pa=False,
                                    )
                                except Exception as e:
                                    #logger.error(f"Error during fitting with radius factor {r_factor}: {e}")
                                    continue
                                if len(isolist.sma) > 0:
                                    #logger.info(f'Found valid isophotes with radius factor: {r_factor}')
                                    break

                            #logger.info(f"Found {len(isolist.sma)} isophotes with larger radius")

                    #logger.info(f"Found {len(isolist.sma)} isophotes")

                    # Extract data and add good isophotes to our collection
                    smas_kpc = isolist.sma * kpc_per_pixel

                    for i in range(len(isolist.sma)):
                        sma_pixels = isolist.sma[i]
                        sma_kpc = smas_kpc[i]

                        # Only consider good fits (gradient error below threshold)
                        if isolist.grad_r_error[i] < 0.1:
                            # Store the good isophote if it's better than what we have or if we don't have it
                            if sma_pixels not in good_isophotes or isolist.grad_r_error[i] < good_isophotes[sma_pixels][
                                'grad_error']:
                                good_isophotes[sma_pixels] = {
                                    'sma': sma_pixels,
                                    'sma_kpc': sma_kpc,
                                    'eps': isolist.eps[i],
                                    'pa': isolist.pa[i],
                                    'x0': isolist.x0[i],
                                    'y0': isolist.y0[i],
                                    'grad_error': isolist.grad_r_error[i],
                                    'intens': isolist.intens[i],
                                    'rms': isolist.rms[i]
                                }

                                #logger.info(f"Found good isophote at SMA={sma_pixels:.2f} pixels ({sma_kpc:.2f} kpc)")
                                #logger.info(
                                    #f"  e={isolist.eps[i]:.2f}, PA={isolist.pa[i]:.2f}, grad_err={isolist.grad_r_error[i]:.4f}")

                            # Check if this isophote satisfies any of our target radii
                            for mult in target_multipliers:
                                target = target_radii[mult]
                                # Allow a small tolerance (5%)
                                if abs(sma_kpc - target) < 0.05 * target:
                                    #if not target_radii_met[mult]:
                                        #logger.info(
                                            #f"Target {mult}×Reff ({target:.2f} kpc) met with isophote at {sma_kpc:.2f} kpc")
                                    target_radii_met[mult] = True

                except Exception as e:
                    logger.error(f"Error during fitting: {e}")
                    traceback.print_exc()

        # Log which targets were met
        # for mult in target_multipliers:
        #     status = "✓" if target_radii_met[mult] else "✗"
        #     logger.info(f"Target {mult}×Reff: {status}")

        if not all(target_radii_met.values()):
            # logger.info("\n--- Relaxing gradient error threshold to fill missing target radii ---")

            # Starting from slightly above our initial threshold, gradually increase up to a maximum
            max_threshold = 0.5  # Maximum threshold we'll accept
            threshold_steps = [0.15, 0.2, 0.3, 0.4, 0.5]  # Gradually increasing thresholds

            for relaxed_threshold in threshold_steps:
                # logger.info(f"Trying relaxed threshold: {relaxed_threshold:.2f}")

                # Go through all our previous fitting results with the relaxed threshold
                for result in all_fitting_results:
                    isolist = result['isolist']

                    # Skip if no isophotes were found in this result
                    if len(isolist.sma) == 0:
                        continue

                    smas_kpc = isolist.sma * kpc_per_pixel

                    for i in range(len(isolist.sma)):
                        sma_pixels = isolist.sma[i]
                        sma_kpc = smas_kpc[i]

                        # Apply relaxed threshold
                        if isolist.grad_r_error[i] < relaxed_threshold:
                            # Only add if this is better than what we have or we don't have it
                            if (sma_pixels not in good_isophotes or
                                    isolist.grad_r_error[i] < good_isophotes[sma_pixels]['grad_error']):
                                good_isophotes[sma_pixels] = {
                                    'sma': sma_pixels,
                                    'sma_kpc': sma_kpc,
                                    'eps': isolist.eps[i],
                                    'pa': isolist.pa[i],
                                    'x0': isolist.x0[i],
                                    'y0': isolist.y0[i],
                                    'grad_error': isolist.grad_r_error[i],
                                    'intens': isolist.intens[i],
                                    'rms': isolist.rms[i]
                                }

                                # logger.info(f"With relaxed threshold, found isophote at SMA={sma_pixels:.2f} pixels ({sma_kpc:.2f} kpc)")
                                # logger.info(f"  e={isolist.eps[i]:.2f}, PA={isolist.pa[i]:.2f}, grad_err={isolist.grad_r_error[i]:.4f}")

                            # Check if this isophote satisfies any target radii
                            for mult in target_multipliers:
                                if not target_radii_met[mult]:
                                    target = target_radii[mult]
                                    # Allow a small tolerance (5%)
                                    if abs(sma_kpc - target) < 0.05 * target:
                                        target_radii_met[mult] = True
                                        # logger.info(f"Target {mult}×Reff ({target:.2f} kpc) met with relaxed threshold")

                # Check if we've met all targets now
                if all(target_radii_met.values()):
                    # logger.info(f"All target radii met with relaxed threshold {relaxed_threshold:.2f}")
                    break

        # Convert our dictionary of good isophotes to the expected format for return
        result = []
        for sma in sorted(good_isophotes.keys()):
            iso = good_isophotes[sma]
            result.append([
                iso['sma'],
                iso['eps'],
                iso['pa'],
                iso['grad_error'],
                iso['x0'],
                iso['y0'],
                iso['intens'],
                iso['rms']
            ])

        # Visualize the final results if requested
        if plot:
            self.visualize_results(image_data, good_isophotes, target_radii,
                                   all_fitting_results, kpc_per_pixel)

        return result, all(target_radii_met.values())

    def visualize_results(self, image, good_isophotes, target_radii, all_fitting_results, kpc_per_pixel):
        """Visualize the isophote fitting results"""
        # Create a figure with multiple subplots
        plt.figure(figsize=(18, 12))

        # Plot the image with all good isophotes
        plt.subplot(2, 3, 1)
        plt.imshow(np.log10(image), origin='lower', cmap='viridis', vmin=1)

        # Draw all good isophotes
        from matplotlib.patches import Ellipse as MplEllipse

        # Different colors for different isophotes
        colors = plt.cm.tab10(np.linspace(0, 1, len(good_isophotes)))

        # Track which isophotes match our target radii
        target_isophotes = {}
        for mult, target_kpc in target_radii.items():
            closest_sma = None
            min_diff = float('inf')

            for sma_pix, iso in good_isophotes.items():
                sma_kpc = iso['sma_kpc']
                diff = abs(sma_kpc - target_kpc)

                if diff < min_diff and diff < 0.05 * target_kpc:
                    min_diff = diff
                    closest_sma = sma_pix

            if closest_sma is not None:
                target_isophotes[mult] = closest_sma

        # Draw ellipses for all good isophotes
        for i, (sma_pix, iso) in enumerate(good_isophotes.items()):
            # Special color for target isophotes
            is_target = False
            for mult, target_sma in target_isophotes.items():
                if sma_pix == target_sma:
                    is_target = True
                    color = 'red' if mult == 2.0 else 'green' if mult == 3.0 else 'blue'
                    label = f"{mult}×Reff"
                    linewidth = 2
                    break

            if not is_target:
                color = colors[i % len(colors)]
                label = None
                linewidth = 1

            # Convert position angle to degrees for ellipse plotting
            angle_deg = np.degrees(iso['pa'])

            # Semi-minor axis
            b = sma_pix * (1 - iso['eps'])

            # Draw ellipse
            ell = MplEllipse((iso['x0'], iso['y0']),
                             width=2 * sma_pix, height=2 * b,
                             angle=angle_deg,
                             edgecolor=color, facecolor='none', linewidth=linewidth,
                             label=label if label else None)
            plt.gca().add_patch(ell)

        # Add a legend for target isophotes
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, loc='upper right')

        plt.title('All Good Isophotes')
        plt.colorbar(label='Intensity')

        # Plot ellipticity vs. semi-major axis
        plt.subplot(2, 3, 2)
        smas = np.array([iso['sma_kpc'] for iso in good_isophotes.values()])
        eps = np.array([iso['eps'] for iso in good_isophotes.values()])

        # Sort by SMA
        sort_idx = np.argsort(smas)
        smas = smas[sort_idx]
        eps = eps[sort_idx]

        plt.plot(smas, eps, 'o-')

        # Mark target radii
        for mult, target_kpc in target_radii.items():
            if mult in target_isophotes:
                target_idx = np.where(smas == good_isophotes[target_isophotes[mult]]['sma_kpc'])[0]
                if len(target_idx) > 0:
                    color = 'red' if mult == 2.0 else 'green' if mult == 3.0 else 'blue'
                    plt.plot(smas[target_idx], eps[target_idx], 'o', color=color,
                             markersize=10, label=f"{mult}×Reff")

        plt.xlabel('Semi-major axis (kpc)')
        plt.ylabel('Ellipticity')
        plt.title('Ellipticity vs. Semi-major axis')
        plt.legend()
        plt.grid(True)

        # Plot position angle vs. semi-major axis
        plt.subplot(2, 3, 3)
        pas = np.array([iso['pa'] for iso in good_isophotes.values()])[sort_idx]

        # Convert to degrees
        pas_deg = np.degrees(pas) % 180

        plt.plot(smas, pas_deg, 'o-')

        # Mark target radii
        for mult, target_kpc in target_radii.items():
            if mult in target_isophotes:
                target_idx = np.where(smas == good_isophotes[target_isophotes[mult]]['sma_kpc'])[0]
                if len(target_idx) > 0:
                    color = 'red' if mult == 2.0 else 'green' if mult == 3.0 else 'blue'
                    plt.plot(smas[target_idx], pas_deg[target_idx], 'o', color=color,
                             markersize=10, label=f"{mult}×Reff")

        plt.xlabel('Semi-major axis (kpc)')
        plt.ylabel('Position Angle (degrees)')
        plt.title('Position Angle vs. Semi-major axis')
        plt.legend()
        plt.grid(True)

        # Plot gradient error vs. semi-major axis
        plt.subplot(2, 3, 4)
        errors = np.array([iso['grad_error'] for iso in good_isophotes.values()])[sort_idx]

        plt.plot(smas, errors, 'o-')
        plt.axhline(y=0.1, color='r', linestyle='--', label='Error threshold')

        # Mark target radii
        for mult, target_kpc in target_radii.items():
            if mult in target_isophotes:
                target_idx = np.where(smas == good_isophotes[target_isophotes[mult]]['sma_kpc'])[0]
                if len(target_idx) > 0:
                    color = 'red' if mult == 2.0 else 'green' if mult == 3.0 else 'blue'
                    plt.plot(smas[target_idx], errors[target_idx], 'o', color=color,
                             markersize=10, label=f"{mult}×Reff")

        plt.xlabel('Semi-major axis (kpc)')
        plt.ylabel('Gradient Error')
        plt.title('Gradient Error vs. Semi-major axis')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        # Plot center x and y vs. semi-major axis
        plt.subplot(2, 3, 5)
        x0s = np.array([iso['x0'] for iso in good_isophotes.values()])[sort_idx]
        y0s = np.array([iso['y0'] for iso in good_isophotes.values()])[sort_idx]

        plt.plot(smas, x0s, 'o-', label='x0')
        plt.plot(smas, y0s, 'o-', label='y0')

        plt.xlabel('Semi-major axis (kpc)')
        plt.ylabel('Center position (pixels)')
        plt.title('Ellipse Center vs. Semi-major axis')
        plt.legend()
        plt.grid(True)

        # Plot progress of fitting
        plt.subplot(2, 3, 6)

        # For each fitting step, plot the resulting isophotes
        markers = ['o', 's', '^', 'D']

        for i, result in enumerate(all_fitting_results):
            isolist = result['isolist']
            step_factor = result['step_factor']

            if len(isolist.sma) > 0:
                smas_kpc = isolist.sma * kpc_per_pixel
                errors = isolist.grad_r_error

                plt.plot(smas_kpc, errors, markers[i % len(markers)], alpha=0.7,
                         label=f"Step {step_factor}", zorder=10 - i)

        plt.axhline(y=0.1, color='r', linestyle='--', label='Error threshold')
        plt.xlabel('Semi-major axis (kpc)')
        plt.ylabel('Gradient Error')
        plt.title('Fitting Progress')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        plt.tight_layout()
        plt.show()

    def get_isophote(self, existing_properties):
        """
        Fit elliptical isophotes to galaxy images using moment-based initialization.

        Args:
            existing_properties: Dictionary containing required properties:
                - 'halo_images': List of image data arrays
                - 'image_reffs': Effective radii for each image (in kpc)
                - 'image_orientations': List of orientation angles
                - 'Rhalf': Half-light radius used for scaling (in pixels)

        Returns:
            List of isophote parameters for each orientation
        """
        images = existing_properties['halo_images'][0:2]
        reff_values = existing_properties['image_reffs'][0:2]
        orientations = existing_properties['image_orientations'][0:2]
        Rhalf = existing_properties['Rhalf']

        # Step size factors ordered from largest to smallest
        step_size_factors = [1.0, 0.5, 0.25, 0.125]

        # Calculate kpc per pixel scale
        kpc_per_pixel = (9 * Rhalf) / images[0].shape[0]
        #logger.info(f"kpc_per_pixel scale: {kpc_per_pixel:.4f}")

        # Check if we should use parallel processing
        n_cores = min(40, len(images))  # Use at most 40 cores or number of images
        n_cores = 1


        params = pymp.shared.dict()
        prog = pymp.shared.array((1,), dtype=int)
        prog[0] = 0


        # Precompute initial estimates to avoid recalculation in parallel processing
        initial_params = []
        for k, img in enumerate(images):
            try:
                # Only visualize the first image if visualization is enabled
                plot_this_image = self.visualization_enabled and k == 70

                center_x, center_y, initial_eps, initial_pa, radius_factor = self.estimate_initial_params(
                    img, reff_values[k], kpc_per_pixel, plot=plot_this_image)

                initial_params.append((center_x, center_y, initial_eps, initial_pa, radius_factor))

            except Exception as e:
                logger.error(f"Error estimating initial parameters for image {k}: {e}")
                traceback.print_exc()
                raise


        print(f'\tGenerating images: {round(prog[0] / len(images) * 100, 2)}%')
        with pymp.Parallel(n_cores) as p:
            for k in p.range(0, len(images)):
                try:
                    image_data = images[k]
                    radius = reff_values[k]
                    orientation = orientations[k]

                    # Get precomputed initial parameters
                    center_x, center_y, initial_eps, initial_pa, radius_factor = initial_params[k]
                    center = (center_x, center_y)

                    # Fit isophotes with iterative refinement
                    param_i, all_targets_met = self.fit_single_image(image_data,radius,step_size_factors,
                                                    center, initial_eps, initial_pa,radius_factor,
                                                    kpc_per_pixel, plot=self.visualization_enabled and k == 39)


                    # Log success or warning
                    # if all_targets_met:
                    #     logger.info(f"Successfully fit all target radii for orientation {orientation}")
                    # else:
                    #     logger.warning(f"Could not fit all target radii for orientation {orientation}")

                except Exception as e:
                    print(f'Error in orientation {orientation}: {e}')
                    traceback.print_exc()
                    logger.error(f'Error in orientation {orientation}: {e}')
                    raise

                with p.lock:
                    params[orientation] = param_i
                prog[0] += 1
                myprint(f'\tGenerating images: {round(prog[0] / (len(images)) * 100, 2)}%',
                        clear=True)



        params_dict = dict(params)


        # Sort by orientation
        params_sorted = dict(sorted(params_dict.items()))

        # Unpack results from sorted dictionary
        isophote_parameters = [params_sorted[key] for key in params_sorted]

        return isophote_parameters



    def live_calculate(self, existing_properties):
        """Entry point for analysis"""
        logger.info("Starting IsophoteAnalysis")

        # Process images and get parameters
        isophote_parameters = self.get_isophote(existing_properties)

        return isophote_parameters




class r_80(PynbodyPropertyCalculation):
    # get the radius that contains 80% of the mass of the stars
    names = ['r_80']
    def __init__(self, simulation):
        super().__init__(simulation)

    def calculate(self, halo, existing_properties):
        N_star = len(halo.s)
        # get radius that contains 80% of star particles
        rsort = halo.s['r'][np.argsort(halo.s['r'])]
        r_8 = rsort[int(0.8 * N_star)]
        return r_8



class Shape_Profile(PynbodyPropertyCalculation):
    """
    Calculate the axis ratio of the stellar and dark matter components of a halo, only for where stars and dark matter both exist.
    """
    names = ['ba_s', 'ca_s', 'rbins_s', 'ba_d', 'ca_d', 'rbins_d']

    def __init__(self, simulation):
        super().__init__(simulation)

    @staticmethod
    def process_shape(particles, rin, rout, bins):
        rbins, axis_lengths, num_particles, rotations = shape(particles,
                                                              nbins=bins,
                                                              ndim=3, rmin=rin,
                                                              rmax=rout,
                                                              max_iterations=175,
                                                              tol=5e-3,
                                                              justify=False)
        ba = axis_lengths[:, 1] / axis_lengths[:, 0]
        ca = axis_lengths[:, 2] / axis_lengths[:, 0]
        shape_dict = {'ba': ba, 'ca': ca, 'rbins': rbins}
        return shape_dict

    def _get_shape(self, halo):
        nan_array= np.array([np.nan]*100)
        nan_dict  = {'ba': nan_array, 'ca': nan_array, 'rbins': nan_array}
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        rin = 0.1
        rout = None
        # Find the stellar shape of the halo
        try:
            # Find the number of particles in the halo
            N_star = len(halo.s)
            if N_star == 0:
                starshape = nan_dict
            else:
                # Find the number of bins to use
                bins = int(get_bins(N_star))
                starshape = self.process_shape(halo.s, rin, rout, bins)

        except Exception as e:
            #print(f'Error in halo {halo}: {e}')
            #raise error
            print(N_star)
            raise e
            traceback.print_exc()
            starshape = nan_dict
        try:
            N_dm = len(halo.dm)
            bins = get_bins(N_dm)
            darkshape = self.process_shape(halo.dm, rin, rout, bins)
                
        except Exception as e:
            #print(f'Error in halo {halo}: {e}')
            traceback.print_exc()
            darkshape = nan_dict
            raise e
        return starshape, darkshape

    def calculate(self, halo, existing_properties):
        starshape,darkshape = self._get_shape(halo)
        ba_s, ca_s, rbins_s = starshape['ba'], starshape['ca'], starshape['rbins']
        ba_d, ca_d, rbins_d = darkshape['ba'], darkshape['ca'], darkshape['rbins']
        return ba_s, ca_s, rbins_s, ba_d, ca_d, rbins_d


    def plot_xlog(self):
        return False

    def plot_ylog(self):
        return False

    def plot_xlabel(self):
        return r'$r_{bins}$ (kpc)'
    def plot_ylabel(self):
        return r'$b/a or c/a$'

    def plot_xvalues(self,for_data):
        # need someway to find out if this is the stellar or dark matter profile and return the correct rbins
        return
        

class SmoothAxisRatio(LivePropertyCalculation):
    names = ['ba_s_smoothed', 'ca_s_smoothed', 'ba_d_smoothed', 'ca_d_smoothed']

    def requires_property(self):
        return ['ba_s', 'ba_d', 'ca_s', 'ca_d', 'rbins_s','rbins_d']

    @staticmethod
    def nan_func(x):
        return np.nan
    @staticmethod
    def smooth_shape(rbins, ba, ca, k=4):
        s_factor = 1
        """
        Smooth and filter data, handling a few NaN values gracefully.

        Parameters:
        rbins, ba, ca: array-like, input data
        k: int, degree of the smoothing spline (default 3)
        s_factor: float, smoothing factor as a fraction of len(rbins) (default 0.01)
        residual_threshold, jump_threshold, jump_percentage: unused in this version

        Returns:
        rbins, ba, ca: filtered arrays
        ba_s, ca_s: smoothed spline functions
        """
        import numpy as np
        from scipy.interpolate import UnivariateSpline

        # Remove rows where either ba or ca is NaN
        mask = ~np.isnan(ba) & ~np.isnan(ca)
        rbins_filtered = rbins[mask]
        ba_filtered = ba[mask]
        ca_filtered = ca[mask]

        # Calculate smoothing parameter
        s = s_factor * len(rbins_filtered)

        # Create splines
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)



        n = len(rbins_filtered)
        # calculate residuals and remove outliers
        ba_residuals = ba_filtered - ba_s(rbins_filtered)
        ca_residuals = ca_filtered - ca_s(rbins_filtered)
        # calculate the standard deviation of the residuals
        ba_std = np.std(ba_residuals)
        ca_std = np.std(ca_residuals)
        # remove outliers
        d = 3

        mask = np.abs(ba_residuals) < d * ba_std

        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]
        mask = np.abs(ca_residuals[mask]) < d * ca_std
        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]
        # Recreate splines
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)

        diff = np.diff(rbins_filtered, prepend=0)

        mask = diff > 1

        rbins_filtered = rbins_filtered[~mask]
        ba_filtered = ba_filtered[~mask]
        ca_filtered = ca_filtered[~mask]
        # Recreate splines
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)


        return rbins_filtered, ba_filtered, ca_filtered, ba_s, ca_s

    def calculate(self, halo, existing_properties):
        rbins_s = existing_properties['rbins_s']
        rbins_d = existing_properties['rbins_d']
        #print(rbins_s,rbins_d)
        rbins_s, ba_s, ca_s, ba_s_spline, ca_s_spline = self.smooth_shape(rbins_s, existing_properties['ba_s'],
                                                                            existing_properties['ca_s'])
        rbins_d, ba_d, ca_d, ba_d_spline, ca_d_spline = self.smooth_shape(rbins_d, existing_properties['ba_d'],
                                                                            existing_properties['ca_d'])

        return ba_s_spline, ca_s_spline, ba_d_spline, ca_d_spline
    

class DynamicalMass(PynbodyPropertyCalculation):
    """
    Calculate the dynamical mass of a halo using the circular velocity profile at the half-light radius.
    :returns: Mdyn in Msol
    """
    names = 'Mdyn'

    def requires_property(self):
        return ['Rhalf']

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        Rhalf = existing_properties['Rhalf']
        #Rhalf = pynbody.analysis.luminosity.half_light_r(halo)

        prof = pynbody.analysis.profile.Profile(halo,type='lin',min=.25,max=5*Rhalf,ndim=2,nbins=int((5*Rhalf)/0.1))
        indeff = np.argmin(np.abs(prof['rbins']-Rhalf))
        veff = prof['v_circ'][indeff]
        Mdyn=  ( (Rhalf*1e3)*veff**2)/(4.3009172706e-3)

        return Mdyn

class SersicFit(PynbodyPropertyCalculation):
    names = ['reff', 'rhalf']

    @staticmethod
    def sersic(r, mueff, reff, n):
        return mueff + 2.5 * (0.868 * n - 0.142) * ((r / reff) ** (1. / n) - 1)

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        # Get the surface density profile
        try:
            Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        except:
            Rhalf = np.nan
        try:
            prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25,
                                                    max=5 * Rhalf, ndim=2,
                                                    nbins=int(
                                                        (5 * Rhalf) / 0.1))
            vband = prof['sb,V']
            smooth = np.nanmean(
                np.pad(vband.astype(float), (0, 3 - vband.size % 3),
                       mode='constant', constant_values=np.nan).reshape(
                    -1, 3), axis=1)
            x = np.arange(len(smooth)) * 0.3 + 0.15
            x[0] = .05
            if True in np.isnan(smooth):
                x = np.delete(x, np.where(np.isnan(smooth) == True))
                y = np.delete(smooth, np.where(np.isnan(smooth) == True))
            else:
                y = smooth
            r0 = x[int(len(x) / 2)]
            m0 = np.mean(y[:3])
            par, ign = curve_fit(self.sersic, x, y, p0=(m0, r0, 1),
                                 bounds=([10, 0, 0.5], [40, 100, 16.5]))
            reff = pynbody.array.SimArray(par[1], 'kpc')
        except:
            print("Sersic fit failed")
            print(traceback.format_exc())
            # set reff to value of later halo
            try:
                reff = halo.calculate('later(1).reff')
            except:
                reff = np.nan
        return reff, Rhalf
        # except:
        #     print("Sersic fit failed")
        #     print(traceback.format_exc())
        #     return np.nan


class dynamical_time(PynbodyPropertyCalculation):
    names = ['tdyn']

    def requires_property(self):
        return ['rbins']

    def calculate(self, halo, existing_properties):
        pynbody.analysis.angmom.faceon(halo)
        rbins = existing_properties['rbins']
        prof = pynbody.analysis.profile.Profile(halo, bins=rbins, ndim=2)
        mass_enc = prof['mass_enc']
        dyntime = (rbins ** 3 / (2 * pynbody.units.G * mass_enc)) ** (1 / 2)
        return dyntime


class BaryonicFractionReff(PynbodyPropertyCalculation):
    names = ['Mvir_within_reff', 'Mstar_within_reff', 'Mgas_within_reff',
             'Mb_mvir_within_reff']

    def requires_property(self):
        return ['reff', 'max_radius']

    @staticmethod
    def mass_properties_within_r(halo, r):
        # halo should be in physcial units, but just in case
        halo.physical_units()

        sphere_filter = pynbody.filt.Sphere(r)
        sphere = halo[sphere_filter]

        m_tot = (sphere['mass'].sum().in_units('Msol'))
        m_gas = (sphere.gas['mass'].sum().in_units('Msol'))
        m_star = (sphere.star['mass'].sum().in_units('Msol'))
        m_dm = (sphere.dm['mass'].sum().in_units('Msol'))
        m_vir_within_r = m_gas + m_star + m_dm
        # assert that all of these values are positive, and not close to 0 they are stored as pynbody SimArrays in units of solar masses
        # assert that m_tot is the sum of the other masses within floating point error
        assert np.isclose(m_tot, m_vir_within_r,
                          rtol=1e-10), f"Total mass is {m_tot}, sum of components is {m_gas + m_star + m_dm}"
        Mb_within_r = m_gas + m_star
        mb_mvir_within_r = Mb_within_r / m_vir_within_r

        return m_vir_within_r, m_star, m_gas, mb_mvir_within_r

    def calculate(self, halo, existing_properties):
        reff = existing_properties['reff']
        Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff = self.mass_properties_within_r(
            halo, reff)
        return Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff


class BaryonicFractionVirial(PynbodyPropertyCalculation):
    names = ['Mvir', 'Mstar', 'Mgas', 'Mb_mvir']

    def calculate(self, halo, existing_properties):
        m_gas = halo.gas['mass'].sum().in_units('Msol').view(np.ndarray)
        m_star = halo.star['mass'].sum().in_units('Msol').view(np.ndarray)
        m_dm = halo.dm['mass'].sum().in_units('Msol').view(np.ndarray)
        m_vir = halo['mass'].sum().in_units('Msol').view(np.ndarray)
        try:
            Mb = m_gas + m_star
            mb_mvir = Mb / m_vir
        except ZeroDivisionError:
            mb_mvir = np.nan

        return m_vir, m_star, m_gas, mb_mvir


class DMDensityProfile(PynbodyPropertyCalculation):
    names = ['rho_dm', 'rho_dm_rbins']

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        prof = pynbody.analysis.profile.Profile(halo.dm, type='log', min=0.1,
                                                ndim=3)
        rbins = prof['rbins']
        rho_dm = prof['density']
        return rho_dm, rbins



