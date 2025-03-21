import math
import os
import random
import sys
import timeit
import george
from george import kernels
from scipy.optimize import minimize
import george
from george import kernels
from scipy.optimize import minimize
import emcee

from numpy import array as a
        
import matplotlib.colors as mcolors
from tqdm import tqdm
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from sklearn.neighbors import KernelDensity


rng = np.random.default_rng()

colors = list(mcolors.XKCD_COLORS)


class Lightcurve:
    
    def __init__(self, time_arr, mag_arr, mag_err_arr, fwhm_arr, abs_mag_arr, abs_mag_err_arr, phase, ID, obj_type=None, parent_id=None, standardized=False):
        """
        Representation of a lightcurve (time-magnitude series)

        @param ID: The ID of the sequence. [ID]_n is the nth subsequence of curve [ID]
        @param obj_type: classification of lightcurve, if known
        @param parent_id: if this curve is a subsequence, this is the id of the curve it was sampled from
        @type time_arr: np.array
        @type mag_arr: np.array
        @type mag_err_arr: np.array
        @type ID: str
        """
        self.data = {}
        self.data["times"] = time_arr
        self.data["mag"] = mag_arr
        self.data["abs_mag"] = abs_mag_arr if abs_mag_arr is not None else np.empty_like(mag_arr)
        self.data["mag_err"] = mag_err_arr
        self.data["abs_mag_err"] = abs_mag_err_arr if abs_mag_err_arr is not None else np.empty_like(mag_arr)
        self.data["fwhm"] = fwhm_arr
        self.data["phase"] = phase if phase is not None else np.empty_like(mag_arr)
        self.folded = phase is not None
        self.data = Table(self.data)
        self._cadence = None

        for col in self.data.colnames:
            setattr(self,col,self.data[col])

        self.data.sort("times")
        
        self.id = ID
        self.obj_type = obj_type
        self.parent_id = parent_id
        self.num_subseqs = 0
        self._marker_color = None  # for plotting
        self.standardized = standardized

    @property
    def cadence(self):
        if self._cadence is not None:
            return self._cadence
        self._cadence = np.median(np.diff(self.times))
        return self._cadence
    
    @property
    def marker_color(self):
        # don't spend time choosing unless we're actually going to plot this curve
        if self._marker_color is None:
            self._marker_color = random.choice(colors)
        return self._marker_color

    @property
    def total_time(self):
        if len(self.times):
            return self.times[-1] - self.times[0]
        return 0
    
    def fold(self, period, normalize_phase=False, normalize_section=[-0.5,0.5]):
        '''
        Parameters
        ----------
        period : 
        normalize : if True, the phase will be set within [-0.5, 0.5]
        time : name of time column

        note that the light curve should be sorted by time before fold
        '''
        if self.folded==True:
            return
        phase = self.times.copy()
        phase = (phase - phase[0]) % period
        self.period = period
        self.phase_span = period
        if normalize_phase==True:
            self.phase_span = normalize_section[1] - normalize_section[0]
            phase = (phase / max(phase)) * self.phase_span + normalize_section[0]
        self.phase = phase
        self.data.sort('phase')
        self.folded = True
    
    def standardize(self, verbose=False):
        """
        Standardize this lightcurve inplace: subtract mean mag and divide by std dev to get mean mag of 0 and std dev of 1
        @rtype: None
        """
        if self.standardized:
            print(f"Warning! Standardizing curve {self.id} (already standardized)")
        mean_mag = np.mean(self.mag)
        std_dev_mag = np.std(self.mag)
        if verbose:
            print("Before standardization:")
            print(f"Mean mag: {mean_mag}")
            print(f"Standard deviation: {std_dev_mag}")
        self.mag -= mean_mag
        self.mag /= std_dev_mag
        self.mag_err /= std_dev_mag
        self.standardized = True
        if verbose:
            mean_mag = np.mean(self.mag)
            std_dev_mag = np.std(self.mag)
            print("After standardization:")
            print(f"Mean mag: {mean_mag}")
            print(f"Standard deviation: {std_dev_mag}")

    def _make_plot(self, coplot=None, fig=None, ax=None, abs_mag=False, **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel('Time (HJD)', fontsize=14)
            ax.set_ylabel('Mag', fontsize=14)
            ax.set_title('Lightcurve for Object ' + self.id, fontsize=16)
        else:
            assert fig
        if abs_mag:
            ax.scatter(self.times, self.abs_mag, **kwargs)
        else:
            ax.scatter(self.times, self.mag, **kwargs)

        # plt.errorbar(self.times, self.mag, yerr=self.mag_err, fmt='o', color=random.choice(colors))
        if coplot:
            for lightcurve in coplot:
                ax.scatter(lightcurve.times, lightcurve.mag, c=lightcurve.marker_color, **kwargs)
        ax.invert_yaxis()
        return fig

    def plot(self, coplot=None, fig=None, ax=None, abs_mag=False, **kwargs):
        if not fig:
            fig, ax = plt.subplots(figsize=(10, 5))
        fig = self._make_plot(coplot=coplot, fig=fig, ax=ax, abs_mag=abs_mag,**kwargs)

    # not in use, for posterity
    def manual_get_subseq(self, start_time, end_time):
        # bounds checking
        if self.times[0] > end_time or self.times[-1] < start_time:
            self.num_subseqs += 1
            return Lightcurve([], [], [], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type, standardized=False)

        # find first and last time indices
        curve_start = None
        curve_end = None
        for i in range(len(self.times)):
            if curve_start is None and self.times[i] >= start_time:
                curve_start = i
            if self.times[i] > end_time:
                curve_end = i
                break
        if curve_end is None:
            # we ran out of sequence before we hit the end - that's ok, we just need to set our end time or it will be None
            curve_end = len(self.times)

        sub_times = self.times[curve_start:curve_end]
        sub_mags = self.mag[curve_start:curve_end]
        sub_mag_err = self.mag_err[curve_start:curve_end]

        # print(f"making subseq for window between {start_time} and {end_time} with {len(sub_times)} data points.")
        self.num_subseqs += 1
        return Lightcurve(sub_times, sub_mags, sub_mag_err, ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id,
                          obj_type=self.obj_type, standardized=self.standardized)

        # gather times in between
        # associate with magnitude and mag errr
        # construct and return lightcurve object

    
    def get_subseq(self, start_time, end_time):
        if self.times[0] > end_time or self.times[-1] < start_time:
            self.num_subseqs += 1
            return Lightcurve([], [], [], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type)
        curve_indices = np.where(np.logical_and(start_time <= self.times, self.times<end_time))[0]
        self.num_subseqs += 1
        d = self.data[curve_indices]
        return Lightcurve(d["times"], d["mag"], d["mag_err"], d["fwhm"],d["abs_mag"],d["abs_mag_err"],d["phase"], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id,obj_type=self.obj_type)

    
    def random_subsequence(self, window_duration):
        # if not self.standardized:
        #     self.standardize()
        index_choices = np.where(self.times < self.times[-1]-window_duration)[0]
        if len(index_choices):
            index_choices = index_choices[-1]
        else:
            index_choices = 1
        start_idx = random.choice(range(index_choices))
        start = self.times[start_idx]
        end = start + window_duration
        return self.get_subseq(start, end)

    
    def sliding_subseqs(self, window_duration, time_interval):
        """
        Sample a window of duration window_duration, then move ahead time_interval (time_interval<<window_duration) and repeat
        @param window_duration: duration of subsequence, days
        @param time_interval: days between beginnings of each subsequence
        @return: list of subseqs
        """
        i = 0
        start = self.times[0]
        end = start + window_duration
        subseqs = []
        while start + i < self.times[-1]:
            subseqs.append(self.get_subseq(start + i, end + i))
            i += time_interval
        return subseqs


    def plot_subseqs(self, subsequences, start_times, end_times, also_plot_individual=False, abs_mag=False):
        """
        Plot this lightcurve, then overplot the curves of subsequences provided.
        @param subsequences: Lightcurve objects
        @param start_times: start times of the subsequences
        @param end_times: start times of the subsequences
        @param also_plot_individual: bool. if True, will also plot each subseq on its own axis in the figure
        """
        assert len(subsequences) == len(start_times) and len(start_times) == len(end_times)
        if also_plot_individual:
            fig, axes = plt.subplots(len(subsequences)+1, sharex=True, sharey=True, figsize=(18, 10))
            ax = axes[0]
            ax.set_title(f"Lightcurve", fontsize="x-small", loc="left", verticalalignment="top")
        else:
            fig, ax = plt.subplots()

        fig = self._make_plot(coplot=subsequences, fig=fig, ax=ax, abs_mag=abs_mag)
        i = 0
        for subseq, start, end in zip(subsequences, start_times, end_times):
            # draw lines showing where it should have sliced the subseq
            ax.axvline(x=start, color=subseq.marker_color, linestyle='--',
                       label=f't={start}')
            ax.axvline(x=end, color=subseq.marker_color, linestyle='--',
                       label=f't={end}')
            if also_plot_individual:
                if abs_mag:
                    axes[i+1].scatter(subseq.times, subseq.abs_mag, c=subseq.marker_color)
                else:
                    axes[i+1].scatter(subseq.times, subseq.mag, c=subseq.marker_color)
                axes[i+1].set_title(f"Subsequence {i+1}",fontsize="x-small",loc="left",verticalalignment="top")
            i += 1
        fig.supxlabel('Time (HJD)', fontsize=25 if also_plot_individual else "large")
        fig.supylabel('Abs Mag' if abs_mag else 'App Mag', fontsize=25 if also_plot_individual else "large",horizontalalignment="left")
        fig.suptitle('Lightcurve for Object ' + self.id, fontsize=25 if also_plot_individual else "large")
        fig.legend()
        fig.show()

    def synthesize_lightcurve(self,len_window_d,own_period,guess_length_scale=20, nwalkers=10, burn_in=100, production=200):
        x = a(self.times)
        y = a(self.abs_mag)-np.mean(self.abs_mag)
        err = a(self.abs_mag_err)
        
        guess_length_scale = 20
        
        signal_to_noises = np.abs(y) / np.sqrt(
                err ** 2 + (1e-2 * np.max(y)) ** 2
            )
        scale = np.abs(y[signal_to_noises.argmax()])
        
        kernel1 = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
            [guess_length_scale ** 2], ndim=1
        )
        
        kernel2 = kernels.ExpSine2Kernel(gamma=10, log_period=np.log(own_period), axes=0)
        kernel2 *= kernels.ExpSquaredKernel(metric=15,ndim=1, axes=0)
        
        kernel = kernel1 + kernel2
        
        model = george.GP(kernel)
        model.compute(x, err)
        
        def neg_ln_like(p):
            model.set_parameter_vector(p)
            return -model.log_likelihood(y)
        def grad_neg_ln_like(p):
            model.set_parameter_vector(p)
            return -model.grad_log_likelihood(y)
            
        result = minimize(neg_ln_like, model.get_parameter_vector(), jac=grad_neg_ln_like, method="L-BFGS-B")
        
        print(result.x)
        model.set_parameter_vector(result.x)
        
        def lnprob(p):
            model.set_parameter_vector(p)
            return model.log_likelihood(y, quiet=True) + model.log_prior()

        # success = False
        # while not success:
        #     try:
        #         initial = model.get_parameter_vector()
        #         ndim, nwalkers = len(initial), 10
        #         p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        #         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
                
        #         # print("Running burn-in...")
        #         p0, _, _ = sampler.run_mcmc(p0, burn_in)
        #         sampler.reset()
                
        #         # print("Running production...")
        #         sampler.run_mcmc(p0, production)
        #         success = True
        #     except RuntimeError as e:
        #         print(f"Got MCMC exception {e}. Doubling nwalkers")
        #         nwalkers *= 2

        # setup 
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(x[:, None])
        # nsamples = int(200*rng.random() + 400) 
        # test_x = np.unique(np.round(((kde.sample(nsamples).flatten() + rng.integers(min(x),max(x),size=1)[0] ) % max(x))))
        # nsamples = int(min(rng.random()+0.25,1)*len(self.mag))
        nsamples = len(self.mag)
        test_x = kde.sample(nsamples).flatten()
        # print("test_x:",test_x)
        # # pull from our mcmc sampler
        # samples = sampler.flatchain
        # w = np.random.randint(sampler.chain.shape[0])
        # n = np.random.randint(sampler.chain.shape[1])
        # model.set_parameter_vector(sampler.chain[w, n])
        
        # generate data
        mu, var = model.predict(y, test_x, return_var=True, return_cov=False, kernel=kernel2)
        std = np.sqrt(var)
        
        mean_error =  np.mean(err)
        mean_pred_std = np.mean(std)
        scaled_std = std * mean_error / mean_pred_std
        
        # drop high-error points
        valid_idx = np.where(scaled_std < 2*mean_error)
        test_x = test_x[valid_idx]
        mu = mu[valid_idx]
        scaled_std = scaled_std[valid_idx]
        # err = err[valid_idx]
        
        # add noise
        noise = a([rng.normal(mean, sigma)%(3*sigma)for mean, sigma in zip(mu, scaled_std)])
        # err += noise
        tag = rng.integers(1e4,1e5,1)[0]      
        
        lc = ASASSN_Lightcurve(test_x,mu,scaled_std,np.empty_like(mu),mu,scaled_std,None,ID=f"{self.id}_{tag}",parent_id=self.id,obj_type=self.obj_type)
        return lc.random_subsequence(len_window_d)
    
# subclass of the general lightcurve class that represents an ASAS-SN lightcurve read from a .dat file
class ASASSN_Lightcurve(Lightcurve):
    @classmethod
    def from_dat_file(cls, path: str, abs_mag=False, phase=False):
        """
        Read a provided .dat file representing an ASAS-SN lightcurve.
        @rtype: ASASSN_Lightcurve
        """
        lc_df = pd.read_csv(path, sep='\t',usecols=["HJD", "mag", "mag_err","FWHM"]+(["abs_mag","abs_mag_err"] if abs_mag else []) + (["phase"] if phase else []))
        time = lc_df['HJD'].astype(np.float32).to_numpy()  # convert the times to floats
        mag = np.array([float(v.replace(">", '').replace("<", '')) for v in
               lc_df['mag'].astype(str)])  # convert the mags, remove "<" and ">"
        mag_err = lc_df['mag_err'].astype(np.float32).to_numpy()
        fwhm = lc_df['FWHM'].astype(np.float32).to_numpy()
        abs_mag_arr = lc_df["abs_mag"] if abs_mag else None
        abs_mag_err = lc_df["abs_mag_err"] if abs_mag else None
        phase = lc_df["phase"] if phase else None

        # obj_type = lc_df["ML_classification"]
        obj_type = "SAGE_NONE"
        ID = ASASSN_Lightcurve.id_from_filename(path)
        return cls(time, mag, mag_err,fwhm,abs_mag_arr, abs_mag_err, phase, ID=ID,obj_type=obj_type)

    @classmethod
    def from_cleaned_file(cls,path: str):
        t = pd.read_csv(path,sep="\t")
        obj_type = "SAGE_NONE"
        ID = ASASSN_Lightcurve.id_from_filename(path)
        try:
            phase = t["phase"]
        except:
            phase = None
        return cls(t["HJD"], t["mag"], t["mag_err"],t["FWHM"],t["abs_mag"], t["abs_mag_err"], phase, ID=ID,obj_type=obj_type)

    @classmethod
    def from_df_pickle(cls, path: str):
        """
        Read a provided .pkl file representing a pickled dataframe.
        @rtype: ASASSN_Lightcurve
        """
        lc_df = lightly_unpickle(path)
        time = lc_df['HJD'].astype(np.float32).to_numpy()  # convert the times to floats
        mag = np.array([float(v.replace(">", '').replace("<", '')) for v in
               lc_df['mag'].astype(str)])  # convert the mags, remove "<" and ">"
        mag_err = lc_df['mag_err'].astype(np.float32).to_numpy()
        # obj_type = lc_df["ML_classification"]     commented out while writing twed.py
        ID = ASASSN_Lightcurve.id_from_filename(path)
        return cls(time, mag, mag_err, ID=ID)
    
    # @classmethod
    # def from_lc_pickle(cls, path: str):
    #     """
    #     Read a provided .pkl file representing a pickled lightcurve.
    #     @rtype: ASASSN_Lightcurve
    #     """
    #     return lightly_unpickle(path)

    @staticmethod
    def id_from_filename(filename):
        ID = filename.split(os.sep)[-1].split('.dat')[0].split('_')  # parse the ID
        ID = ID[0] + '_' + ID[1]
        ID = ID.replace("-", "_").replace("+", "_").replace(".", "_")
        return ID

    # @staticmethod
    # def filename_from_id(ID):
    #     i = ID.split('_')
    #     return f"{i[0]}-{i[1]}_{i[2]}.{i[3]}+{i[4]}.{i[5]}.dat"

    @staticmethod
    def filename_from_id(ID):
        i = ID.replace(" ","_").split('_')
        return f"{i[0]}_{i[1]}.dat"
        
    
    @staticmethod
    def parent_id_from_child_id(child_ID):
        return ASASSN_Lightcurve.id_from_filename(ASASSN_Lightcurve.filename_from_id(child_ID))

    def save(self, path: str):
        """
        Create a .dat file representing this ASAS-SN lightcurve.
        """
        self.data.write(path,format="csv",delimiter="\t")
        # cols = {"HJD":self.times, "mag":self.mag, "mag_err":self.mag_err, "FWHM":self.fwhm}
        # if self.abs_mag is not None:
        #     cols["abs_mag"] = self.abs_mag
        #     cols["abs_mag_err"] = self.abs_mag_err
        # if self.phase is not None:
        #     cols["phase"] = self.phase

        # df = pd.DataFrame(cols)
        # df.to_csv(path,sep="\t",index=None)
        

if __name__ == "__main__":
    # this main function is just used for testing stuff

    # "ASASSN-V_J055803.37-143014.8.dat"

    # this was used to create "lightcurves.txt" (a list of a couple lightcurve names so i don't have to call os.listdir):
    # files = os.listdir(pickled_df_dir)
    # with open("../all_pickled_lightcurve_names.txt", "w+") as f:
    #     f.write('\n'.join(files))
    # exit()

    with open("../all_lightcurve_names.txt", "r") as f:
        files = [p.replace("\n", '') for p in f.readlines()]

    # files = files[:20]
    # for i in tqdm(range(len(files)), desc='plotting lightcurves', position=0, leave=True):
    #     f = files[i]
    #     seq = ASASSN_Lightcurve.from_dat_file(os.path.join(lc_dir, f))
    # with open("all_lightcurve_names.txt", "r") as f:
    #     path = os.path.join(lc_dir, f.readline().replace("\n", ''))

    seq = ASASSN_Lightcurve.from_dat_file(os.path.join(lc_dir, random.choice(files)))
        # seq.plot()
    start = seq.times[math.floor(len(seq.times) / 4)]
    end = seq.times[math.floor(3 * len(seq.times) / 4)]

    sub_seq = seq.get_subseq(start, end)
    # seq.plot(coplot=[sub_seq])
    # seq.plot()
    # seq.plot_subseqs(subsequences=[sub_seq],start_times=[start],end_times=[end])

    subseqs = []
    starts = []
    ends = []
    timewindow = 250  # days
    offset = 200  # days
    start = seq.times[0]
    end = start+timewindow
    i = 0
    num_repeats = 3
    # print(f"Manual: {timeit.timeit(lambda: seq.manual_get_subseq(start+i, end+i), number=num_repeats)}s")
    print(f"Random subseq: {timeit.timeit(lambda: seq.random_subsequence(i), number=num_repeats)}s")

    while start+i < seq.times[-1]-timewindow:
        subseqs.append(seq.get_subseq(start+i, end+i))
        starts.append(start+i)
        ends.append(end+i)
        i += offset

    seq.plot_subseqs(subsequences=subseqs, start_times=starts, end_times=ends, also_plot_individual=True)

    subseqs = seq.sliding_subseqs(250,10)
    print(subseqs)