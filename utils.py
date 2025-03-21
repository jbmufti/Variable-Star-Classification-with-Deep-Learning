import tomlkit
import numpy as np

def read_config(fpath):
    # read in a config file
    with open(fpath,"rb") as f:
        return tomlkit.load(f)

def n_hist_bins(data):
    # find approx appropriate number of bins for a histogram based on data
    return int(1+3.3*np.log(len(data)))

def hist(values,bins:int,**kwargs):
    # make an np histogram (binning) of input values
    bin_vals, bin_edges = np.histogram(values,bins=bins,**kwargs)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return bin_centers, bin_vals, bin_edges, bin_width