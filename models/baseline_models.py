### Baseline models to compare InPCA trajectories to.
### kernel_density_estimator (KDE), 
### histogram with uniform prior, 
### normalizing flow with invertible neural network


import numpy as np
from ICL import MultiResolutionPDF
from sklearn.neighbors import KernelDensity

def silverman_bandwidth(data):
    n = len(data)
    std = np.std(data, ddof=1)  # Use sample standard deviation
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return 1.06 * min(std, iqr/1.34) * n**(-1/5)

def histogram_for_series(time_series, prec = 2):
    PDF_list = [histogram(None,prec)]
    for i in range(1, len(time_series)):
        points = time_series[:i]
        estimated_PDF = histogram(points,prec)
        PDF_list += [estimated_PDF]
    return PDF_list


def histogram(points = None, prec = 2):
    estimated_PDF = MultiResolutionPDF(prec)
    if points is None:
        return estimated_PDF
    # Define the bins
    bins = np.linspace(0, 10, 10**prec + 1)

    # Calculate histogram
    uniform_bias = 0.7
    counts, _ = np.histogram(points, bins=bins)
    counts = counts.astype(np.float64) + uniform_bias

    # Update the PDF with the calculated probabilities
    estimated_PDF.bin_height_arr = counts
    estimated_PDF.normalize()

    return estimated_PDF
        

# def KDE_for_series(time_series, kernel = 'gaussian', prec = 2, bw_list = None):
#     PDF_list = [KDE(None,kernel,prec)]
#     for i in range(1, len(time_series)):
#         points = time_series[:i]
#         n = len(points)
#         bw = None
#         if bw_list is not None:
#             bw = bw_list[n]            
#         estimated_PDF = KDE(points,kernel,prec,bw)
#         PDF_list += [estimated_PDF]
#     return PDF_list

# Calculate Silverman's rule bandwidths
def real_silverman_bandwidth(data):
    n = len(data)
    std = np.std(data, ddof=1)  # Use sample standard deviation
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bw = 1.06 * min(std, iqr/1.34) * n**(-1/5)
    if not np.isnan(bw):
        return bw
    else:
        return 1.06 * n**(-1/5)


def KDE_for_series(time_series, kernel = 'gaussian', prec = 2, bw_list = None):
    """
    Force default to real Silverman bandwidth
    """
    PDF_list = [KDE(None,kernel,prec)]
    for i in range(1, len(time_series)):
        points = time_series[:i]
        n = len(points)
        bw = None
        if bw_list is not None:
            bw = bw_list[n]     
        else:
            bw = real_silverman_bandwidth(points)
        estimated_PDF = KDE(points,kernel,prec,bw)
        PDF_list += [estimated_PDF]
    return PDF_list



        
def KDE(points=None, kernel = 'gaussian', prec = 2, bw = None):
    estimated_PDF = MultiResolutionPDF(prec)
    if points is None:
        return estimated_PDF
    if isinstance(kernel, str):
        n = len(points)
        if bw is None:
            kde = KernelDensity(kernel=kernel, 
                            bandwidth="silverman"
                            )
        else:            
            kde = KernelDensity(kernel=kernel, 
                            bandwidth=bw
                            )            
            
        kde.fit(points.reshape(-1, 1))
        def PDF(x_array):
            log_density = kde.score_samples(x_array.reshape(-1, 1))
            return np.exp(log_density).flatten()
        estimated_PDF.discretize(PDF, mode = 'pdf')
    # elif callable(kernel):
    #     if bw is None:
    #         bw = silverman_bandwidth(points)
        
    #     def PDF(x_array):
    #         result = np.zeros_like(x_array)
    #         for point in points:
    #             result += kernel((x_array - point) / bw)
    #         return result / (len(points) * bw)
        
    #     estimated_PDF.discretize(PDF, mode='pdf')
    elif callable(kernel):
        if bw is None:
            bw = silverman_bandwidth(points)
        
        def PDF(x_array):
            # Vectorized computation
            x_matrix = x_array[:, np.newaxis] - points
            kernel_matrix = kernel(x_matrix / bw)
            return np.sum(kernel_matrix, axis=1) / (len(points) * bw)
        
        estimated_PDF.discretize(PDF, mode='pdf')
    else:
        raise ValueError("Invalid kernel type. Must be a string or a callable function.")
    
    return estimated_PDF
