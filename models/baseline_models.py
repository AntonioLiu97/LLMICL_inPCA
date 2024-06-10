### Baseline models to compare InPCA trajectories to.
### kernel_density_estimator (KDE), 
### histogram with uniform prior, 
### normalizing flow with invertible neural network


import numpy as np
from ICL import MultiResolutionPDF
from sklearn.neighbors import KernelDensity

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
        

def KDE_for_series(time_series, kernel = 'gaussian', prec = 2):
    PDF_list = [KDE(None,kernel,prec)]
    for i in range(1, len(time_series)):
        points = time_series[:i]
        estimated_PDF = KDE(points,kernel,prec)
        PDF_list += [estimated_PDF]
    return PDF_list

def KDE(points=None, kernel = 'gaussian', prec = 2):
    estimated_PDF = MultiResolutionPDF(prec)
    if points is None:
        return estimated_PDF
    else:
        n = len(points)
        bw = 1 / n ** (1/5)
        kde = KernelDensity(kernel=kernel, 
                            bandwidth="silverman"
                            # bandwidth="scott"
                            # bandwidth=bw
                            )
        kde.fit(points.reshape(-1, 1))
        def PDF(x_array):
            log_density = kde.score_samples(x_array.reshape(-1, 1))
            return np.exp(log_density).flatten()
        estimated_PDF.discretize(PDF, mode = 'pdf')
        return estimated_PDF
    
        