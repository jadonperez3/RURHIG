import ROOT

def gaussian(x, par): #defines gaussian curve with parameters defined by the user
    amp = par[0]
    mean = par[1]
    stdev = par[2]

    if stdev <= 0: #corrects for invalid stdev values
        return 0
    
    return amp * ROOT.TMath.Exp(-0.5 * ((x[0] - mean) / stdev) ** 2)

h = ROOT.TH1D("h", "Gaussian Histogram; Bins; Counts", 100, -10, 10) #Creates histogram with ("internal title", "title; Xlabel; Ylabel ", bins, xmin, xmax)
h.FillRandom("gaus", 100000) #Fills randomly from gaussian 100000 times
curve = ROOT.TF1("curve", gaussian, -10, 10, 3) #creates a fit curve between -10 and 10 with 3 parameters
curve.SetParameters(1000, 0.0, 1.0) #initial guesses for parameters
fit = h.Fit(curve, "S") #"S" option allows us to see the fit parameters at the end
canvas = ROOT.TCanvas()
h.Draw()
canvas.Draw()
with ROOT.TFile('gaussian.root',"RECREATE") as F:
    h.Write()
    F.Close()
