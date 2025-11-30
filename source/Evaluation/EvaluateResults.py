"""Class containing equations."""
from sympy import *

from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot

def CalculateBestModelOutput(model):
        # pass in a string view of the "model" as str(symplified_best)
        # this string view of the equation may reference any of the other inputs, AT, V, AP, RH we registered
        # we then use eval of this string to calculate the answer for these inputs
        return eval(model) 

def test_evaluate_results(
        holdout,
        symplified_best
):
    

    predPE = CalculateBestModelOutput(
        str(symplified_best))

    predPE.describe()
    predPE.head()

    print("Mean squared error: %.2f" % mean_squared_error(holdout.Ratio, predPE))
    print("R2 score : %.2f" % r2_score(holdout.Ratio, predPE))

    pyplot.rcParams['figure.figsize'] = [20, 5]
    plotlen=200
    pyplot.plot(predPE.head(plotlen))       # predictions are in blue
    pyplot.plot(holdout.Ratio.head(plotlen-2)) # actual values are in orange
    pyplot.show()

    pyplot.rcParams['figure.figsize'] = [10, 5]
    hfig = pyplot.figure()
    ax = hfig.add_subplot(111)

    numBins = 100
    #ax.hist(holdout.Ratio.head(plotlen)-predPE.head(plotlen),numBins,color='green',alpha=0.8)
    ax.hist(holdout.Ratio-predPE,numBins,color='green',alpha=0.8)
    pyplot.show()
