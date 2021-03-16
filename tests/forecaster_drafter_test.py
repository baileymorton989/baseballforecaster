#import the libraries
from baseballforecaster import Forecaster, Drafter

#perform the analysis:
if __name__ == "__main__":

    #create Forecaster object
    forecaster = Forecaster()

    #perform monte carlo simulation to forecast player performance
    forecaster.monte_carlo_forecast()
    
    #define the drafter object
    drafter = Drafter(forecaster)
    
    #simulate a fantasy baseball draft
    drafter.draft()
    
    #save the forecasting, clustering, and drafting results
    drafter.excel_converter()
