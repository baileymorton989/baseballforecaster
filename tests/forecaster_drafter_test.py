#import the libraries
from baseballforecaster import Forecaster, Drafter, entry, get_dates

#perform the analysis:
if __name__ == "__main__":
    
    #get parameters for forecasting
    simulations, num_competitors, num_rounds, num_iterations = entry()
    
    #get the time period for forecastings
    time_period = get_dates()

    #create Forecaster object
    forecaster = Forecaster(simulations, num_competitors, num_rounds, num_iterations,time_period)
    
    #run the forecast
    forecaster.monte_carlo_forecast()
    
    #define the drafter object
    drafter = Drafter(forecaster)
    
    #simulate a fantasy baseball draft
    drafter.draft()
    
    #save the forecasting, clustering, and drafting results
    drafter.excel_converter()
