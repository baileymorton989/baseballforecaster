#import the libraries
from baseballforecaster import Forecaster, DraftState, prepare_draft, draft, MLBPlayer, UCT, excel_converter

#perform the analysis:
if __name__ == "__main__":

    #create Forecaster object
    forecaster = Forecaster()

    #perform monte carlo simulation to forecast player performance
    forecaster.monte_carlo_forecast()
    
    #prepare draft
    DraftState = prepare_draft(DraftState)

    #simulate a fantasy baseball draft
    draft_results = draft(forecaster, MLBPlayer, DraftState, UCT)
    
    #save the file
    forecaster = excel_converter(forecaster, draft_results)