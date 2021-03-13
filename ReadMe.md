# baseballforecaster

`baseballforecaster` is a Python library built on top of [pybaseball](https://github.com/jldbc/pybaseball) for forecasting baseball performance and simulating fantasy baseball [drafts](https://github.com/ykeuter/ffl/blob/master/notebooks/mcts.ipynb). Currently, full-season monte carlo-based forecasting and drafting is available to generate risk-adjusted players to more accurately measure performance. We are developing in-season monte carlo-based forecasting, full season and in-season lookup-based forecasting, as well as full-season and in-season ML-based forecasting.

The `Forecaster` module is the standard method for full-season forecasting and drafting. The user can select the time period for analysis, the number of simulations, number of draft rounds, number of drafters, and number of draft iterations. We will continue to allow for greater flexbility and complexity as we develop and improve these techniques. You can further explore the source code in the [src](https://github.com/baileymorton989/baseballforecaster) folder.

## Installation

Installation will soon be made simple by using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install baseballforecaster
```
For now you can simply clone the repo and run the following:

```bash
pip install -e .
```

## Usage

Here is a simple example using `Forecaster` We will use the `pybaseball` library to scrape and consolidate the data and we can use `argparse` to provide ime period for analysis, the number of simulations, draft rounds, number of drafters, draft iterations, and the exploration factor. All results are conveniently stored as [pandas](https://pandas.pydata.org/) dataframes for further analysis 

`Forecaster` Example : 

```python

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
```

## Contributing
We are open to pull requests and look forward to expanding this library further to tackle more complex games. Please open an issue to discuss any changes or improvements.
To install `baseballforecaster`, along with the tools you need to develop and run tests, run the following in your virtualenv:

```bash
$pip install -e .[dev]
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
