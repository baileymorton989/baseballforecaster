#for data cleaning and analysis
import pandas as pd
import numpy as np
from random import randint

#for visualization
import matplotlib.pyplot as plt
import seaborn as sns

#for directory-related functions
import os
import glob
import getpass    

#for web-scraping baseball data
import pybaseball as pyb

#for drafting
import math
import random

#for clustering
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.model_selection import train_test_split, cross_validate

#import time to see how long the script runs for
import time
import datetime
from datetime import date, timedelta

#import tkinter to build GUIs
import tkinter as tk
from tkinter import filedialog

#for warnings
import warnings
warnings.filterwarnings("ignore")

#for progress bar
from tqdm import tqdm

#enter forecasting and drafting parameters
def entry():
    root = tk.Tk()
    root.geometry("400x300")
    root.title('Select Forecasting and Drafting Parameters')
    label_simulations = tk.Label(root, text='Choose the number of simulations for forecasting')
    entry_simulations = tk.Entry(root)
    label_num_competitors = tk.Label(root, text='Choose Number of Competitors')
    entry_num_competitors = tk.Entry(root)
    label_num_rounds = tk.Label(root, text='Choose the number of rounds in the draft')
    entry_num_rounds = tk.Entry(root)
    label_num_iterations = tk.Label(root, text="Choose the number of iterations for the Draft Agent's Exploration")
    entry_num_iterations = tk.Entry(root)
    label_simulations.pack()
    entry_simulations.pack()
    label_num_competitors.pack()
    entry_num_competitors.pack()
    label_num_rounds.pack()
    entry_num_rounds.pack()
    label_num_iterations.pack()
    entry_num_iterations.pack()

    def enter_params():
        global simulations
        global num_competitors
        global num_rounds
        global num_iterations
        simulations = int(entry_simulations.get())
        num_competitors = int(entry_num_competitors.get())
        num_rounds = int(entry_num_rounds.get())
        num_iterations = int(entry_num_iterations.get())
        root.destroy()

    def get_params():
        global dateStore
        dateStore = True
        enter_params()

    get_params_button = tk.Button(root, text='Submit', command= get_params)
    get_params_button.pack()
    		
    root.mainloop()
    
    return simulations, num_competitors, num_rounds, num_iterations
    
#allow the user to select a date range
def get_dates() : 
    root = tk.Tk()
    root.geometry("400x300")
    root.title('Select Start and End time')
    label_start = tk.Label(root, text='Start Year: YYYY')
    entry_start = tk.Entry(root)
    label_end = tk.Label(root, text='End Year: YYYY')
    entry_end = tk.Entry(root)
    label_start.pack()
    entry_start.pack()
    label_end.pack()
    entry_end.pack()

    def enter_year():
        global start_time
        global end_time
        start_time = datetime.datetime.strptime(entry_start.get(),'%Y')
        end_time =datetime.datetime.strptime(entry_end.get(),'%Y')
        root.destroy()

    def get_year():
        global dateStore
        dateStore = True
        enter_year()

    get_year_button = tk.Button(root, text='Submit', command= get_year)
    get_year_button.pack()
    		
    root.mainloop()
    
    #get range of years 
    date_range = pd.date_range(start=start_time, end = end_time,freq='D')
    
    #create dictionary to store years
    years = {str(date.year) : date.year for date in date_range}

    return years


#make a dictionary with a dataframe for each season for hitters, pitchers, and teams
def make_period_dicts(dictionary):  
    batter_df = {dic:pyb.batting_stats(int(dic), qual = False) for dic in dictionary.keys()}
    pitcher_df = {dic:pyb.pitching_stats(int(dic), qual = False) for dic in dictionary.keys()}

    return batter_df , pitcher_df
    
#forecaster class
class Forecaster:

    def __init__(self):
        self.user = getpass.getuser()
        self.today = date.today().strftime("%m_%d_%y")
        self.simulations, self.num_competitors, self.num_rounds, self.num_iterations = entry()
        self.years = get_dates()
        print('Downloading Data')
        print('')
        self.seasons_dict_batter, self.seasons_dict_pitcher = make_period_dicts(self.years)
        
    #perform monte carlo full season forecast
    def monte_carlo_forecast(self):   
        print('Constructing the Database')
        print('')
        #merge the frames together 
        def merge_dict(dfDict, onCols, how='outer', naFill=None):    
          keys = list(dfDict.keys())
          for i in range(len(keys)):
            key = keys[i]
            df0 = dfDict[key]
            cols = list(df0.columns)
            valueCols = list(filter(lambda x: x not in (onCols), cols))
            df0 = df0[onCols + valueCols]
            df0.columns = onCols + [(s + '_' + key) for s in valueCols] 
            if (i == 0):
              outDf = df0
            else:
              outDf = pd.merge(outDf, df0, how=how, on=onCols)   
          if (naFill != None):
            outDf = outDf.fillna(naFill)
          return(outDf)

        #get the column names
        def get_column_names(dictionary): 
            key_list = list(dictionary.keys())
            columns_list = list(dictionary[key_list[0]].columns)
            return columns_list

        self.pitcher_columns_list, self.batter_columns_list = get_column_names(self.seasons_dict_pitcher), get_column_names(self.seasons_dict_batter)
            
        #merge the seasons together
        def merge_season_dicts():
            self.merged_batter_seasons_dict = merge_dict(self.seasons_dict_batter, self.batter_columns_list, how = 'outer', naFill = None)
            self.merged_pitcher_seasons_dict = merge_dict(self.seasons_dict_pitcher, self.pitcher_columns_list, how = 'outer', naFill = None)
            return self.merged_batter_seasons_dict, self.merged_pitcher_seasons_dict

        merge_season_dicts()

        #make a dataframe for each hitter
        def make_player_dicts(dataframe):
            df = {name : dataframe[dataframe['Name']==name] for name in dataframe['Name']}
            return df

        self.batter_dict, self.pitcher_dict = make_player_dicts(self.merged_batter_seasons_dict), make_player_dicts(self.merged_pitcher_seasons_dict)

        #get the current year
        def get_year_names(dictionary):       
            keys_list = list(dictionary.keys())
            return keys_list

        self.years_list = get_year_names(self.years)
        self.current_year = self.years_list[-1]

        #get only the players who played in the current year
        def filter_for_current_players(dictionary, year):    
            current_dict = {name : dictionary[name] for name in dictionary.keys() if dictionary[name]['Season'].values[-1]==int(year)}
            return current_dict

        self.current_pitcher_dict, self.current_batter_dict = filter_for_current_players(self.pitcher_dict, self.current_year), filter_for_current_players(self.batter_dict, self.current_year)

        #raw stats for batters and pitchers
        def stats():
            batter_stats = ['1B', '2B','3B', 'HR','R','RBI','BB','SO','SB', 'IDfg']
            pitcher_stats = ['W', 'IP', 'ER', 'SO',  'BB', 'SV', 'HLD', 'IDfg']
            return batter_stats, pitcher_stats

        self.batter_stats, self.pitcher_stats = stats()

        #filter by these stats
        def filter_for_current_stats(dictionary, stats):   
            current_dict = {name:dictionary[name][stats] for name in dictionary.keys()}
            return current_dict

        self.current_stat_batter, self.current_stat_pitcher = filter_for_current_stats(self.current_batter_dict, self.batter_stats), filter_for_current_stats(self.current_pitcher_dict, self.pitcher_stats)

        #team names and their abbreviations
        def teams_abbreviatons():
            team_list = ['Diamondbacks-ARI', 'Braves-ATL', 'Orioles-BAL', 'Red Sox-BOS', 'Cubs-CHC', 
            'White Sox-CHW', 'Reds-CIN' , 'Indians-CLE' , 'Rockies-COL', 'Tigers-DET' ,
            'Marlins-MIA' ,'Astros-HOU' ,'Royals-KCR' ,'Angels-LAA','Dodgers-LAD',
            'Brewers-MIL' ,'Twins-MIN','Mets-NYM','Yankees-NYY','Athletics-OAK','Phillies-PHI', 
            'Pirates-PIT' ,'Padres-SDP' ,'Giants-SFG','Mariners-SEA', 'Cardinals-STL', 
            'Rays-TB' ,'Rangers-TEX' ,'Blue Jays-TOR' ,'Nationals-WSN']
            return team_list

        self.team_list = teams_abbreviatons()

        #split the team names
        def split_names(team_list) : 
            split_list = [team.split('-') for team in team_list]       
            return split_list

        self.split_teams = split_names(self.team_list)
            
        #create dict for team names
        def create_dict(team_list):     
            teams_dict = {team[1]: team[0] for team in team_list}    
            return teams_dict

        self.teams_dict = create_dict(self.split_teams)

        #get a list of the teams
        def get_team_name_lists(team_list):
            team_list_full = [team.split('-')[0] for team in team_list]
            team_list_abv = [team.split('-')[1] for team in team_list]    
            return team_list_full, team_list_abv

        self.team_list_full, self.team_list_abv = get_team_name_lists(self.team_list)

        #get all the schedules
        def get_schedules(team_list_abv, years_list, team_list_full):
            season_list = []
            season_list  = [{team_list_ful: {year_list:pyb.schedule_and_record(int(year_list), team_list_ab)}} for year_list in years_list for team_list_ab, team_list_ful in zip(team_list_abv, team_list_full)]       
            return season_list

        self.season_list = get_schedules(self.team_list_abv, self.years_list, self.team_list_full)

        #drop pitchers from the hitters list
        def drop_pitchers(current_stat_batter, current_stat_pitcher):
            for key in current_stat_pitcher.keys():
                if key in current_stat_batter.keys() and key in current_stat_pitcher.keys():
                    del current_stat_batter[key]       
            return current_stat_batter

        self.current_stat_batter = drop_pitchers(self.current_stat_batter, self.current_stat_pitcher)

        #create averages for each player for each stat
        def player_averager(dictionary):
            average_players ={}
            for key in dictionary.keys():
                average_players.update({key : dictionary[key].mean().round().to_frame().transpose()})
                average_players[key] = average_players[key].reset_index()
                average_players[key].rename(columns = {'index': 'Name'}, inplace = True)
                average_players[key]['Name']= key     
            return average_players

        self.average_batters, self.average_pitchers = player_averager(self.current_stat_batter), player_averager(self.current_stat_pitcher)

        #columns to merge on
        def merge_columns(average_batters, average_pitchers):
            #return list(average_batters['Mike Trout'].columns), list(average_pitchers['Gerrit Cole'].columns)
            return list(average_batters[list(average_batters.keys())[0]].columns), list(average_pitchers[list(average_pitchers.keys())[0]].columns)


        self.batter_columns, self.pitcher_columns = merge_columns(self.average_batters, self.average_pitchers)

        #merge the average players to create the clusters
        def average_merger(average_batters, batter_columns,average_pitchers, pitcher_columns):
            return merge_dict(average_batters, batter_columns, how = 'outer', naFill = None), merge_dict(average_pitchers, pitcher_columns, how = 'outer', naFill = None)

        self.merged_batter_df, self.merged_pitcher_df = average_merger(self.average_batters, self.batter_columns, self.average_pitchers, self.pitcher_columns)
       
        #separate starting and relief pitchers and account for overlap
        def separate_pitchers(merged_pitcher_df):
            starting_pitchers = merged_pitcher_df[(merged_pitcher_df['SV'] ==0) &(merged_pitcher_df['HLD'] ==0) | (merged_pitcher_df['Name']=='Joe Musgrove') | (merged_pitcher_df['Name']=='Brad Keller')]
            relief_pitchers = merged_pitcher_df[(merged_pitcher_df['SV'] >=1) & (merged_pitcher_df['SV'] <8) | (merged_pitcher_df['HLD']> 0)  & (merged_pitcher_df['Name'] !='Joe Musgrove') & (merged_pitcher_df['Name']!='Brad Keller')]
            closers =  merged_pitcher_df[(merged_pitcher_df['SV'] >10) & (merged_pitcher_df['HLD'] >= 0) & (merged_pitcher_df['Name'] !='Joe Musgrove') & (merged_pitcher_df['Name']!='Brad Keller')]
            return starting_pitchers, relief_pitchers, closers

        self.starting_pitchers, self.relief_pitchers, self.closers = separate_pitchers(self.merged_pitcher_df)

        #cluster players to obtain a prior distrbution for each stat
        print('Clustering Players')
        print('')
        def mean_shift(data,quantile) : 

            #split the data
            train = data.drop('Name', axis =1)
            if 'Cluster Label' in train.columns:            
                train = data.drop(['Name', 'Cluster Label', 'IDfg'], axis =1)
            else:
                pass
                
            #estimate the bandwith
            bandwidth = estimate_bandwidth(train, quantile=quantile, n_samples=100000)

            #instantiate the mean shift clustering object
            ms = MeanShift(bandwidth = bandwidth, bin_seeding = True, cluster_all =True, n_jobs = None )
            
            #fit the model to the training data
            ms.fit(train)
            
            #grab the cluster labels and centers
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            
            #find the number of  unique labels
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            
            #find the clusters
            cluster_finder = data
            cluster_finder['Cluster Label'] = labels
            
            #create the clusters
            clusters = [cluster_finder[cluster_finder['Cluster Label']==label] for label in labels_unique]

            #extract the summary statistics
            cluster_describers = [cluster.describe() for cluster in clusters]
            
            return cluster_finder, clusters, cluster_describers

        self.cluster_finder_batter, self.clusters_batter, self.cluster_describers_batter = mean_shift(self.merged_batter_df,0.16)
        self.cluster_finder_starting_pitcher, self.clusters_starting_pitcher, self.cluster_describers_starting_pitcher = mean_shift(self.starting_pitchers, 0.18)
        self.cluster_finder_relief_pitcher, self.clusters_relief_pitcher, self.cluster_describers_relief_pitcher = mean_shift(self.relief_pitchers, 0.2)
        self.cluster_finder_closer, self.clusters_closer, self.cluster_describer_closer = mean_shift(self.closers, 0.19)

        #match the pitcher subsets properly
        def subset_pitchers(dictionary, dataframe):
            for key in dictionary.keys():
                dictionary = {key: dictionary[key] for key in dataframe['Name']}
                    
            return dictionary

        self.current_stat_starting = subset_pitchers(self.current_stat_pitcher, self.cluster_finder_starting_pitcher)
        self.current_stat_relief = subset_pitchers(self.current_stat_pitcher, self.cluster_finder_relief_pitcher)
        self.current_stat_closer = subset_pitchers(self.current_stat_pitcher, self.cluster_finder_closer)

        #use the clusters to make distributions for rookies 
        #also use clusters for a similarity comparison for non-rookies
        def player_matcher(dictionary,dataframe,columns):
            for key in dictionary.keys() :
                label = int(dataframe[dataframe['Name'] == key]['Cluster Label'])
                dictionary[key].loc[key] = dataframe[dataframe['Cluster Label']==label][columns[1:]].mean().round()
            
            return dictionary

        self.full_batters = player_matcher(self.current_stat_batter, self.cluster_finder_batter,self.batter_columns)
        self.full_starters = player_matcher(self.current_stat_starting, self.cluster_finder_starting_pitcher,self.pitcher_columns)
        self.full_relievers = player_matcher(self.current_stat_relief, self.cluster_finder_relief_pitcher,self.pitcher_columns)
        self.full_closers = player_matcher(self.current_stat_closer, self.cluster_finder_closer,self.pitcher_columns)

        #sample over the player distributions        
        def sample_averager(dictionary,simulations):
            sample_players = {}
            sample_players_risk = {}
            for key in tqdm(dictionary.keys()):
                if len(dictionary[key]) > 1 :
                    for column in dictionary[key]:
                        if column == 'IDfg':
                            dictionary[key]= dictionary[key].replace([np.inf, -np.inf], np.nan).fillna(0) #if not needed, remove
                            randomizer = 0
                            means = 0
                            stdevs = 0
                            dictionary[key].loc[key + ' Mean',column] = means
                            dictionary[key].loc[key + ' Risk',column] = stdevs
                            sample_players.update({key : dictionary[key].loc[key + ' Mean'].to_frame().transpose()})
                            sample_players_risk.update({key: dictionary[key].loc[key + ' Risk'].to_frame().transpose()})
                            sample_players_risk[key]= sample_players_risk[key].replace([np.inf, -np.inf], np.nan).fillna(0) #if not needed, remove
                            sample_players[key] = sample_players[key].reset_index()
                            sample_players[key].rename(columns = {'index': 'Name'}, inplace = True)
                            sample_players_risk[key] = sample_players_risk[key].reset_index()
                            sample_players_risk[key].rename(columns = {'index': 'Name'}, inplace = True)
                        else:
                            dictionary[key]= dictionary[key].replace([np.inf, -np.inf], np.nan).fillna(0) #if not needed, remove
                            randomizer = [randint(int(0.9*np.mean(dictionary[key][column])) + int(0.1*min(dictionary[key][column])), int(0.1*np.mean(dictionary[key][column]))+ int(0.9*max(dictionary[key][column]))) for i in range(simulations)]
                            means = np.mean(randomizer)
                            stdevs = np.std(randomizer)
                            dictionary[key].loc[key + ' Mean',column] = means
                            dictionary[key].loc[key + ' Risk',column] = stdevs
                            sample_players.update({key : dictionary[key].loc[key + ' Mean'].to_frame().transpose()})
                            sample_players_risk.update({key: dictionary[key].loc[key + ' Risk'].to_frame().transpose()})
                            sample_players_risk[key]= sample_players_risk[key].replace([np.inf, -np.inf], np.nan).fillna(0) #if not needed, remove
                            sample_players[key] = sample_players[key].reset_index()
                            sample_players[key].rename(columns = {'index': 'Name'}, inplace = True)
                            sample_players_risk[key] = sample_players_risk[key].reset_index()
                            sample_players_risk[key].rename(columns = {'index': 'Name'}, inplace = True)
            
            return sample_players, sample_players_risk
        
        self.sample_batters, self.sample_batters_risk = sample_averager(self.full_batters, self.simulations)
        self.sample_starting_pitchers, self.sample_starting_pitchers_risk = sample_averager(self.full_starters, self.simulations)
        self.sample_relief_pitchers, self.sample_relief_pitchers_risk  = sample_averager(self.full_relievers, self.simulations)
        self.sample_closers, self.sample_closers_risk = sample_averager(self.full_closers, self.simulations)


        #get the dataframes for fantasy points
        #for point totals
        self.sample_batters = merge_dict(self.sample_batters, self.batter_columns)
        self.sample_starting_pitchers = merge_dict(self.sample_starting_pitchers, self.pitcher_columns)
        self.sample_relief_pitchers = merge_dict(self.sample_relief_pitchers, self.pitcher_columns)
        self.sample_closers = merge_dict(self.sample_closers, self.pitcher_columns)

        #for risk
        self.sample_batters_risk = merge_dict(self.sample_batters_risk, self.batter_columns)
        self.sample_starting_pitchers_risk = merge_dict(self.sample_starting_pitchers_risk, self.pitcher_columns)
        self.sample_relief_pitchers_risk = merge_dict(self.sample_relief_pitchers_risk, self.pitcher_columns)
        self.sample_closers_risk = merge_dict(self.sample_closers_risk, self.pitcher_columns)

        #calculate fantasy points for batters
        def fantasy_batter_points(dataframe):
            dataframe['Fantasy Total'] = 2*dataframe['1B'] + 4*dataframe['2B'] + 6*dataframe['3B'] + 8*dataframe['HR']+ 1*dataframe['BB'] + 1*dataframe['R']+ 1.5*dataframe['RBI'] -0.5*dataframe['SO'] + 3*dataframe['SB']
            
            return dataframe

        #for points
        self.sample_batters = fantasy_batter_points(self.sample_batters)

        #for risk
        self.sample_batters_risk = fantasy_batter_points(self.sample_batters_risk)

        #calculate fantasy points for pitchers
        def fantasy_pitcher_points(dataframe):
            
            dataframe['Fantasy Total'] = 0.99*dataframe['IP'] -3*dataframe['ER'] + 4*dataframe['W'] + 3*dataframe['SV']+ 3*dataframe['SO'] + 2*dataframe['HLD']
            
            return dataframe

        #for points
        self.sample_starting_pitchers = fantasy_pitcher_points(self.sample_starting_pitchers)
        self.sample_relief_pitchers = fantasy_pitcher_points(self.sample_relief_pitchers)
        self.sample_closers = fantasy_pitcher_points(self.sample_closers)

        #for risk
        self.sample_starting_pitchers_risk = fantasy_pitcher_points(self.sample_starting_pitchers_risk)
        self.sample_relief_pitchers_risk = fantasy_pitcher_points(self.sample_relief_pitchers_risk)
        self.sample_closers_risk = fantasy_pitcher_points(self.sample_closers_risk)
        
        print('')
        print('Simulation Complete')
        print('')

        #naive risk adjusted scores
        def risk_adjusted_scores(points, risk):
            
            #get risk adjusted scores
            risk_adjusted_score = []
            for score, risk in zip(points['Fantasy Total'], risk['Fantasy Total']):
                risk_adjusted_score.append(0.75*score - 0.25*risk)
            
            #make new dataframe
            risk_adjusted = pd.DataFrame({'IDfg': points['IDfg'],'Name':points['Name'].apply(lambda x : x.replace(' Mean', '')), 'Risk Adjusted Score': risk_adjusted_score})
            
            return risk_adjusted

        #hitters
        self.risk_adjusted_batters = risk_adjusted_scores(self.sample_batters, self.sample_batters_risk)
        self.risk_adjusted_batters['IDfg'] = self.merged_batter_df['IDfg']

        #pitchers
        self.risk_adjusted_starters = risk_adjusted_scores(self.sample_starting_pitchers, self.sample_starting_pitchers_risk) 
        self.risk_adjusted_relief = risk_adjusted_scores(self.sample_relief_pitchers, self.sample_relief_pitchers_risk)
        self.risk_adjusted_closers = risk_adjusted_scores(self.sample_closers, self.sample_closers_risk)

        #add positions
        def add_fielding_positions(start_time, end_time, players):
            
            #chadwick register for players who played in most recent season of analysis
            #this will be used for cross-referencing player IDs
            chadwick_register = pyb.chadwick_register()

            #lahman database to grab positions
            lahman = pyb.lahman.fielding()
            lahman['key_bbref'] = lahman['playerID']
            lahman = lahman.drop(columns = ['playerID'])
            lahman = lahman.drop_duplicates('key_bbref')
            
            #merge frames
            merged = pd.merge(lahman,chadwick_register, on = 'key_bbref', how = 'outer')
            merged = merged[['yearID','key_bbref', 'teamID','POS', 'key_fangraphs', 'name_first', 'name_last']]
            merged['IDfg'] = merged['key_fangraphs']
            merged.drop(columns = ['key_fangraphs'], inplace = True) #drop missing players for now, which is very few
            
            #merge with player positions
            players = pd.merge(players, merged, on = 'IDfg', how = 'left')
            players = players[['Name', 'POS', 'Risk Adjusted Score']]
            players.dropna(inplace = True)
            
            #fix Ohtani
            #we will find a way to add his pitching stats
            def ohtani(x):
                if x == 'P':
                    return 'SP'
                else:
                    return x
                
            players['POS'] = players['POS'].apply(lambda x : ohtani(x))
            
            return players

        self.risk_adjusted_batters = add_fielding_positions(start_time, end_time, self.risk_adjusted_batters)

        #add pitcher positions
        def add_pitching_positions(starters, relievers, closers):
            
            #naive criteria to separate into RP and SP
            starters['POS'] = ['SP' for i in range(0,len(starters))] 
            relievers['POS'] = ['RP' for i in range(0,len(relievers))] 
            closers['POS'] = ['RP' for i in range(0,len(closers))] 
            
            return starters, relievers, closers
            
        self.risk_adjusted_starters, self.risk_adjusted_relief, self.risk_adjusted_closers = add_pitching_positions(self.risk_adjusted_starters, self.risk_adjusted_relief, self.risk_adjusted_closers)

        #change IDs
        def id_changer(players):
            
            players['IDfg'] = [i for i in range(len(players))]
            players = players[['IDfg', 'Name', 'POS', 'Risk Adjusted Score']]
            
            return players

        self.risk_adjusted_starters = id_changer(self.risk_adjusted_starters)
        self.risk_adjusted_relief = id_changer(self.risk_adjusted_relief)
        self.risk_adjusted_closers = id_changer(self.risk_adjusted_closers)
        self.risk_adjusted_batters = id_changer(self.risk_adjusted_batters)

        #combine all players
        def combine_all_players(batters, starters, relievers, closers):
            
            players = batters.append(starters)
            players = players.append(closers)
            players = players.append(relievers)
            
            return players

        self.all_players = combine_all_players(self.risk_adjusted_batters, self.risk_adjusted_starters, self.risk_adjusted_relief, self.risk_adjusted_closers)
            
#Adapt Drafting Technique from : https://github.com/ykeuter/ffl/blob/master/notebooks/mcts.ipynb

#create the draft state so we know who has been taken and who is available and who's turn it is
class DraftState:
    def __init__(self, rosters, turns, freeagents, playerjm=None):
        self.rosters = rosters
        self.turns = turns
        self.freeagents = freeagents
        self.playerJustMoved = playerjm

#create a player object with relevant attributes
class MLBPlayer:
    def __init__(self, name, position, points):
        self.name = name
        self.position = position
        self.points = points
        
    def __repr__(self):
        return "|".join([self.name, self.position, str(self.points)])
    
#create weights the so algorithm can choose relative to which positions they have chosen from
def GetResult(self, playerjm):
    """ Get the game result from the viewpoint of playerjm.
    """
    if playerjm is None: return 0
    
    pos_wgts = {
        ("SP"): [.9, .9, .9 ,.6, .6 ,.6],
        ("RP"): [.8, .7, .6, .5,.5],
        ("C"): [.6,.4],
        ("1B"): [.7,],
        ("2B"): [.7, .6],
        ("SS"): [.7, .6],
        ("3B"): [.7, .6],
        ("1B", "2B", "3B", "SS", "OF", "C"): [.6],
        ("1B", "2B", "3B", "SS"): [.6],
        ("OF"): [.7,.7,.7,.5,.4],
    }

    result = 0
    # map the drafted players to the weights
    for p in self.rosters[playerjm]:
        max_wgt, _, max_pos, old_wgts = max(
            ((wgts[0], -len(lineup_pos), lineup_pos, wgts) for lineup_pos, wgts in pos_wgts.items()
                if p.position in lineup_pos),
            default=(0, 0, (), []))
        if max_wgt > 0:
            result += max_wgt * p.points
            old_wgts.pop(0)
            if not old_wgts:
                pos_wgts.pop(max_pos)
                
    # map the remaining weights to the top three free agents
    for pos, wgts in pos_wgts.items():
        result += np.mean([p.points for p in self.freeagents if p.position in pos][:3]) * sum(wgts)
        
    return result
        
# DraftState.GetResult = GetResult

#possible moves for each state, given the position
def GetMoves(self):
    """ Get all possible moves from this state.
    """
    pos_max = {"SP": 6, "RP": 5, "1B": 1, "C":2, "2B":2 , "3B":2, "SS": 2, "OF":5}

    if len(self.turns) == 0: return []

    roster_positions = np.array([p.position for p in self.rosters[self.turns[0]]], dtype=str)
    moves = [pos for pos, max_ in pos_max.items() if np.sum(roster_positions == pos) < max_]
    return moves

# DraftState.GetMoves = GetMoves

#update states after each move
def DoMove(self, move):
    """ Update a state by carrying out the given move.
        Must update playerJustMoved.
    """
    player = next(p for p in self.freeagents if p.position == move)
    self.freeagents.remove(player)
    rosterId = self.turns.pop(0)
    self.rosters[rosterId].append(player)
    self.playerJustMoved = rosterId
    
# DraftState.DoMove = DoMove

def Clone(self):
    """ Create a deep clone of this game state.
    """
    rosters = list(map(lambda r: r[:], self.rosters))
    st = DraftState(rosters, self.turns[:], self.freeagents[:],
            self.playerJustMoved)
    return st

# DraftState.Clone = Clone

# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        UCTK = 1000 #200 #2000 #100 #20000 
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + UCTK * math.sqrt(2*math.log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

#prepare the draft
def prepare_draft(DraftState) : 
    #create position weights for drafting importance
    DraftState.GetResult = GetResult

    #assign possible moves for each player at each state
    DraftState.GetMoves = GetMoves

    #update states of the draft after each move
    DraftState.DoMove = DoMove

    #create a deep clone of this game state
    DraftState.Clone = Clone
    
    return DraftState

#simulate a fantasy faseball draft
def draft(forecaster, MLBPlayer, DraftState, UCT) : 
    print('')
    print('Drafting')
    print('')
    #import projections
    forecaster.all_players.set_index('IDfg', inplace = True)
    forecaster.mlb_players = forecaster.all_players
    freeagents = [MLBPlayer(*p) for p in forecaster.mlb_players.itertuples(index=False, name=None)]

    #create draft competitors
    num_competitors = forecaster.num_competitors
    rosters = [[] for _ in range(num_competitors)] # empty rosters to start with

    #create number of rounds and turns
    num_rounds = forecaster.num_rounds
    turns = []
    # generate turns by snake order
    for i in range(num_rounds):
        turns += reversed(range(num_competitors)) if i % 2 else range(num_competitors)
        
    #create draft states
    state = DraftState(rosters, turns, freeagents)
    iterations = forecaster.num_iterations
    while state.GetMoves() != []:
        move = UCT(state, iterations)
        print(move, end=".")
        state.DoMove(move)
    print('')
    print('Draft Complete')
    #draft results
    return pd.DataFrame({"Team " + str(i + 1): r for i, r in enumerate(state.rosters)})

#convert the dataframes to excel sheets
def excel_converter(forecaster, draft_results):
    
    #excel file
    writer = pd.ExcelWriter(f'C:\\Users\\{forecaster.user}\\Downloads\\{end_time.year +1}_Projections_{forecaster.today}.xlsx')
    
    #Drafting
    draft_results.to_excel(writer, sheet_name = 'Mock Draft',index = False)
    
    #full list
    forecaster.all_players.to_excel(writer, sheet_name = 'All Players',index = False)
    
    #risk-adjusted
    forecaster.risk_adjusted_batters.to_excel(writer, sheet_name = 'Risk Adjusted Batters',index = False)
    forecaster.risk_adjusted_starters.to_excel(writer, sheet_name = 'Risk Adjusted Starters',index = False)
    forecaster.risk_adjusted_relief.to_excel(writer, sheet_name = 'Risk Adjusted Relief',index = False)
    forecaster.risk_adjusted_closers.to_excel(writer, sheet_name = 'Risk Adjusted Closers',index = False)

    #points
    forecaster.sample_batters.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_batters.to_excel(writer, sheet_name='Batters Projection',index = False)
    forecaster.sample_starting_pitchers.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_starting_pitchers.to_excel(writer, sheet_name='Starters Projection',index = False)
    forecaster.sample_relief_pitchers.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_relief_pitchers.to_excel(writer, sheet_name='Relievers Projection',index = False)
    forecaster.sample_closers.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_closers.to_excel(writer, sheet_name='Closers Projection',index = False)
    
    #risk
    forecaster.sample_batters_risk.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_batters_risk.to_excel(writer, sheet_name='Batters Risk',index = False)
    forecaster.sample_starting_pitchers_risk.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_starting_pitchers_risk.to_excel(writer, sheet_name='Starters Risk',index = False)
    forecaster.sample_relief_pitchers_risk.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_relief_pitchers_risk.to_excel(writer, sheet_name='Relievers Risk',index = False)
    forecaster.sample_closers_risk.drop(columns = ['IDfg'], inplace = True)
    forecaster.sample_closers_risk.to_excel(writer, sheet_name='Closers Risk',index = False)
    
    #clusters
    forecaster.cluster_finder_batter.to_excel(writer, sheet_name = 'Batter Clusters',index = False)
    forecaster.cluster_finder_starting_pitcher.to_excel(writer, sheet_name = 'Starting Clusters',index = False)
    forecaster.cluster_finder_relief_pitcher.to_excel(writer, sheet_name = 'Relief Clusters',index = False)
    forecaster.cluster_finder_closer.to_excel(writer, sheet_name = 'Closer Clusters',index = False)
      
    #save file
    writer.save()
    
    return forecaster

#call the excel converter
def call_converter(forecaster, draft_results):
    return excel_converter(forecaster, draft_results)
