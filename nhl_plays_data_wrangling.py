# -*- coding: utf-8 -*-
"""
This script contains functions for wrangling scraped play-by-play and odds
data from nhl-games to generate relevant features for modeling.
We form statitistics in an expanding window, such that for each game we have 
statistics for both home and away teams averaged from their past games that
season as the set of features to be used for modeling and the game result as
the label.

Author: Lauri Heikka

"""

def process_season_data(year, period_min = 15):
    """
    This function takes as argument season year as a double digit integer and
    imports relevant play-by-play data and odds data, stored in the project
    folder in CSV-format.
    
    In addition, the function takes as an argument a user-selected minimum
    amount of games required to compute the expanding window statistics,
    period_min
    
    The function returns a full set of games for the season, for which we have
    statistics available (as determined by the period_min argument), ready 
    to be used as training data for the model.
    
    
    """
    
    """
    We begin by computing statistics for each game basedon the play-by-play
    data.
    
    """
    # Import relevant libraries and datasets based on the year selected
    import numpy as np
    import pandas as pd
    season_data = 'plays_season_{}.csv'.format(year)
    odds_data = 'odds{}.csv'.format(year)
    links_data = 'links{}.csv'.format(year)
    
    data = pd.read_csv(season_data, index_col=0)
    
    

    # The periods in which each play occurrs are stored as a string and
    # contain some oddities, so they need some wrangling.
    di = {"Per":999, "1st":1, "2nd":2,"3rd":3,
          "Gam":999, "OT'": 4,"Sho":999, "arl":float('NaN')}
    
    data['period'] = float('NaN')
  
    data.loc[data.team.isna(),'period'] = data[data.team.isna()].description.str.strip('End of ').str.slice(0, 3).values
    data.loc[(data.description == "Game End'") & (data.time != '20:00'), 'period'] = 4

    data.replace({'period': di}, inplace=True)

    data.period.fillna(method='bfill', inplace=True)    
    
    # Drop unnecessary items from plays
    data.dropna(subset=['team'], inplace=True)   
    data = data[(data.period != 999)]

    # Change leftpos values for periods 2 and 4 to account for changing sides
    values = -data.loc[(data.period == 2)|(data.period == 4), 'leftpos']
    data.loc[(data.period == 2)|(data.period == 4), 'leftpos'] = values
    
    # Wrangle gametime to numeric and constantly running
    data.time = pd.to_numeric(data.time.str.split(':', expand=True)[0]) + pd.to_numeric(data.time.str.split(':', expand=True)[1])/60

    data.loc[data.period == 2, 'time'] += 20
    data.loc[data.period == 3, 'time'] += 40
    data.loc[data.period == 4, 'time'] += 60

    # Get initial home and away teams from average shot positions
    data.description = data.description.str.lower()
    homes = data[data.description.str.contains('of net|over net|saved|goal!')].groupby(['gameid','team']).agg({'leftpos':'mean'}).reset_index(level='team')
    aways = homes.loc[homes.leftpos < 0, 'team']
    homes = homes.loc[homes.leftpos > 0, 'team']
    data = data.join(aways, on='gameid', rsuffix='away')
    data = data.join(homes, on='gameid', rsuffix='home')
    
    # Add dummies for different play types
    data['shot'] = 0
    data.loc[data.description.str.contains('blocked|of net|over net|saved|goal!'), 'shot'] = 1
    
    data['takeaway'] = 0
    data.loc[data.description.str.contains('takeaway'), 'takeaway'] = 1

    data['giveaway'] = 0
    data.loc[data.description.str.contains('giveaway'), 'giveaway'] = 1
    
    data['faceoff'] = 0
    data.loc[data.description.str.contains('faceoff'), 'faceoff'] = 1
    
    data['hit'] = 0
    data.loc[data.description.str.contains(' hit '), 'hit'] = 1
    
    data['block'] = 0
    data.loc[data.description.str.contains('blocked'), 'block'] = 1
    conds = [(data.block==1)&(data.team==data.teamaway), (data.block==1)&(data.team==data.teamhome)]
    choices = [data.teamhome, data.teamaway]
    data.team = np.select(conds, choices, default=data.team)
    
    

    data['saved'] = 0
    data.loc[data.description.str.contains(' saved '), 'saved'] = 1
    
    data['wide'] = 0
    data.loc[data.description.str.contains('of net|over net'), 'wide'] = 1
    
    data['goal'] = 0
    data.loc[data.description.str.contains('goal!'), 'goal'] = 1
    
    
    penalties = ['checking','tripping','sticking','interference','holding','delay of','slashing']
    
    data['penalty'] = 0
    data.loc[data.description.str.contains('|'.join(penalties)), 'penalty'] = 1
    
    # Faceoffs and shot distances are computed based on the shot positions
    data['h_off_fo'] = 0
    data.loc[(data.faceoff == 1)&(data.leftpos==-46.23), 'h_off_fo'] = 1
        
    data['a_off_fo'] = 0
    data.loc[(data.faceoff == 1)&(data.leftpos==46.23), 'a_off_fo'] = 1
    data.leftpos.describe()    
    data['dist'] = ((data['bottompos'])**2+(60-abs(data.leftpos))**2)**0.5
    data.dist = data.dist * data.shot

    data['angle'] = np.arcsin(abs(data['bottompos'])/data['dist'])
    data.angle = data.angle * data.shot

    data['wt_shots'] = data.shot * (data.time / data.period / 20 + 1) * data.period / 3
    

    # Compute game-statistics for each team in each game
    game_data = data.groupby(['gameid', 'team']).agg({
            'teamhome' : [lambda x: x.iloc[0]],
            'teamaway' : [lambda x: x.iloc[0]],
            'takeaway' : 'sum',
            'giveaway' : 'sum',
            'faceoff' : 'sum',
            'hit' : 'sum',
            'block' : 'sum',
            'saved' : 'sum',
            'wide' : 'sum',
            'goal' : 'sum',
            'penalty' : 'sum',
            'h_off_fo' : 'sum',
            'a_off_fo' : 'sum',
            'wt_shots' : 'sum',
            'shot' : 'sum',
            'dist' : 'sum',
            'angle' : 'sum',
            'period' : 'max'})
    
    # Distance and angle need to be divided by the shot amount, to get averages
    # instead of a sum of distances and angles
    game_data.dist = game_data.dist / game_data.shot
    game_data.angle = game_data.angle / game_data.shot
    game_data = game_data.T.reset_index(level=1, drop=True).T
    
    game_data.reset_index(level=1, inplace=True)
    

    #separate home stats and away stats
    home_stats = game_data[game_data.team == game_data.teamhome].drop(['teamhome','teamaway'], axis=1)
    away_stats = game_data[game_data.team == game_data.teamaway].drop(['teamhome','teamaway'], axis=1)
    
    home_stats.rename(columns = {'faceoff':'fow','h_off_fo':'off_fow', 'a_off_fo':'def_fow'}, inplace=True)
    away_stats.rename(columns = {'faceoff':'fow','h_off_fo':'def_fow', 'a_off_fo':'off_fow'}, inplace=True)
    
    
    home_stats_fin = home_stats.join(away_stats.drop('team', axis = 1), rsuffix='_allowed')
    away_stats_fin = away_stats.join(home_stats.drop('team', axis = 1), rsuffix='_allowed')
    away_stats_fin = away_stats_fin[home_stats_fin.columns]
    
    all_stats = home_stats_fin.append(away_stats_fin)
    
    all_stats.sort_index(inplace=True)
    
    
    # compute some additional statistics to aggregate many statistics together
    # in order  to limit the number of features
    all_stats['def_fo'] = all_stats.def_fow_allowed + all_stats.def_fow
    all_stats['off_fo'] = all_stats.off_fow_allowed + all_stats.off_fow
    all_stats['fop'] = all_stats.fow / (all_stats.fow + all_stats.fow_allowed)
    all_stats['svp'] = all_stats.saved_allowed / (all_stats.goal_allowed + all_stats.saved_allowed)
    all_stats['giveaway_rat'] = all_stats.giveaway - all_stats.giveaway_allowed
    all_stats['takeaway_rat'] = all_stats.takeaway - all_stats.takeaway_allowed
    all_stats['penalty_rat'] = all_stats.penalty - all_stats.penalty_allowed
    all_stats['angle_rat'] = all_stats.angle - all_stats.angle_allowed
    
    # drop the unnecessary features
    all_stats = all_stats.drop(['fow','fow_allowed','off_fow', 'def_fow','def_fow_allowed', 
                    'off_fow_allowed','block', 'saved','wide','fow_allowed',
                    'hit_allowed','block_allowed', 'saved_allowed', 'wide_allowed',
                    'giveaway','takeaway','giveaway_allowed','penalty_allowed',
                    'penalty','takeaway_allowed','angle','angle_allowed'], axis=1)
    all_stats = all_stats[all_stats.index.duplicated(keep=False)]
    
    # extract the winning (to be later used to form the label), using the 
    # goals column
    home_goals = all_stats[['team','goal','period']][1::2]
    away_goals = all_stats[['team','goal','period']][0::2]
    home_goals['winner'] = np.select([(home_goals.goal > away_goals.goal)&(home_goals.period == 3),
              (home_goals.goal < away_goals.goal)&(home_goals.period == 3)], [home_goals.team, away_goals.team])
    result = home_goals[['winner']]

    all_stats = all_stats.drop(['period_allowed','period'], axis=1)

    """
    We are now ready to produce game statistics, where for each team and each 
    game we have ex-ante available statistics from earlier in the season to be
    used as features in the predictive modeling phase.
    
    """

    # compute expanding-window means for the stats for each team
    roll_stats = all_stats.groupby('team').expanding(min_periods=period_min).mean()
    
    # lag the stats by one game, so we only have ex-ante available information
    roll_stats = roll_stats.groupby('team').shift(1)

    # construct the games from the expanding window stats
    games = game_data[['teamhome', 'teamaway']]
    games = games[~games.index.duplicated(keep='first')]
    games = games.join(roll_stats, on=['teamhome', 'gameid'])
    games = games.join(roll_stats, on=['teamaway', 'gameid'], rsuffix='_away')
    
    """
    Finally, we need to combine the game statistics with odds for each game, by
    joining the datasets on dates and the teams playing.
    The play-by-play data does not contain dates for the games, so we use a 
    separate dataset to get dates for the game.
    
    
    """
  
    # Import dates and odds, format dates to datetime format for both the odds
    # and the dates datasets
    links = pd.read_csv(links_data, index_col=0).reset_index(drop=True)
    links.link = pd.to_numeric(links.link.str.strip('http://www.nhl.com/gamecenter/'))
    links.set_index('link', inplace = True)
    links.date = links.date.str.split(', ', expand=True)[1]
    links['day'] = links.date.str.split(' ', expand=True)[1]
    links['month'] = links.date.str.split(' ', expand=True)[0]
    links['year'] = '2016'
    
    links.date = pd.to_datetime(links.day + links.month + links.year, format='%d%b%Y')
    links.drop(['day', 'month', 'year'], axis=1, inplace=True)
    
    odds = pd.read_csv(odds_data, index_col=0).reset_index(drop=True)
    odds.date = odds.date.str.strip(' - Pre-season').str.strip(' - All St')
    odds_dates = odds.date.str.split(' ',n=2, expand=True)
    odds['month'] = odds_dates[1]
    odds['day'] = odds_dates[0]
    odds['year'] = '2016'
    
    odds.date = pd.to_datetime(odds.day + odds.month + odds.year, format='%d%b%Y')
    odds.drop(['day', 'month', 'year'], axis=1, inplace=True)
    
    # join the dates to the games data
    games = games.join(links, how='left').reset_index()
    
    
    def merge_games_odds(games, odds):
        """
        This is a helper function, that takes in the games and the odds
        datasets as arguments and returns a merged dataset. The play-by-play
        data does not assign home and away teams correctly, so we have to do
        two  separate merges to get all games. The home and away teams are 
        correctly assigned for the odds-data, so that one is used 
        """
        # merge games, where the home team is assigned correctly with odds data
        games_data = pd.merge(games, odds, how='left', left_on=['date', 'teamhome','teamaway'], right_on=['date','home','away'])
        games_data = pd.merge(games_data, odds, how='left', left_on=['date', 'teamhome','teamaway'], right_on=['date','away','home'])
        games_data.home_x.fillna(games_data.home_y, inplace=True)    
        games_data.away_x.fillna(games_data.away_y, inplace=True)
        games_data.odds_x_x.fillna(games_data.odds_x_y, inplace=True)  
        games_data.odds_1_x.fillna(games_data.odds_1_y, inplace=True)  
        games_data.odds_2_x.fillna(games_data.odds_2_y, inplace=True)  
        games_data.drop(['home_y', 'away_y', 'odds_1_y', 'odds_x_y', 'odds_2_y', 'date'], axis=1, inplace=True)
        games_data.rename({'home_x':'home', 'away_x':'away', 'odds_1_x':'odds_1', 'odds_x_x':'odds_x', 'odds_2_x':'odds_2'},
                          axis=1, inplace=True)
        
        # get the games, where home team is assigned incorrectly and flip the 
        # columns so that home and away stats are assigned to the correct team
        deranged_games = games_data.loc[games_data.teamhome == games_data.away,
                       ['gameid','teamhome','teamaway', 'hit_away', 'goal_away',
                       'wt_shots_away', 'shot_away', 'dist_away', 'goal_allowed_away',
                       'wt_shots_allowed_away', 'shot_allowed_away', 'dist_allowed_away',
                       'def_fo_away', 'off_fo_away', 'fop_away', 'svp_away',
                       'giveaway_rat_away', 'takeaway_rat_away', 'penalty_rat_away',
                       'angle_rat_away', 'hit', 'goal', 'wt_shots', 'shot',
                       'dist', 'goal_allowed', 'wt_shots_allowed', 'shot_allowed',
                       'dist_allowed', 'def_fo', 'off_fo', 'fop', 'svp', 'giveaway_rat',
                       'takeaway_rat', 'penalty_rat', 'angle_rat','home', 'away', 'odds_1',
                       'odds_x', 'odds_2']]
        deranged_games.index = games_data[games_data.teamhome == games_data.away].index
        deranged_games.columns = games_data.columns
        
        # merge the two datasets and sort, to return a final set of games
        games_data = games_data[games_data.teamhome == games_data.home]
        games_data = games_data.append(deranged_games).sort_index().dropna()
        return games_data
    
    # because the odds-data has dates in european timezone games played on one 
    # day in the US is assigned to the following day in Europe, we need to do
    # the merge twice, offsetting the date of the odds data dates by one day
    df1 = merge_games_odds(games, odds)
    odds.date -= pd.DateOffset(1)
    df2 = merge_games_odds(games, odds)
    
    # join the two merges and do some final prettifying
    final_data = df1.append(df2).sort_values('gameid').set_index('gameid').drop(['teamhome','teamaway'], axis=1)
    
    final_data = final_data.join(result)
 
    final_data.rename({'winner':'result'}, axis=1, inplace=True)
    
    # make the label column 'result', indicate the game result as a 1, x or 2
    #instead of the name of the winning team
    final_data.result = np.select([final_data.home == final_data.result, final_data.result == 0,
               final_data.result == final_data.away], ['1','x','2'])

    return final_data



def process_current_data(year, odds, period_min = 15):
    """
    This function repeats the data processing steps from the above function
    for live deployment of the model.
    
    The function takes in the same arguments as the function above and 
    in addition, a table of games available for betting with team names
    and current odds for the game
    
    """
    import numpy as np
    import pandas as pd
    
    season_data = 'plays_season_{}.csv'.format(year)
    
    data = pd.read_csv(season_data, index_col=0)
    
    

    # Wrangle periods
    di = {"Per":999, "1st":1, "2nd":2,"3rd":3,
          "Gam":999, "OT'": 4,"Sho":999, "arl":float('NaN')}
    
    data['period'] = float('NaN')
  
    data.loc[data.team.isna(),'period'] = data[data.team.isna()].description.str.strip('End of ').str.slice(0, 3).values
    data.loc[(data.description == "Game End'") & (data.time != '20:00'), 'period'] = 4

    data.replace({'period': di}, inplace=True)
    
    data.period.fillna(method='bfill', inplace=True)    
    
    # Drop unnecessary items from plays
    data.dropna(subset=['team'], inplace=True)   
    data = data[(data.period != 999)]

    # Change leftpos values for periods 2 and 4 to account for changing sides
    values = -data.loc[(data.period == 2)|(data.period == 4), 'leftpos']
    data.loc[(data.period == 2)|(data.period == 4), 'leftpos'] = values
    
    # Wrangle time to numeric and constantly running
    data.time = pd.to_numeric(data.time.str.split(':', expand=True)[0]) + pd.to_numeric(data.time.str.split(':', expand=True)[1])/60

    data.loc[data.period == 2, 'time'] += 20
    data.loc[data.period == 3, 'time'] += 40
    data.loc[data.period == 4, 'time'] += 60

    # Get home and away teams from shot positions 
    
    data.description = data.description.str.lower()
    homes = data[data.description.str.contains('of net|over net|saved|goal!')].groupby(['gameid','team']).agg({'leftpos':'mean'}).reset_index(level='team')
    aways = homes.loc[homes.leftpos < 0, 'team']
    homes = homes.loc[homes.leftpos > 0, 'team']
    data = data.join(aways, on='gameid', rsuffix='away')
    data = data.join(homes, on='gameid', rsuffix='home')
    
    # Add dummies for different play types
    data['shot'] = 0
    data.loc[data.description.str.contains('blocked|of net|over net|saved|goal!'), 'shot'] = 1
    
    data['takeaway'] = 0
    data.loc[data.description.str.contains('takeaway'), 'takeaway'] = 1

    data['giveaway'] = 0
    data.loc[data.description.str.contains('giveaway'), 'giveaway'] = 1
    
    data['faceoff'] = 0
    data.loc[data.description.str.contains('faceoff'), 'faceoff'] = 1
    
    data['hit'] = 0
    data.loc[data.description.str.contains(' hit '), 'hit'] = 1
    
    data['block'] = 0
    data.loc[data.description.str.contains('blocked'), 'block'] = 1
    conds = [(data.block==1)&(data.team==data.teamaway), (data.block==1)&(data.team==data.teamhome)]
    choices = [data.teamhome, data.teamaway]
    data.team = np.select(conds, choices, default=data.team)
    
    

    data['saved'] = 0
    data.loc[data.description.str.contains(' saved '), 'saved'] = 1
    
    data['wide'] = 0
    data.loc[data.description.str.contains('of net|over net'), 'wide'] = 1
    
    data['goal'] = 0
    data.loc[data.description.str.contains('goal!'), 'goal'] = 1
    
    
    penalties = ['checking','tripping','sticking','interference','holding','delay of','slashing']
    
    data['penalty'] = 0
    data.loc[data.description.str.contains('|'.join(penalties)), 'penalty'] = 1
    
    data['h_off_fo'] = 0
    data.loc[(data.faceoff == 1)&(data.leftpos==-46.23), 'h_off_fo'] = 1
        
    data['a_off_fo'] = 0
    data.loc[(data.faceoff == 1)&(data.leftpos==46.23), 'a_off_fo'] = 1
    data.leftpos.describe()    
    data['dist'] = ((data['bottompos'])**2+(60-abs(data.leftpos))**2)**0.5
    data.dist = data.dist * data.shot

    data['angle'] = np.arcsin(abs(data['bottompos'])/data['dist'])
    data.angle = data.angle * data.shot

    data['wt_shots'] = data.shot * (data.time / data.period / 20 + 1) * data.period / 3
    
    

    game_data = data.groupby(['gameid', 'team']).agg({
            'teamhome' : [lambda x: x.iloc[0]],
            'teamaway' : [lambda x: x.iloc[0]],
            'takeaway' : 'sum',
            'giveaway' : 'sum',
            'faceoff' : 'sum',
            'hit' : 'sum',
            'block' : 'sum',
            'saved' : 'sum',
            'wide' : 'sum',
            'goal' : 'sum',
            'penalty' : 'sum',
            'h_off_fo' : 'sum',
            'a_off_fo' : 'sum',
            'wt_shots' : 'sum',
            'shot' : 'sum',
            'dist' : 'sum',
            'angle' : 'sum'})
    game_data.dist = game_data.dist / game_data.shot
    game_data.angle = game_data.angle / game_data.shot
    game_data = game_data.T.reset_index(level=1, drop=True).T
    
    game_data.reset_index(level=1, inplace=True)

        
    #separate home stats and away stats
    home_stats = game_data[game_data.team == game_data.teamhome].drop(['teamhome','teamaway'], axis=1)
    away_stats = game_data[game_data.team == game_data.teamaway].drop(['teamhome','teamaway'], axis=1)
    
    home_stats.rename(columns = {'faceoff':'fow','h_off_fo':'off_fow', 'a_off_fo':'def_fow'}, inplace=True)
    away_stats.rename(columns = {'faceoff':'fow','h_off_fo':'def_fow', 'a_off_fo':'off_fow'}, inplace=True)
    
    
    home_stats_fin = home_stats.join(away_stats.drop('team', axis = 1), rsuffix='_allowed')
    away_stats_fin = away_stats.join(home_stats.drop('team', axis = 1), rsuffix='_allowed')
    away_stats_fin = away_stats_fin[home_stats_fin.columns]
    
    all_stats = home_stats_fin.append(away_stats_fin)
    
    all_stats.sort_index(inplace=True)
    
    
    
    
    all_stats['def_fo'] = all_stats.def_fow_allowed + all_stats.def_fow
    all_stats['off_fo'] = all_stats.def_fow_allowed + all_stats.def_fow
    all_stats['fop'] = all_stats.fow / (all_stats.fow + all_stats.fow_allowed)
    all_stats['svp'] = all_stats.saved_allowed / (all_stats.goal_allowed + all_stats.saved_allowed)
    
    all_stats['on_goal'] = (all_stats.saved + all_stats.goal)/all_stats.shot
    all_stats['on_goal_allowed'] = (all_stats.saved_allowed + all_stats.goal)/all_stats.shot
    
    #drop some extra columns to keep the stuff manageable
    """
    potentially leave this out depending on how the training goes
    """
    all_stats = all_stats.drop(['fow','fow_allowed','off_fow', 'def_fow','def_fow_allowed', 
                    'off_fow_allowed','block', 'saved','wide','fow_allowed',
                    'hit_allowed','block_allowed', 'saved_allowed', 'wide_allowed'], axis=1)
    
    
    #compute expanding-window means for the stats
    roll_stats = all_stats.groupby('team').expanding(min_periods=period_min).mean()

    roll_stats = roll_stats.groupby('team').shift(1).reset_index(level=0)
    
    teams = roll_stats[~roll_stats.team.duplicated(keep='last')]
    

    # finally, merge the computed team statistics with the current games to
    # form a full set of feature to use for generating predictions with the model
    games_data = pd.merge(odds, teams, how='left', left_on=['home'], right_on=['team'])
    games_data = pd.merge(games_data, teams.add_suffix('_away'), how='left', left_on=['away'], right_on=['team_away'])
    
    

    games_data = games_data[['takeaway', 'giveaway', 'hit', 'goal', 'penalty', 'wt_shots', 'shot',
                             'dist', 'angle', 'takeaway_allowed', 'giveaway_allowed', 'goal_allowed',
                             'penalty_allowed', 'wt_shots_allowed', 'shot_allowed', 'dist_allowed',
                             'angle_allowed', 'def_fo', 'off_fo', 'fop', 'svp', 'on_goal',
                             'on_goal_allowed', 'takeaway_away', 'giveaway_away', 'hit_away',
                             'goal_away', 'penalty_away', 'wt_shots_away', 'shot_away', 'dist_away',
                             'angle_away', 'takeaway_allowed_away', 'giveaway_allowed_away',
                             'goal_allowed_away', 'penalty_allowed_away', 'wt_shots_allowed_away',
                             'shot_allowed_away', 'dist_allowed_away', 'angle_allowed_away',
                             'def_fo_away', 'off_fo_away', 'fop_away', 'svp_away', 'on_goal_away',
                             'on_goal_allowed_away', 'home', 'away', 'odds_1', 'odds_x', 'odds_2']]

    games_data.dropna(inplace=True)
    return games_data


