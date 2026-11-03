# importing libraries
import numpy as np 
import pandas as pd 

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from feature_engine.encoding import RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
from feature_engine.datetime import DatetimeFeatures
from feature_engine.transformation import PowerTransformer
from feature_engine.selection import SelectBySingleFeaturePerformance

import warnings

# Display Settings
sklearn.set_config(transform_output='pandas')
# outputs are in the form of DataFrame rather than np.array
warnings.filterwarnings('ignore') # ignore warning

# Pipelines 
def feature_transformer_pipeline():
    
    airline_transformer = Pipeline(steps=[
        # fill missing values with most frequent value
        ('imputer', SimpleImputer(strategy='most_frequent')),
    
        # group catergories to Other which are less than 10% in data
        ('grouper', RareLabelEncoder(tol=0.05, n_categories=2, replace_with='other', missing_values='ignore')),
    
        # OneHotEnconding
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    doj_transformer = Pipeline(steps=[
    
        # Date time feature extraction
        ('dt', DatetimeFeatures(features_to_extract=['month', 'week', 'day_of_week', 'day_of_month', 'day_of_year'], missing_values='ignore', yearfirst=True, format='mixed')),
    
        # Scaling
        ('scale', MinMaxScaler())
    ])
    
    location_p1 = Pipeline(steps=[
        # replacing the categories in categorical features by the mean value of the target variable shown by each category
        ('encoder', MeanEncoder(missing_values='ignore', unseen='ignore')),
    
        # tranforming it close to normal distribution
        ('transform', PowerTransformer())
    ])
    
    def is_north(df):
    
        col = df.columns.to_list()
        north_cities = ['Delhi', 'New Delhi', 'Kolkata', 'Mumbai']
    
        return (df.assign(
            **{
                f'{i}_is_north' : df.loc[:, i].isin(north_cities).astype(int) for i in col
            }
        )
        .drop(columns=col))
    
    source_destination_transformer = FeatureUnion([
        ('Pipe1', location_p1),
        ('Pipe2', FunctionTransformer(func=is_north))
    ])
    
    time_p1 = Pipeline(steps=[
        # extracting hour and min
        ('dt', DatetimeFeatures(features_to_extract=['hour', 'minute'], missing_values='ignore')),
        # scaling 
        ('scaler', MinMaxScaler())
    ])
    
    def time_period(df):
    
        def helper(data):
            
            data = pd.to_datetime(data).hour
            if 5<= data < 12:
                return 'morning'
            elif 12 <= data < 18:
                return 'afternoon'
            elif 18<= data <21:
                return 'evening'
            else:
                return 'night'
    
                
        col = df.columns.to_list()
    
        return (
            df.assign(
                **{
                    f'{i}_period': df.loc[:, i].apply(helper) for i in col
                }
            )
            .drop(columns=col)
        )
    
    time_p2 = Pipeline(steps=[
        # passing time_period function
        ('period', FunctionTransformer(func=time_period, feature_names_out=lambda transformer, input_features: [
            f'{col}_period' for col in input_features
        ])),
        # CountFrequencty encoder 
        ('encoder', CountFrequencyEncoder()),
        # scaling
        ('scaler', MinMaxScaler())
    ])
    
    dep_arrival_time_tranformer = FeatureUnion([
        ('part1', time_p1),
        ('part2', time_p2)
    ])
    
    def duration_cat(df):
    
        bins = [0, 180, 360, 700, float('inf')]
    
        labels = ['short', 'medium', 'long', 'very long']
        
        return pd.cut(
            df['duration'],
            bins=bins,
            labels=labels
        ).rename('duration_category').to_frame()
    
    duration_transformer = FeatureUnion([
        
        ('cat', Pipeline(steps=[
            # appling duration_cat function 
            ('dur_cat', FunctionTransformer(func=duration_cat, feature_names_out=lambda transformer, input_features: [
            'duration_category'
        ])),
            # label encoding
            ('encoder', OrdinalEncoder(categories=[['short', 'medium', 'long', 'very long']]))
            ])
        ),
    
        ('encoder', StandardScaler())
    ])
    
    def is_direct_flight(df):
        return df.total_stops.eq(0).astype(int).to_frame()
    
    stops_tansformer = Pipeline(steps=[
        # filling nan values with most_frequent
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # applying is direct flight function 
        ('direct', FunctionTransformer(func=is_direct_flight, feature_names_out=lambda transformer, input_features: [
            'total_stops'
        ]))
    ])
    
    def have_info(df):
        return df.additional_info.eq('No Info').astype(int).to_frame()
    
    info_transformer = FeatureUnion([
        ('pipe1', Pipeline(steps=[
            # handling missing values 
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # grouping less than 10% categoris to others
            ('grouper', RareLabelEncoder(tol=0.1, n_categories=2, replace_with='other')),
            # oneHot Encoding
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))    
            ])
        ),
    
        ('have_info', FunctionTransformer(func=have_info))
    ])
    
    col_transformer = ColumnTransformer(transformers=[
        
        ('airline', airline_transformer, ['airline']),
        ('doj', doj_transformer, ['date_of_journey']),
        ('location', source_destination_transformer, ['source', 'destination']),
        ('time', dep_arrival_time_tranformer, ['dep_time', 'arrival_time']),
        ('duration', duration_transformer, ['duration']),
        ('stops', stops_tansformer, ['total_stops']),
        ('info', info_transformer, ['additional_info'])
        
    ])
    
    model = RandomForestRegressor(max_depth=4, random_state=149,n_jobs=-1)
    selector = SelectBySingleFeaturePerformance(model, scoring='r2', cv=5, threshold=0.1)
    
    selection = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('selector', selector)
    ])

    # X_transform = selection.fit_transform(X, Y)

    return selection