import pandas as pd
import numpy as np
import ast
from datetime import datetime

def user_data_preprocessing(users_df):
    """
    Preprocesses the user data for the deep learning model. Creates new features from the user data.
    Args:
        users_df (pd.DataFrame): The user data.
    Returns:
        pd.DataFrame: The preprocessed user data.
    """
    users_df = users_df.copy()

    # Rename columns for consistency
    users_df.rename(columns={
        'name': 'user_name',
        'review_count': 'user_review_count'
    }, inplace=True)

    # Parse datetime
    users_df['yelping_since'] = pd.to_datetime(users_df['yelping_since'])
    users_df['yelping_years'] = (pd.to_datetime('now') - users_df['yelping_since']).dt.days / 365.0
    users_df['yelping_years'] = users_df['yelping_years'].fillna(0)

    # Core engagement
    users_df['reviews_per_year'] = users_df['user_review_count'] / (users_df['yelping_years'] + 1)
    users_df['engagement_score'] = (users_df['useful'] + users_df['funny'] + users_df['cool'] + users_df['fans']) / (users_df['yelping_years'] + 1)
    users_df['engagement_per_review'] = (users_df['useful'] + users_df['funny'] + users_df['cool']) / (users_df['user_review_count'] + 1)

    # Compliments
    compliment_cols = [col for col in users_df.columns if col.startswith('compliment_')]
    users_df['total_compliments'] = users_df[compliment_cols].sum(axis=1)
    users_df['compliments_per_review'] = users_df['total_compliments'] / (users_df['user_review_count'] + 1)
    users_df['compliment_type_count'] = users_df[compliment_cols].astype(bool).sum(axis=1)

    # Social metrics
    users_df['num_friends'] = users_df['friends'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    users_df['friend_density'] = users_df['num_friends'] / (users_df['yelping_years'] + 1)
    users_df['friend_to_fan_ratio'] = users_df['num_friends'] / (users_df['fans'] + 1)

    # Elite years
    users_df['elite_years_count'] = users_df['elite'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    users_df['is_elite'] = users_df['elite'].apply(lambda x: 0 if x == '' else 1)
    users_df['elite_score'] = np.log1p(users_df['elite_years_count'])

    # Rating behavior
    users_df['tough_reviewer'] = (users_df['average_stars'] < 3.0).astype(int)

    # Grouped compliments
    social_comps = ['compliment_cool', 'compliment_funny', 'compliment_photos']
    thoughtful_comps = ['compliment_note', 'compliment_writer', 'compliment_plain']
    users_df['social_compliments'] = users_df[social_comps].sum(axis=1)
    users_df['thoughtful_compliments'] = users_df[thoughtful_comps].sum(axis=1)

    # Drop columns that won't be used directly
    drop_cols = ['friends', 'elite', 'yelping_since', 'cool', 'funny', 'useful', 'fans', 'average_stars', 'compliment_hot', 'compliment_list', 'compliment_note', 'compliment_writer', 'compliment_photos', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_hot', 'compliment_list', 'compliment_note', 'compliment_writer', 'compliment_photos', 'compliment_plain', 'compliment_more', 'compliment_profile', 'compliment_cute']
    users_df.drop(columns=drop_cols, inplace=True)

    return users_df

def fix_time(t):
    """
    Helper function to fix time string formatting.
    """
    parts = t.split(':')
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}"

def safe_parse_dict(s):
    """Helper function to safely parse dictionary-like strings."""
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except:
            return {}
    return {}

def extract_hour_features(hours_str):
    """ Helper function to extract hour features from the business data."""
    hours_dict = safe_parse_dict(hours_str)
    total_minutes = 0
    open_weekends = 0
    open_late = 0
    days_open = 0

    for day, time_str in hours_dict.items():
        try:
            open_time, close_time = time_str.split('-')
            open_dt = datetime.strptime(fix_time(open_time), "%H:%M")
            close_dt = datetime.strptime(fix_time(close_time), "%H:%M")

            # Handle overnight hours
            if close_dt <= open_dt:
                duration = (24 - open_dt.hour + close_dt.hour) * 60 + (close_dt.minute - open_dt.minute)
            else:
                duration = (close_dt.hour - open_dt.hour) * 60 + (close_dt.minute - open_dt.minute)

            total_minutes += duration
            days_open += 1

            if close_dt.hour >= 22:
                open_late = 1
            if day in ['Saturday', 'Sunday']:
                open_weekends = 1

        except Exception as e:
            continue

    return pd.Series({
        'weekly_hours': round(total_minutes / 60, 1),
        'open_weekends': open_weekends,
        'open_late': open_late,
        'days_open': days_open
    })
    
def business_data_preprocessing(business_df, top_n_categories=20, inference=False):
    """ Preprocessing pipeline for the business data.
    Args:
        business_df (pd.DataFrame): The business data.
        top_n_categories (int): The number of top categories to consider.
        inference (bool): Whether to perform inference.
    Returns:
        pd.DataFrame: The preprocessed business data.
    """
    # Filter for open restaurants
    business_df = business_df[
        business_df['categories'].apply(lambda x: isinstance(x, str) and 'Restaurants' in x)
    ]
    business_df = business_df[business_df['is_open'] == 1].copy()

    # Rename
    business_df.rename(columns={
        'stars': 'rating',
        'review_count': 'business_review_count',
        'name': 'business_name'
    }, inplace=True)

    # Log popularity, binary rating
    business_df['popularity'] = np.log1p(business_df['business_review_count'])
    business_df['high_rating'] = (business_df['rating'] >= 4).astype(int)

    # Parse attributes safely
    curated_attributes = [
        'BusinessAcceptsCreditCards', 'RestaurantsTakeOut', 'RestaurantsDelivery',
        'OutdoorSeating', 'ByAppointmentOnly', 'Caters', 'RestaurantsGoodForGroups',
        'RestaurantsReservations', 'RestaurantsPriceRange2', 'BikeParking', 'WheelchairAccessible', 'GoodForKids', 'DogsAllowed', 'HasTV', 'HappyHour', 'DriveThru'
    ]

    def extract_attribute(attr_str, key):
        attr_dict = safe_parse_dict(attr_str)
        val = attr_dict.get(key)

        if val == 'True' or val is True:
            return 1
        elif val == 'False' or val is False:
            return 0
        else:
            return -1  # unknown or missing


    for attr in curated_attributes:
        business_df[f'attr_{attr.lower()}'] = business_df['attributes'].apply(lambda x: extract_attribute(x, attr))

    # Category processing
    business_df['category_list'] = business_df['categories'].apply(lambda x: [c.strip() for c in x.split(',')])
    all_categories = business_df['category_list'].explode()
    top_categories = all_categories.value_counts().head(top_n_categories).index.tolist()

    cat_df = pd.DataFrame()
    for cat in top_categories:
        col_name = f"cat_{cat.lower().replace(' ', '_')}"
        cat_df[col_name] = business_df['category_list'].apply(lambda x: int(cat in x))

    # Extract hour features
    hour_features = business_df['hours'].apply(extract_hour_features)

    if inference:
        # Final dataframe
        final_df = pd.concat([
            business_df.drop(columns=[
            'attributes', 'categories', 'category_list',
            'address', 'postal_code',
            'latitude', 'longitude', 'hours'
        ]),
        cat_df,
        hour_features
    ], axis=1)
        
    else:
        final_df = pd.concat([
            business_df.drop(columns=[
            'attributes', 'categories', 'category_list',
            'address', 'postal_code', 'city', 'state',
            'latitude', 'longitude', 'hours'
        ]), 
        cat_df,
        hour_features
    ], axis=1)

    return final_df
