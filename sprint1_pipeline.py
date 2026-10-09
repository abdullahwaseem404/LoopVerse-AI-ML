import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

def aqi_to_pm25(aqi):
    if pd.isna(aqi): return np.nan
    if aqi <= 50: return aqi * (12.0 / 50.0)
    elif aqi <= 100: return ((aqi - 51) * (35.4 - 12.1) / 49) + 12.1
    elif aqi <= 150: return ((aqi - 101) * (55.4 - 35.5) / 49) + 35.5
    elif aqi <= 200: return ((aqi - 151) * (150.4 - 55.5) / 49) + 55.5
    elif aqi <= 300: return ((aqi - 201) * (250.4 - 150.5) / 99) + 150.5
    elif aqi <= 400: return ((aqi - 301) * (350.4 - 250.5) / 99) + 250.5
    else: return ((aqi - 401) * (500.4 - 350.5) / 99) + 350.5

def load_and_standardize():
    # Pointing to the exact file paths from the repository structure
    batch1 = pd.read_csv('sensors/batch_1_sensor_data.csv')
    batch2 = pd.read_csv('sensors/batch_2_sensor_data.csv')
    
    # Process Batch 1: UTC Timestamp -> PKT Date, reading_value is already PM2.5
    batch1['timestamp'] = pd.to_datetime(batch1['timestamp'])
    if batch1['timestamp'].dt.tz is None:
        batch1['timestamp'] = batch1['timestamp'].dt.tz_localize('UTC')
    batch1['timestamp'] = batch1['timestamp'].dt.tz_convert('Asia/Karachi')
    batch1['local_date'] = batch1['timestamp'].dt.date
    batch1.rename(columns={'reading_value': 'pm25'}, inplace=True)
    
    # Process Batch 2: PKT Timestamp -> PKT Date, reading_value is AQI
    batch2['timestamp'] = pd.to_datetime(batch2['timestamp'])
    batch2['local_date'] = batch2['timestamp'].dt.date
    batch2['pm25'] = batch2['reading_value'].apply(aqi_to_pm25)
    
    # Select only unified columns and merge
    c1_clean = batch1[['sensor_id', 'local_date', 'pm25']]
    c2_clean = batch2[['sensor_id', 'local_date', 'pm25']]
    df = pd.concat([c1_clean, c2_clean]).drop_duplicates(subset=['sensor_id', 'local_date'])
    return df

def clean_data(df):
    df['pm25'] = df['pm25'].replace(-999, np.nan)
    df = df.sort_values(by=['sensor_id', 'local_date']).reset_index(drop=True)
    
    # Detect 3-day flatline fault
    df['is_flatline'] = df.groupby('sensor_id')['pm25'].transform(lambda x: x == x.shift(1))
    df.loc[df['is_flatline'], 'pm25'] = np.nan
    
    daily_df = df.groupby(['sensor_id', 'local_date'])['pm25'].mean().reset_index()
    daily_df['pm25'] = daily_df.groupby('sensor_id')['pm25'].transform(lambda x: x.fillna(x.rolling(3, min_periods=1).mean()))
    return daily_df

def engineer_features(df):
    df['pm25_lag1'] = df.groupby('sensor_id')['pm25'].shift(1)
    df['pm25_rolling3'] = df.groupby('sensor_id')['pm25_lag1'].rolling(3).mean().reset_index(0, drop=True)
    return df.dropna()

def train_and_predict(df):
    # Dynamically read holdout requirements from the repository's holdout folder
    holdout_reqs = pd.read_csv('holdout/holdout_inputs.csv')
    holdout_reqs['target_date'] = pd.to_datetime(holdout_reqs['target_date']).dt.date
    
    # Filter training data chronologically
    min_holdout_date = holdout_reqs['target_date'].min()
    train = df[df['local_date'] < min_holdout_date]
    
    features = ['pm25_lag1', 'pm25_rolling3']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(train[features], train['pm25'])
    
    predictions = []
    HAZARDOUS_THRESHOLD = 165
    
    for _, row in holdout_reqs.iterrows():
        sensor = row['sensor_id']
        date = row['target_date']
        
        sensor_history = df[(df['sensor_id'] == sensor) & (df['local_date'] < date)]
        if sensor_history.empty:
            continue
            
        sensor_data = sensor_history.iloc[-1]
        X_pred = pd.DataFrame([[sensor_data['pm25_lag1'], sensor_data['pm25_rolling3']]], columns=['pm25_lag1', 'pm25_rolling3'])
        pred_pm25 = model.predict(X_pred)[0]        
        predictions.append({
            'sensor_id': sensor,
            'target_date': date.strftime('%Y-%m-%d'),
            'predicted_pm25': round(pred_pm25, 2),
            'hazardous': int(pred_pm25 >= HAZARDOUS_THRESHOLD) # SPEC: int 0 or 1, named 'hazardous'
        })
            
    preds_df = pd.DataFrame(predictions)
    preds_df.to_csv('predictions.csv', index=False)
    print(f"predictions.csv generated successfully with {len(preds_df)} rows.")

if __name__ == "__main__":
    print("--- Starting Sprint 1 Pipeline ---")
    df = load_and_standardize()
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    train_and_predict(df_features)