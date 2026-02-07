import pandas as pd
import yfinance as yf
from src.detector import TopologicalAnomalyDetector

event_date = '2020-02-24'
event_dt = pd.to_datetime(event_date)

data = yf.download('SPY', start='2019-01-01', end='2020-03-15', progress=False)
prices = data['Close'].values
dates = data.index

# Use the same defaults as validator
window_size = 50
baseline_period = 10
sensitivity = 2.0

det = TopologicalAnomalyDetector(window_size=window_size, baseline_period=baseline_period, sensitivity=sensitivity)
train_size = 300

print('len(prices)=', len(prices), 'train_size=', train_size)
det.fit(prices[:train_size])
print('adaptive_thresholds:', det.adaptive_thresholds)

max_vals = {'betti_score': 0.0, 'wasserstein': 0.0, 'entropy_delta': 0.0}
max_at = None

for i in range(train_size, len(prices)):
    w = prices[i-window_size:i]
    res = det.detect(w)
    m = res['metrics']
    for k in max_vals:
        if m[k] > max_vals[k]:
            max_vals[k] = float(m[k])
            max_at = (dates[i], int((event_dt - dates[i]).days))

print('max metrics overall:', max_vals, 'at', max_at)

print('\n--- days_before 10..0 snapshot ---')
for i in range(train_size, len(prices)):
    days_before = int((event_dt - dates[i]).days)
    if 0 <= days_before <= 10:
        w = prices[i-window_size:i]
        res = det.detect(w)
        print(dates[i].date(), 'days_before', days_before, 'is_anom', res['is_anomaly'], 'conf', res['confidence'])
        print('  metrics:', res['metrics'])
        print('  signals:', res['signals'])
