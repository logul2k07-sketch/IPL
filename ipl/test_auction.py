from app import app, BATTERS
import time

c = app.test_client()
player = BATTERS[0]
print('Testing player:', player)
start = time.time()
r = c.post('/auction/profile', json={'name': player})
end = time.time()
print('Status:', r.status_code, 'Time:', round(end-start,3))
try:
    print('JSON keys:', list(r.get_json().keys()))
except Exception as e:
    print('Failed to get JSON:', e)
