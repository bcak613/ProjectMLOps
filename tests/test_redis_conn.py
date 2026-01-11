import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('foo', 'bar')
    value = r.get('foo')
    print(f"Successfully connected to Redis. Value for 'foo': {value}")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
