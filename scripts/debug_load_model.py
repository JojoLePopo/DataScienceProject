import joblib, traceback
from pathlib import Path
p = Path('models/best_model.pkl')
print('Trying to load', p.resolve())
try:
    obj = joblib.load(p)
    print('Loaded object type:', type(obj))
except Exception:
    print('--- Exception while loading model ---')
    traceback.print_exc()
    import sys
    sys.exit(2)
print('Done')
