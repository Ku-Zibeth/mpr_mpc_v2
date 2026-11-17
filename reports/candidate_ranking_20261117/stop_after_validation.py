import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent
while not (R/'validation_collection_status.json').exists() or len(json.loads((R/'validation_collection_status.json').read_text()))<48:time.sleep(3)
print('Prefixed validation collection complete; spare workers collect inner-stop maps without reading sealed outcomes.',flush=True)
with (R/'stop_collection.log').open('x') as log:subprocess.run([sys.executable,'-u',str(R/'collect_queue.py'),'--stop-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
