import os, sys, re
os.chdir(r'd:\24ADA27\INTEN\New folder')
sys.stdout.reconfigure(encoding='utf-8')
import app as a
c = a.app.test_client()
r = c.get('/')
html = r.data.decode('utf-8')
js_start = html.rfind('<script>')
js_end = html.rfind('</script>')
js = html[js_start:js_end]
fns = re.findall(r'(?:async )?function (\w+)', js)
print('Functions:', fns)
expected = ['switchTab','post','predictWicket','predictRuns','predictScore','analyzeBowler','analyzeBatter','predictWin','toggleChaseFields','updateCompareForm','runCompare','renderCompare','renderCompareChart','predictMatchup']
missing = [f for f in expected if f not in fns]
print('Missing:', missing if missing else 'NONE - all good!')
