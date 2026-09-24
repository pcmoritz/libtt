import collections, gzip, json, pathlib, sys
for root in sys.argv[1:]:
    result=[]
    for path in pathlib.Path(root).rglob('*.trace.json.gz'):
        with gzip.open(path,'rt') as f: trace=json.load(f)
        agg=collections.defaultdict(list)
        for e in trace.get('traceEvents',[]):
            if e.get('ph')=='X' and e.get('dur',0)>0:
                agg[e.get('name','')].append(e['dur'])
        rows=[{'name':k,'calls':len(v),'inclusive_total_ms':sum(v)/1000,'mean_ms':sum(v)/len(v)/1000,'max_ms':max(v)/1000} for k,v in agg.items()]
        rows.sort(key=lambda r:r['inclusive_total_ms'],reverse=True)
        result.append({'trace':str(path),'note':'Inclusive host/Python durations overlap; do not sum across names. Device kernel times are unavailable.','top_events':rows[:60]})
    p=pathlib.Path(root)/'trace-summary.json'; p.write_text(json.dumps(result,indent=2)); print(p)
