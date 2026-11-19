from study import *
if __name__=='__main__':
    assert (R/'development_complete.json').exists()
    methods=['A','Ttrack','Tnative','Tconditional'];scenes=load(R/'protocol.json')['development_scenes'][:2]
    dump(R/'latency_protocol.json',dict(scenes=scenes,methods=methods,control_state=100,warmup=5,samples=30,scope='fixed real states reconstructed by100 exact A controls; synchronized full agent.act, single collector, no environment step during measurement'))
    for scene in scenes:
        for method in methods:
            with (R/f'latency_{method}_{scene}.log').open('x') as out:
                subprocess.run([sys.executable,'-u',str(R/'latency_probe.py'),'--scene',str(scene),'--method',method],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
            print('LATENCY_DONE',method,scene,flush=True)
