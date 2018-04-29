Mainly two things need to be set up.

1. Change the path in DeepTools/deep_config to reflect the path to the penv environment correctly (for the runtime)
2. Re-run DeepBuild/deep_config.sh from the module folder, to create penv (which supports torch) inside ./PlanteomeDeepSegment/penv/
3. The staging path may also change (in runtime-module.cfg)
4. The path to the model in ./DeepTools/deep_python.py needs to reflect the proper path to the model (line 122)
