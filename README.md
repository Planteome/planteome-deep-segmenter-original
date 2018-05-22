# Planteome Deep Segmenter
** Segmentation and classification module built for the CyVerse BisQue image analysis environment. **

This software is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license.

Configuration instructions for BisQue hosts:
1. Change the path in DeepTools/deep_config to reflect the path to the penv environment correctly (for the runtime)
2. Re-run DeepBuild/deep_config.sh from the module folder, to create penv (which supports torch) inside ./PlanteomeDeepSegment/penv/
3. The staging path may also change (in runtime-module.cfg)
4. The path to the model in ./DeepTools/deep_python.py needs to reflect the proper path to the model (line 122)
