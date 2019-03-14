# Planteome Deep Segmenter
**Segmentation and classification module built for the CyVerse BisQue image analysis environment.**

This software is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license.

Bisque configuration instructions for a dockerized BisQue instance:

1. In a terminal type : docker run -p 8080:8080 cbiucsb/bisque05:stable
2. Open a new terminal and type : docker ps -a
3. Note the above container name under NAMES
4. docker exec -it <container_name> /bin/bash
5. (optional) you may 'stop' and then 'start' the container at any time using docker commands (docker stop and docker start commands)
6. bisque is under /source, and virtualenv is activate with source /usr/lib/bisque/bin/activate (for commands bq-admin server stop/start/restart)
7. Stop the container, start it again, 'exec' into it to make sure you can use bash (docker exec -it <container_name> /bin/bash)
8. Go to 127.0.0.1:8080 and join as admin (password: admin)
9. Under admin/module manager (top right corner), enter engine url :localhost:8080/engine_service in the field of the right panel
10. Find the git cloned module PlanteomeDeepSegment (should be in source/modules/Pla..) and register it

Module configuration instructions for dockerized BisQue hosts:

1. Run this to be able to install pytorch: sudo apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
2. Change the path in DeepTools/deep_config to reflect the path to the penv environment correctly
3. Re-run DeepBuild/deep_config.sh from the module folder, to create penv (which supports torch) at ./PlanteomeDeepSegment/penv/
4. The staging path may also change (in runtime-module.cfg)
5. The path to the models in ./DeepTools/deep_python.py needs to reflect the proper path for the installation (one path per model, three included so far)

