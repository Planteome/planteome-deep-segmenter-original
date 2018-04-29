rm -rf ./penv
virtualenv ./penv
source ./penv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
pip install torchvision==0.2.1
pip install numpy==1.14.3
pip install pymaxflow==1.2.8
pip install opencv-python==3.4.0.12
pip install scikit-image==0.13.1

deactivate
