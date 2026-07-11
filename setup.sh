pip install -r requirement.txt

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RUPTURES=0.0.0 pip install -e .

cd multi_scripts
git clone https://github.com/nlgandnlu/SegFormer.git
cd ..
