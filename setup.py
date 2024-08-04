from setuptools import find_packages,setup
from typing import List

edot_str_rem = '-e .' # string will be used to make sure not included as required library
'''
NOTE: use above string in requirements.txt file to link to setup.py, such that
the setup code is built when installing requirements. However, we do not want to 
load the string as a package so need to delete in below function.
'''
def get_requirements(file_name:str)->List[str]:
    '''
    This function will return the list of requirments from the
    specified requirements file.
    '''
    reqs = [] #initialize list
    with open(file_name) as file:
        reqs = file.readlines()
        reqs = [req.replace("\n","") for req in reqs]
        
    if edot_str_rem in reqs:
        reqs.remove(edot_str_rem)

    return reqs

setup(
    name='LunarRover',
    version='0.0.1',
    author='TaylorC',
    author_email='taylor2@alumni.stanford.edu',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)