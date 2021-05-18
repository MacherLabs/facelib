from setuptools import setup

setup(
    name='face',

    version='1.1.0',
    description='Face detection and recognition',
    url='http://demo.vedalabs.in/',

    # Author details
    author='Atinderpal Singh',
    author_email='atinderpalap@gmail.com',

    license='Commercial',

    packages=['face'],
    package_data={
        'face': ['models/*'],
    },

    install_requires=['numpy','common @ git+https://github.com/MacherLabs/common.git'],
    zip_safe=False
    )
