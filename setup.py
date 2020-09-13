import setuptools

setuptools.setup(
    name='erlyx',
    version='0.0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.18.5',
        'pandas==1.0.3',
        'torch==1.5.1',
        'tqdm==4.46.0',
        'h5py==2.10.0',
        'gym[atari]==0.10.0',
        'fire==0.3.1'
    ],
    python_requires='>=3.7',
    zip_safe=False)
