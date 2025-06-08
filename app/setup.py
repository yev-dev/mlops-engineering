from setuptools import setup, find_packages

long_description = """Experimental package to explore financila analysis"""

setup(name='mlops',
      version='0.1',
      description='ML Ops library',
      author='Yevgeniy Yermoshin',
      author_email='yev.developer@gmail.com',
      license='Apache 2.0',
      long_description=long_description,
      keywords=['pandas', 'data', 'analysis', 'fixed income', 'stocks', 'bond', 'equities', 'timeseries', 'quantative', 'strategies', 'backtesting'],
      url='',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=[
          'fastapi',
          'uvicorn',
          'pandas',
          'psutil',
          'pydantic',
          'prometheus_fastapi_instrumentator',
          'scikit-learn',
          'joblib'
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      python_requires='>=3.9',
      package_data={
          '': ['*.json', '*.yaml', '*.yml', '*.txt'],
          'mlops': ['data/*.csv', 'models/*.joblib']
      },
      include_package_data=True,
      zip_safe=False)
