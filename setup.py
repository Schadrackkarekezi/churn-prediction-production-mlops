from setuptools import find_packages, setup

setup(
    name='churn-prediction',
    version='1.0.0',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'imbalanced-learn',
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
)
