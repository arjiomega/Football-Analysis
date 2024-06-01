from setuptools import find_packages, setup

setup(
    name="src",
    python_requires=">=3.11.0",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "ultralytics==8.2.22",
        "roboflow==1.1.30",
        "supervision==0.20.0",
        "opencv-python==4.9.0.80",
        "scikit-learn==1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest==8.2.0",
            "black==24.4.2",
            "pre-commit==3.7.0",
            "notebook==7.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "run-app = src",
        ]
    },
)
