from setuptools import setup

with open('ReadMe.md', 'r') as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setup(
        name = 'baseballforecaster',
        version = '0.0.1',
        description ='A monte-carlo forecaster and drafter for baseball performance',
        long_description = long_description,
        long_description_content_type = 'text_markdown',
        author = 'Bailey Morton',
        author_email = 'baileymorton989@gmail.com',
        url = 'https://github.com/baileymorton989/baseballforecaster',
        license = 'MIT License',
        install_requires = ['pybaseball >=2.1.1', 'pandas >= 1.1.4',
                            'numpy >=1.18.1','scikit-learn >= 0.23.1',
                            'tqdm >= 4.47.0'],
        py_modules = ['baseballforecaster'],
        package_dir = {'': 'src'},
        classifiers =[
             'Programming Language :: Python :: 3',
             'Programming Language :: Python :: 3.6',
             'Programming Language :: Python :: 3.7',
             'Programming Language :: Python :: 3.8',
             'Programming Language :: Python :: 3.9',
             'License :: OSI Approved :: MIT License',
             'Operating System :: OS Independent',
        ],
        extras_require = {
            'dev': [
                'pytest>3.7',
                ],
            },

    )
