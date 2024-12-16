# corpus_pipeline
complimentory repo for "taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades"

## Contents:  
``code\``: pieline code  
``results\``: pipline results including figures and report for 1993 and 2023  
``setup\``: packages needed for pipeline setup  

## Additional Setup Information  
The pipeline uses python 10.1.  
Make sure to use spacy==3.5.0 and to not uniententionally upgrade it while installing other packages. Otherwise coreferee won't be compatibel with the language model "de-core-news-lg".  
Use requirements.txt for pipeline setup and download "de-core-news-lg" and install "coreferee de" afterwards.
