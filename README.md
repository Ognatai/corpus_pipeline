# corpus_pipeline
complimentary repo for "taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades"

## Contents:  
``code\``: pieline code  
``results\``: pipline results including figures and reports   
``setup\``: packages needed for pipeline setup  

## Additional Setup Information  
The pipeline uses python 10.1.  
Make sure to use spacy==3.5.0 and to not unintentionally upgrade it while installing other packages. Otherwise, coreferee won't be compatible with the language model "de-core-news-lg".  
Use requirements.txt for pipeline setup and download "de-core-news-lg" and install "coreferee de" afterwards.
