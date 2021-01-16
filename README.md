# NLP---Press-coverage-of-LBGT-issues

Data preparation 

Data was scraped from searchpages of chosen polish portals. Datasets contain articles from 2020. Different notebook was created for scraping each page, as html code 
differed between them. Scrapers are written using Python's BeautifulSoup package. All are avaliable for review in different folder.

Project goals 

Aim of this project was to preprocess scraped raw data and perform basic NLP to obtain material for further analysis. I created WordClouds of most used words in 
coverage from each portal, computed word frequencies and found most common word collocations and collocations with specific LGBT+ related info. This data can be used to
describe how each portal covers these issues and what reaction they are trying to provoke in readers. 

Additionally I created classification model trained on data from most right and left leaning sites that can be used to initially classify how a piece of text adresses
LBGT+ issues.

Main code is in the file "Analiza przekazu prasowego na temat społeczności LGBT"

Additional scripts are provided in file add_functions.py

Scraping codes are provided as Jupyter Notebooks in folder

Data is stored in the data folder

While setting up you should have main code in the same folder as data files
