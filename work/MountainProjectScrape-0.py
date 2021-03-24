#!/usr/bin/env python

get_ipython().magic('reset -f')

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pathlib import Path

import os
import re
import requests
import urllib

url = 'https://www.mountainproject.com/area/105798167/the-gunks'
soup = BeautifulSoup(requests.get(url).content, 'html.parser')

chromedriver = '/Users/AntoliMac01/Anaconda/anaconda/selenium/chromedriver'
os.environ["webdriver.chrome.driver"] = chromedriver

dataDir = './dataLarge/'
# dataDir = './dataSmall/'

routeNameRe = re.compile('([^/?]+)(?=/?(?:$|\?))')
imageNameRe = re.compile('[^/]+(?=/$|$)')

lef_nav_rows = soup.find_all('div', {'class' : 'lef-nav-row'})

for div in lef_nav_rows:
    cliffURL = div.find('a', href=True)['href']
    
    cliffName = routeNameRe.findall(cliffURL)[0]
    cliffDir = dataDir + cliffName + '/'  
    
    if os.path.isfile(cliffDir + '.done'):
        print(cliffName + ' already processed.  Skipping...')
        continue

    if not os.path.exists(cliffDir):
        os.makedirs(cliffDir)

    print('cliffURL = ' + cliffURL)
          
    cliffSoup = BeautifulSoup(requests.get(cliffURL).content, 'html.parser')
    lef_nav_rows = cliffSoup.find_all('div', {'class' : 'lef-nav-row'})

    for div in lef_nav_rows:
        areaURL = div.find('a', href=True)['href']
        print('  areaURL = ' + areaURL)

        areaSoup = BeautifulSoup(requests.get(areaURL).content, 'html.parser')

        for a in areaSoup.find_all('a', href=True):
            if a['href'].find('route/') > 0:
                routeURL = a['href']
                routeName = routeNameRe.findall(routeURL)[0]
                routeDir = cliffDir + routeName + '/'  
                
                if os.path.isfile(routeDir + '.done'):
                    print('    ' + routeName + ' already processed. Skipping...')
                    continue

                if not os.path.exists(routeDir):
                    os.makedirs(routeDir)
                
                print('    routeURL = ' + routeURL)

                #... Keep going if there's no 'More Photos' button
                try:  
                    driver = webdriver.Chrome(chromedriver)
                    driver.get(routeURL)
                    more_photos_button = driver.find_element_by_id('more-photos-button')
                    more_photos_button.send_keys(Keys.ENTER)
                except:
                    pass

                routeSoup = BeautifulSoup(requests.get(routeURL).content, 'html.parser')

                driver.close()

                for link in routeSoup.find_all('img'):
                    imageURL = link.get("data-original")

                    if imageURL != None and imageURL.rfind('cdn-files') > 0:
                        if re.search('dataLarge', dataDir):
                            imageURL = imageURL.replace('smallMed', 'medium')

                        imageName = routeDir + imageNameRe.findall(imageURL)[0]
                            
                        req = requests.get(imageURL)
                        with open(imageName , "wb") as fb:
                            fb.write(req.content)
                            
                #... Touch a file so we don't reprocess
                Path(routeDir + '.done').touch()
    
    #... Touch a file so we don't reprocess
    Path(cliffDir + '.done').touch()
    
print('All done...')

