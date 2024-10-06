import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL
base_url = 'https://nsdl.co.in/'
data_url = 'https://nsdl.co.in/depository-monthly-statistics.php'

# Directory to save the downloaded files
download_dir = 'downloaded_files'

# Create download directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Function to download files
def download_file(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_name = os.path.join(download_dir, url.split('/')[-1])
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download: {url} with status code: {response.status_code}")

# Scrape the page
response = requests.get(data_url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')
        if href and 'Depository' in link.text and href.endswith('.xlsx'):
            # Convert relative URL to absolute
            absolute_url = urljoin(base_url, href)
            
            # Extract the year from the link text
            for year in range(2019, 2025):
                if str(year) in href:
                    download_file(absolute_url)
                    break
else:
    print(f"Failed to retrieve page with status code: {response.status_code}")
