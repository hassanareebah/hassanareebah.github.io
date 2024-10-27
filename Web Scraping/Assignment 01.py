"""
This script scrapes reviews from Trustpilot for Skype, the data is processed
and saved in a CSV file. 

BeautifulSoup is used to extract data including the company name, date of
publication, rating value, and review text from multiple pages of reviews.

Reviews are then cleaned, keeping only the date and numeric rating value,
and stored in a pandas DataFrame, which is finally exported to a CSV file.

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Get the initial response
resp = requests.get('https://ca.trustpilot.com/review/www.skype.com')

# Parse the HTML content
soup = BeautifulSoup(resp.text, 'html.parser')
print(soup)

# Total number of reviews
item = soup.find(
    name='p',
    attrs={
        'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17'
    }
)
N = int(item.contents[0].replace(',', ''))

# Creating a dataframe to store extracted reviews
df = pd.DataFrame(columns=[
    'companyName', 'datePublished', 'ratingValue', 'reviewBody'
])
rows = []

# Base URL for pagination
base_url = (
    "https://ca.trustpilot.com/review/www.skype.com?languages=all&page={}"
    "&sort=recency"
)

page_number = 1
total_reviews = 0
max_pages = 100

# Loop to go through multiple pages
while True:
    # Constructing URL for the current page
    page_url = base_url.format(page_number)
    print(f"Processing page {page_number}: {page_url}...")

    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    reviews = soup.find_all('div', class_='styles_reviewCardInner__EwDq2')

    if not reviews:
        print("No more reviews found or page structure might have changed.")
        break

    print(f"Found {len(reviews)} reviews on page {page_number}.")
    total_reviews += len(reviews)

    # Iterate through reviews
    for review in reviews:
        company_name = "Skype"

        # Extract date from the 'datetime' attribute of the 'time' element
        date_published_element = review.find('time')
        date_published = (
            date_published_element['datetime']
            if date_published_element else 'N/A'
        )

        # Extract rating from the 'alt' attribute of the image
        rating_element = review.find('img', alt=True)
        if rating_element:
            rating_value = rating_element['alt']
            if rating_value.startswith("Rated"):
                rating_value = rating_value.split(' ')[1]  # Extract only number
            else:
                rating_value = 'N/A'
        else:
            print("Rating element not found")
            rating_value = 'N/A'

        # Extract review body from 'p' tag
        review_body_element = review.find(
            'p',
            class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 '
                   'typography_color-black__5LYEn'
        )
        review_body = (
            review_body_element.text.strip() if review_body_element else 'N/A'
        )

        # Appending each row to the list of rows
        rows.append({
            'companyName': company_name,
            'datePublished': date_published,
            'ratingValue': rating_value,
            'reviewBody': review_body
        })

    # Move to the next page
    page_number += 1

    # Prevent infinite loop if max pages limit is reached
    if page_number > max_pages:
        print("Reached maximum page limit.")
        break

# Convert rows to DataFrame
df = pd.DataFrame(rows)

# Keep only the date from the 'datePublished' value
df['datePublished'] = df['datePublished'].apply(
    lambda x: x.split('T')[0].strip() if x != 'N/A' else x
)

# Keep only the numeric rating value
df['ratingValue'] = df['ratingValue'].apply(
    lambda x: x.split(' ')[1].strip() if x != 'N/A' and len(x.split(' ')) > 1 else x
)

# Save the DataFrame to a CSV file
df.to_csv(
    'C:/Users/ali_s/Desktop/Areebah/Schulich/Term 2/NLP/Skype_Reviews.csv',
    index=False
)
