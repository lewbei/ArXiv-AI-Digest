import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime, timedelta
import time

def download_arxiv_papers_api(query, max_total_results=50, download_dir="arxiv_papers", year=None):
    if year is None:
        print("Error: 'year' parameter is required for date-filtered downloads.")
        return

    os.makedirs(download_dir, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0
    no_pdf_link_count = 0
    failed_download_count = 0
    
    paper_metadata = {}
    metadata_file = os.path.join(download_dir, "paper_metadata.json")

    base_url = "http://export.arxiv.org/api/query?"
    sort_by = "sortBy=submittedDate"
    sort_order = "sortOrder=descending"
    results_per_page = 100 # Max results per request for arXiv API
    
    try:
        # Iterate month by month for the specified year
        for month in range(1, 13):
            start_of_month = datetime(year, month, 1, 0, 0, 0)
            
            # Calculate end of month
            if month == 12:
                end_of_month = datetime(year + 1, 1, 1, 0, 0, 0) - timedelta(seconds=1)
            else:
                end_of_month = datetime(year, month + 1, 1, 0, 0, 0) - timedelta(seconds=1)
            
            # Ensure end_of_month does not exceed current date if year is current year
            if year == datetime.now().year and end_of_month > datetime.now():
                end_of_month = datetime.now()

            start_date_str = start_of_month.strftime("%Y%m%d%H%M%S")
            end_date_str = end_of_month.strftime("%Y%m%d%H%M%S")
            
            # Construct the search query with date filter for the current month
            search_query = f"all:{query.replace(' ', '+')}+AND+submittedDate:[{start_date_str}+TO+{end_date_str}]"

            print(f"\n--- Fetching papers for {start_of_month.strftime('%Y-%m')} ---")
            
            start_index = 0
            month_total_entries_found_api = 0

            while True:
                api_url = f"{base_url}search_query={search_query}&{sort_by}&{sort_order}&start={start_index}&max_results={results_per_page}"
                print(f"Fetching data from arXiv API: {api_url}")

                try:
                    response = requests.get(api_url)
                    print(f"API Response Status Code: {response.status_code}")
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    soup = BeautifulSoup(response.content, 'xml') # Parse as XML

                    entries = soup.find_all('entry')
                    if not entries:
                        print("No more entries found for this month.")
                        break # No more results for this month
                    
                    total_results_tag = soup.find('opensearch:totalResults')
                    if total_results_tag:
                        month_total_entries_found_api = int(total_results_tag.text)
                        print(f"API reported {month_total_entries_found_api} total results for this month.")
                    else:
                        print("Warning: Could not find opensearch:totalResults tag in API response.")
                        month_total_entries_found_api = 0

                    print(f"Found {len(entries)} entries in this page.")

                    for entry in entries:
                        if downloaded_count + skipped_count >= max_total_results:
                            print(f"Reached maximum desired papers ({max_total_results}). Stopping.")
                            break # Stop all downloads if max_total_results is reached

                        title = entry.find('title').text.strip()
                        published = entry.find('published').text.strip()
                        pdf_link_tag = entry.find('link', title='pdf')

                        if pdf_link_tag and 'href' in pdf_link_tag.attrs:
                            pdf_url = pdf_link_tag['href']
                            paper_id = pdf_url.split('/')[-1].replace('.pdf', '')
                            file_name = f"{paper_id}.pdf"
                            file_path = os.path.join(download_dir, file_name)
                            
                            paper_metadata[paper_id] = {
                                "title": title,
                                "published": published
                            }

                            if os.path.exists(file_path):
                                print(f"File {file_name} already exists. Skipping download.")
                                skipped_count += 1
                            else:
                                try:
                                    print(f"Downloading {title} ({file_name})...")
                                    pdf_response = requests.get(pdf_url, stream=True)
                                    pdf_response.raise_for_status()

                                    with open(file_path, 'wb') as f:
                                        for chunk in pdf_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    print(f"Successfully downloaded {file_name}")
                                    downloaded_count += 1
                                except requests.exceptions.RequestException as e:
                                    print(f"Error downloading {pdf_url}: {e}")
                                    failed_download_count += 1
                                except Exception as e:
                                    print(f"An unexpected error occurred during download: {e}")
                                    failed_download_count += 1
                        else:
                            print(f"No PDF link found for entry: {title}")
                            no_pdf_link_count += 1
                    
                    start_index += results_per_page
                    # Be polite to the API
                    time.sleep(5) 
                
                    if downloaded_count + skipped_count >= max_total_results:
                        break # Break from inner while loop if max_total_results is reached

                    # If we fetched fewer entries than results_per_page, it means we've reached the end for this month
                    if len(entries) < results_per_page:
                        break

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching from arXiv API: {e}")
                    break # Stop if there's an API error
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    break # Stop on unexpected errors
            
            if downloaded_count + skipped_count >= max_total_results:
                break # Break from outer for loop if max_total_results is reached

    finally:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(paper_metadata, f, indent=4)
        print(f"Saved paper metadata to {metadata_file}")

        print(f"\n--- Download Summary ---")
        print(f"Total entries found by API (across all months): {month_total_entries_found_api}") # This will show the last month's total
        print(f"Successfully downloaded: {downloaded_count}")
        print(f"Skipped (already exists): {skipped_count}")
        print(f"No PDF link found: {no_pdf_link_count}")
        print(f"Failed to download: {failed_download_count}")
        print(f"------------------------")

if __name__ == "__main__":
    search_term = "Attention Is All You Need cs.AI"
    download_arxiv_papers_api(search_term, max_total_results=10000, year=2020)
