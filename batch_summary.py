"""Batch summary: reuses original library, only adds get_papers_by_date"""
import arxiv
import argparse
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from tqdm import tqdm
from loguru import logger

from main import get_zotero_corpus, filter_corpus
from recommender import rerank_paper
from construct_email import render_email, send_email
from paper import ArxivPaper
from llm import set_global_llm


def get_papers_by_date(query: str, date: str) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    cats = [f"cat:{c.strip()}" for c in query.split('+')]
    search_query = f"({' OR '.join(cats)}) AND submittedDate:[{date.replace('-','')}0000 TO {date.replace('-','')}2359]"
    search = arxiv.Search(query=search_query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=200)
    return [ArxivPaper(p) for p in tqdm(client.results(search), desc=f"Fetching {date}")]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Get config from environment
    zotero_id = os.environ['ZOTERO_ID']
    zotero_key = os.environ['ZOTERO_KEY']
    zotero_ignore = os.environ.get('ZOTERO_IGNORE', '')
    arxiv_query = os.environ['ARXIV_QUERY']
    smtp_server = os.environ['SMTP_SERVER']
    smtp_port = int(os.environ['SMTP_PORT'])
    sender = os.environ['SENDER']
    receiver = os.environ['RECEIVER']
    sender_password = os.environ['SENDER_PASSWORD']
    max_paper_num = int(os.environ.get('MAX_PAPER_NUM', 100))
    use_llm = os.environ.get('USE_LLM_API', '').lower() == 'true'
    language = os.environ.get('LANGUAGE', 'English')

    logger.info("Loading Zotero corpus...")
    corpus = get_zotero_corpus(zotero_id, zotero_key)
    if zotero_ignore:
        corpus = filter_corpus(corpus, zotero_ignore)
    logger.info(f"Loaded {len(corpus)} papers from Zotero")

    if use_llm:
        set_global_llm(
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_API_BASE'),
            model=os.environ.get('MODEL_NAME'),
            lang=language
        )
    else:
        set_global_llm(lang=language)

    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    current = start

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        logger.info(f"Processing {date_str}...")
        
        papers = get_papers_by_date(arxiv_query, date_str)
        if papers:
            papers = rerank_paper(papers, corpus)[:max_paper_num]
            html = render_email(papers)
            # Patch datetime in construct_email module to use custom date
            mock_dt = MagicMock()
            mock_dt.now.return_value.strftime.return_value = date_str.replace('-', '/')
            with patch('construct_email.datetime', mock_dt):
                send_email(sender, receiver, sender_password, smtp_server, smtp_port, html)
            logger.success(f"Email sent for {date_str}")
        else:
            logger.info(f"No papers found for {date_str}")
        
        current += timedelta(days=1)
