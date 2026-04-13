import numpy as np
import os
import re
import datetime
import base64, requests
import argparse
import tiktoken
import feedparser
import urllib.parse
import pytz

from datetime import datetime, timedelta

from get_paper_from_pdf import Paper
from github_issue import make_github_issue
from config import OPENAI_API_KEYS, KEYWORD_LIST, LANGUAGE


# =========================
# time window
# =========================
now = datetime.now(pytz.utc)
yesterday = now - timedelta(days=1.1)


# =========================
# Reader Class
# =========================
class Reader:

    def __init__(self, filter_keys, filter_times_span=(yesterday, now),
                 key_word=None, query=None, root_path='./',
                 sort=None, user_name='default', args=None):

        self.user_name = user_name
        self.key_word = key_word
        self.query = query
        self.sort = sort
        self.args = args

        self.language = getattr(args, "language", "zh")
        if self.language == 'en':
            self.language = 'English'
        else:
            self.language = 'Chinese'

        # filter keys FIX: support string or list
        if isinstance(filter_keys, list):
            self.filter_keys = filter_keys
        else:
            self.filter_keys = str(filter_keys).split()

        self.filter_times_span = filter_times_span
        self.root_path = root_path

        self.chat_api_list = OPENAI_API_KEYS
        self.cur_api = 0

        self.file_format = getattr(args, "file_format", "md")
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    # =========================
    # FIX: replace arxiv SDK (301 issue)
    # =========================
    def get_arxiv(self, max_results=30):
        query_encoded = urllib.parse.quote(self.query)

        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={query_encoded}"
            f"&start=0&max_results={max_results}"
            "&sortBy=lastUpdatedDate"
            "&sortOrder=descending"
        )

        feed = feedparser.parse(url)
        return feed.entries

    def filter_arxiv(self, max_results=30):

        entries = self.get_arxiv(max_results)

        print("all search:")
        for i, e in enumerate(entries):
            print(i, e.title)

        filter_results = []

        print("filter_keys:", self.filter_keys)

        for e in entries:

            abs_text = (e.summary or "").replace('\n', ' ')

            meet_num = 0
            for k in self.filter_keys:
                if k.lower() in abs_text.lower():
                    meet_num += 1

            if meet_num == len(self.filter_keys):
                filter_results.append(e)

        print("筛选后论文数:", len(filter_results))
        return filter_results

    # =========================
    # download pdf (keep original logic)
    # =========================
    def download_pdf(self, filter_results):

        date_str = str(datetime.now())[:13].replace(' ', '-')
        path = self.root_path + 'pdf_files/' + self.query.replace(':', ' ')[:25] + '-' + date_str

        os.makedirs(path, exist_ok=True)

        paper_list = []

        for r in filter_results:
            try:
                title = self.validateTitle(r.title)
                pdf_name = title + ".pdf"

                self.try_download_pdf(r, path, pdf_name)

                paper_path = os.path.join(path, pdf_name)

                paper = Paper(
                    path=paper_path,
                    url=r.link,
                    title=r.title,
                    abs=r.summary.replace('\n', ' '),
                    authers=[]
                )

                paper.parse_pdf()
                paper_list.append(paper)

            except Exception as e:
                print("download_error:", e)

        return paper_list

    def validateTitle(self, title):
        return re.sub(r"[\/\\\:\*\?\"\<\>\|]", "_", title)

    # =========================
    def try_download_pdf(self, result, path, pdf_name):
        result.download_pdf(path, filename=pdf_name)

    # =========================
    def summary_with_chat(self, paper_list, htmls=None):
        if htmls is None:
            htmls = []

        for paper in paper_list:

            text = f"Title:{paper.title}\nAbstract:{paper.abs}"

            summary = self.chat_summary(text)

            htmls.append(f"## {paper.title}")
            htmls.append(summary)

        return htmls

    # =========================
    def chat_summary(self, text, summary_prompt_token=1100):
        openai_key = self.chat_api_list[self.cur_api]
        self.cur_api = (self.cur_api + 1) % len(self.chat_api_list)

        import openai
        openai.api_key = openai_key

        import openai

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize paper"},
                {"role": "user", "content": text[:3000]}
            ]
        )

        return response.choices[0].message.content

    # =========================
    def show_info(self):
        print("Key word:", self.key_word)
        print("Query:", self.query)


# =========================
# main
# =========================
def main(args):

    filter_times_span = (now - timedelta(days=args.filter_times_span), now)

    for filter_key in args.filter_keys:

        query = " AND ".join([f"all:{x}" for x in filter_key.split()])

        reader = Reader(
            key_word=filter_key,
            query=query,
            filter_keys=filter_key,
            filter_times_span=filter_times_span,
            args=args
        )

        reader.show_info()

        results = reader.filter_arxiv(args.max_results)

        papers = reader.download_pdf(results)

        htmls = reader.summary_with_chat(papers)

        make_github_issue(
            title=filter_key,
            body="\n".join(htmls),
            labels=args.filter_keys
        )


# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, default="all:machine AND all:learning")
    parser.add_argument("--key_word", type=str, default="ML")
    parser.add_argument("--filter_keys", type=list, default=KEYWORD_LIST)
    parser.add_argument("--filter_times_span", type=float, default=1.1)
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument("--file_format", type=str, default="md")
    parser.add_argument("--language", type=str, default=LANGUAGE)

    args = parser.parse_args()

    main(args)
    
