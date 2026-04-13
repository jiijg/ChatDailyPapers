import numpy as np
import os
import re
import base64
import requests
import argparse
import tiktoken
import feedparser
import pytz
import urllib.parse

from datetime import datetime, timedelta

from get_paper_from_pdf import Paper
from github_issue import make_github_issue
from config import OPENAI_API_KEYS, KEYWORD_LIST, LANGUAGE


# =========================
# time
# =========================
now = datetime.now(pytz.utc)


# =========================
# Reader
# =========================
class Reader:

    def __init__(self, filter_keys, key_word=None, query=None, args=None):

        self.key_word = key_word
        self.query = query
        self.args = args

        self.language = getattr(args, "language", "zh")

        if isinstance(filter_keys, list):
            self.filter_keys = filter_keys
        else:
            self.filter_keys = str(filter_keys).split()

        self.chat_api_list = OPENAI_API_KEYS
        self.cur_api = 0

        self.encoding = tiktoken.get_encoding("gpt2")

    # =========================
    # FIXED ARXIV (stable)
    # =========================
    def get_arxiv(self, max_results=20):

        # ❗关键修复：不要 encode（导致 CI 空结果）
        query_raw = self.query

        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={query_raw}"
            f"&start=0&max_results={max_results}"
            "&sortBy=lastUpdatedDate"
            "&sortOrder=descending"
        )

        print("\n[ARXIV URL]")
        print(url)

        feed = feedparser.parse(url)

        print("[ARXIV RESULTS]")
        print("entries:", len(feed.entries))

        return feed.entries

    # =========================
    def filter_arxiv(self, max_results=20):

        entries = self.get_arxiv(max_results)

        results = []

        print("\n[FILTER KEYS]", self.filter_keys)

        for e in entries:

            text = (e.summary or "").lower()

            # 🔥 改成 OR 逻辑（否则永远为空）
            if any(k.lower() in text for k in self.filter_keys):
                results.append(e)

        print("[FILTERED RESULTS]", len(results))

        return results

    # =========================
    def download_pdf(self, results):

        path = "./pdf_files"
        os.makedirs(path, exist_ok=True)

        papers = []

        for r in results:
            try:
                title = re.sub(r"[\/\\\:\*\?\"\<\>\|]", "_", r.title)
                pdf_path = os.path.join(path, title + ".pdf")

                r.download_pdf(path, filename=title + ".pdf")

                paper = Paper(
                    path=pdf_path,
                    url=r.link,
                    title=r.title,
                    abs=r.summary,
                    authers=[]
                )

                paper.parse_pdf()
                papers.append(paper)

            except Exception as e:
                print("download error:", e)

        return papers

    # =========================
    def summary_with_chat(self, papers):

        htmls = []

        for p in papers:

            text = f"{p.title}\n{p.abs}"

            summary = self.chat_summary(text)

            htmls.append("## " + p.title)
            htmls.append(summary)

        return htmls

    # =========================
    def chat_summary(self, text):

        import openai
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api = (self.cur_api + 1) % len(self.chat_api_list)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the paper"},
                {"role": "user", "content": text[:3000]}
            ]
        )

        return response.choices[0].message.content


# =========================
# main
# =========================
def main(args):

    for k in args.filter_keys:

        query = "all:" + k.replace(" ", "+")

        reader = Reader(
            key_word=k,
            query=query,
            filter_keys=args.filter_keys,
            args=args
        )

        reader.get_arxiv()

        results = reader.filter_arxiv(args.max_results)

        papers = reader.download_pdf(results)

        htmls = reader.summary_with_chat(papers)

        # 如果没结果，也强制写issue（防止“空运行”）
        if not htmls:
            htmls = [f"No papers found for {k}"]

        make_github_issue(
            title=k,
            body="\n".join(htmls),
            labels=args.filter_keys
        )


# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--filter_keys", type=list, default=KEYWORD_LIST)
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument("--language", type=str, default=LANGUAGE)

    args = parser.parse_args()

    main(args)  
