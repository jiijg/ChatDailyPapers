import os
import re
import argparse
import feedparser
import urllib.parse
import pytz
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

        self.filter_keys = (
            filter_keys if isinstance(filter_keys, list)
            else str(filter_keys).split()
        )

        self.chat_api_list = OPENAI_API_KEYS
        self.cur_api = 0

    # =========================
    # ARXIV STABLE VERSION
    # =========================
    def get_arxiv(self, max_results=20):

        # ❗核心修复：避免 AND / 避免复杂 query
        query_raw = self.query

        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={query_raw}"
            f"&start=0&max_results={max_results}"
            "&sortBy=lastUpdatedDate"
            "&sortOrder=descending"
        )

        print("\n[ARXIV QUERY]")
        print(query_raw)

        feed = feedparser.parse(url)

        print("[ARXIV ENTRIES]")
        print(len(feed.entries))

        return feed.entries

    # =========================
    # FILTER (relaxed OR logic)
    # =========================
    def filter_arxiv(self, max_results=20):

        entries = self.get_arxiv(max_results)

        results = []

        print("\n[FILTER KEYS]", self.filter_keys)

        for e in entries:

            text = (e.summary or "").lower()

            # ✔ OR match（避免 0 结果）
            if any(k.lower() in text for k in self.filter_keys):
                results.append(e)

        print("[FILTER RESULT]", len(results))

        return results

    # =========================
    # DOWNLOAD PDF
    # =========================
    def download_pdf(self, results):

        os.makedirs("./pdf_files", exist_ok=True)

        papers = []

        for r in results:
            try:
                title = re.sub(r"[\/\\\:\*\?\"\<\>\|]", "_", r.title)
                path = "./pdf_files"

                r.download_pdf(path, filename=title + ".pdf")

                paper = Paper(
                    path=os.path.join(path, title + ".pdf"),
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
    # SUMMARY (simple safe version)
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
    # CHAT (safe fallback)
    # =========================
    def chat_summary(self, text):

        try:
            import openai
            openai.api_key = self.chat_api_list[self.cur_api]
            self.cur_api = (self.cur_api + 1) % len(self.chat_api_list)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize paper clearly"},
                    {"role": "user", "content": text[:2500]}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Summary error: {str(e)}"


# =========================
# MAIN
# =========================
def main(args):

    for k in args.filter_keys:

        # ✔ 改成稳定 query（避免 AND）
        query = "all:" + k.replace(" ", "+")

        reader = Reader(
            key_word=k,
            query=query,
            filter_keys=args.filter_keys,
            args=args
        )

        entries = reader.get_arxiv()

        results = reader.filter_arxiv(args.max_results)

        papers = reader.download_pdf(results)

        htmls = reader.summary_with_chat(papers)

        # 🔥 保底机制（防止 0 issue）
        if not htmls:
            htmls = [f"No papers found for: {k}"]

        make_github_issue(
            title=f"Daily Papers - {k}",
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
