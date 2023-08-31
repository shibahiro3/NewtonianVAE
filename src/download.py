"""
公式
https://developers.google.com/drive/api/quickstart/python?hl=ja

参考
https://qiita.com/jkawamoto/items/a19361bff33607264c9f

プロジェクト作成
 &
ライブラリ > Google Drive API の使用の有効化
 &
OAuth関連

Get credentials.json: (by ChatGPT)
   1. Google Cloud Consoleにログインし、プロジェクトを作成します。
   2. 「APIとサービス」>「ダッシュボード」に移動し、「APIを有効化」でGoogle Drive APIを有効にします。
   3. 「APIとサービス」>「認証情報」に移動し、「認証情報を作成」を選択します。
   4. 「サービスアカウント」を選択し、必要な情報を入力してサービスアカウントを作成します。キーのタイプはJSON形式を選択してください。すると、JSON形式の認証情報がダウンロードできます。

pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


Alternative:
    gdown, rclone
"""


import argparse
import copy
import io
import multiprocessing as mp
import os
import threading
import time
import urllib.parse
from datetime import datetime
from pprint import pprint
from typing import List, Optional

import numpy as np
import polars as pl
import polars.type_aliases as pltype
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from mypython.pyutil import human_readable_byte, s2dhms_str
from mypython.terminal import Color, Prompt


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--cred-dir", type=str, metavar="PATH", help="Must include credentials.json. Once authentication is complete, token.json is created.", required=True)
    parser.add_argument("--url", type=str, metavar="ID", required=True)
    parser.add_argument("--dst-dir", type=str, metavar="PATH", required=True)
    parser.add_argument('--ext', nargs="*", type=str, help='Ex. npy jpg   if you not specied, all install')
    parser.add_argument("--num-workers", type=int, default=8, help="how many subprocesses to use for data loading")
    parser.add_argument("--mode", type=str, choices=["only-scan", "all-anyway"], default="all-anyway")  # ["only-scan", "all-anyway", "update"]
    args_dict = vars(parser.parse_args())
    args_dict["src_id"] = Download.id_from_url(args_dict.pop("url"))
    print("id from url:", args_dict["src_id"])
    Download.download(**args_dict)
    # fmt: on


class Download:
    """
    cred_root/
        - token.json
        - credentials.json
    """

    @staticmethod
    def id_from_url(url: str):
        """
        urls = [
            "https://drive.google.com/drive/folders/ID?usp=drive_link",
            "https://drive.google.com/file/d/ID",
            "https://docs.google.com/document/d/ID/edit?usp=drive_link",
            "https://drive.google.com/file/d/ID/view?usp=sharing",
            "https://drive.google.com/uc?id=ID",
            "https://drive.google.com/open?id=ID",
            "https://drive.google.com/uc?export=download&id=ID",
            "https://docs.google.com/spreadsheets/d/ID/edit#gid=123456789",
            "https://drive.google.com/drive/u/0/folders/ID",
            "https://drive.google.com/file/d/ID/view?usp=drive_link"
        ]
        for url in urls:
            print(url)
            print(Download.id_from_url(url))
        """

        id = ""
        parsed = urllib.parse.urlparse(url)
        id_ = urllib.parse.parse_qs(parsed.query).get("id", None)
        if id_ is not None:
            # path=/open
            id = id_[0]
        else:
            sp = parsed.path.split("/")
            # path=/drive/folders/
            # path=/file/d/
            # path=/document/d/
            # path=/drive/u/0/folders/
            for i, s in enumerate(sp):
                if s == "folders" or s == "d":
                    id = sp[i + 1]
                    break

        if id == "":
            raise Exception("connot extract id from url")

        # print(id)
        # assert len(id) >= 33
        return id

    @classmethod
    def download(
        cls, src_id: str, dst_dir: str, cred_dir: str, num_workers: int, mode: str, ext: List[str]
    ):
        assert os.path.exists(cred_dir) and os.path.isdir(cred_dir)

        cls.num_workers = num_workers
        finfo_dl = cls._scan(src_id=src_id, dst_dir=dst_dir, cred_dir=cred_dir, mode=mode, ext=ext)
        if mode != "only-scan":
            cls.mkdirs_()
            cls._download_th(finfo_dl)

    @classmethod
    def _scan(cls, src_id: str, dst_dir: str, cred_dir: str, mode: str, ext):
        print("Getting files info...")

        cls.dst_root = dst_dir
        cls.service = build("drive", "v3", credentials=Download._get_creds(cred_dir))
        pl.Config.set_fmt_str_lengths(300)

        cls.finfo, cls.dirs = Download.get_finfo(service=cls.service, id=src_id, ext=ext)

        finfo_now = pl.DataFrame(
            cls.finfo,
            orient="row",
            schema={"id": pl.Utf8, "modifiedTime": pl.Float64, "path": pl.Utf8, "size": pl.UInt64},
        )

        # definition for _multi_progress
        cls.finfo_fname = os.path.join(dst_dir, "_finfo_.csv")
        cls.finfo_read = None

        if mode == "only-scan" or mode == "all-anyway":
            finfo_dl = finfo_now

        else:
            # ===== TODO =====

            # https://stackoverflow.com/questions/19686430/is-google-drive-file-id-unique-globally
            # assert finfo_now.shape[0] == len(set(finfo_now["id"])) # id is unique

            if os.path.exists(cls.finfo_fname):
                # https://docs.kanaries.net/ja/tutorials/Polars/polars-read-csv
                # read_csv()は、CSVファイルを読み込み、すべてのデータをメモリにロードするシンプルな関数です。 一方、scan_csv()は遅延的に動作するため、collect()が呼び出されるまでデータを読み込みません。
                cls.finfo_read = pl.read_csv(
                    cls.finfo_fname, has_header=False, new_columns=["id", "modifiedTime", "path"]
                )  # finfo_prev

                # print(finfo_prev)
                print(finfo_now)
                print(human_readable_byte(finfo_now["size"].sum()))

                print("\nNew files")
                finfo_new = finfo_now.filter(
                    finfo_now["id"].is_in(set(finfo_now["id"]) - set(cls.finfo_read["id"]))
                )
                print(finfo_new)
                print(human_readable_byte(finfo_new["size"].sum()))

                print("\nUpdated")
                # same as apply set(finfo_now["id"]) & set(finfo_prev["id"]) both
                finfo_now_prev = finfo_now.filter(finfo_now["id"].is_in(cls.finfo_read["id"]))
                finfo_prev_now = cls.finfo_read.filter(cls.finfo_read["id"].is_in(finfo_now["id"]))
                assert (finfo_now_prev["id"] == finfo_prev_now["id"]).all()  # 順序入れ変えに対してはfalseが出る
                finfo_updated = finfo_now_prev.filter(
                    finfo_prev_now["modifiedTime"] + 2 < finfo_now_prev["modifiedTime"]
                )
                print(human_readable_byte(finfo_updated["size"].sum()))

                print("\nDeleated")
                finfo_del = cls.finfo_read.filter(
                    cls.finfo_read["id"].is_in(set(cls.finfo_read["id"]) - set(finfo_now["id"]))
                )
                pprint(finfo_del)

                finfo_dl = finfo_now.filter(
                    finfo_now["id"].is_in(set(finfo_new["id"]) | set(finfo_updated["id"]))
                )
            else:
                finfo_dl = finfo_now

        print("Done")

        print("⭐⭐⭐ Download table ⭐⭐⭐")
        print(finfo_dl)
        print(human_readable_byte(finfo_dl["size"].sum()))

        return finfo_dl

    @classmethod
    def _download_th(cls, finfo_dl: pl.DataFrame):
        # EXE = threading.Thread
        EXE = mp.Process

        prog_queue = mp.Queue()
        ps: List[EXE] = []
        N = cls.num_workers
        L = finfo_dl.shape[0]

        idx = 0
        for n in range(N):  # なぜかctrl+cが効く
            # https://qiita.com/keisuke-nakata/items/c18cda4ded06d3159109
            span = (L + n) // N
            # print(idx, idx + span)
            df = finfo_dl[idx : idx + span]
            idx += span
            # print(n, df.shape)
            ps.append(
                EXE(
                    target=Download._worker,
                    args=(n, prog_queue, copy.deepcopy(cls.service), df, cls.dst_root),
                )
            )

        for p in ps:
            p.start()

        cls._multi_progress(N, prog_queue, finfo_dl)

    @staticmethod
    def _worker(n: int, prog_queue: mp.Queue, service, df: pl.DataFrame, dst_root: str):
        N = df.shape[0]
        for i in range(N):
            prog_item = dict(n=n, per=i / N, done=False, path=df.item(i, "path"))
            prog_queue.put(prog_item)
            Download._download_file(
                service=service,
                file_id=df.item(i, "id"),
                file_path=os.path.join(dst_root, df.item(i, "path")),
                prog_queue=prog_queue,
                prog_item={**prog_item, "N": N},
            )
            prog_queue.put(
                dict(n=n, per=(i + 1) / N, done=False, path="", done_id=df.item(i, "id"))
            )

        prog_queue.put(dict(n=n, per=1, done=True, path=""))  # to id

    @classmethod
    def _multi_progress(cls, N: int, prog_queue: mp.Queue, finfo_dl: pl.DataFrame):
        print("--- workers ---")

        per_all = np.zeros(N)
        done_all = np.full(N, False)
        path_all = [""] * N
        file_per_all = [None] * N

        time_start = time.perf_counter()

        def _printer():
            for n, p in enumerate(per_all):
                if file_per_all[n] != None:
                    more = f" {file_per_all[n] * 100: 6.2f} %"
                else:
                    more = ""

                print(Prompt.del_line + f"{n+1:2d}: {p * 100:6.2f} %  {path_all[n]}" + more)
            print(
                Prompt.del_line
                + f"Total: {per_all.sum() / N * 100:6.2f} %  Elapsed: {s2dhms_str(time.perf_counter() - time_start)}"
            )

        _printer()
        while not done_all.all():
            prog_item: dict = prog_queue.get()  # block
            n = prog_item["n"]
            per_all[n] = prog_item["per"]
            done_all[n] = prog_item["done"]
            path_all[n] = prog_item.get("path", "")
            file_per_all[n] = prog_item.get("file_per", None)

            print(Prompt.cursor_up(N + 2))
            _printer()

            done_id = prog_item.get("done_id", None)
            if done_id != None:
                with open(cls.finfo_fname, mode="ab") as f:
                    if cls.finfo_read is not None:
                        pass

                    finfo_dl[:, ["id", "modifiedTime", "path"]].filter(
                        pl.col("id") == done_id
                    ).write_csv(f, has_header=False)

    @staticmethod
    def get_finfo(service, id, ext):
        """
        Non Blocking
        """

        finfo = []

        def collect_inner(item, p):
            finfo.append(
                [
                    item["id"],
                    datetime.fromisoformat(item["modifiedTime"].replace("Z", "")).timestamp(),
                    p,
                    int(item["size"]),
                ]
            )

        def collect_finfo(item, p):
            if ext is None:
                collect_inner(item, p)
            elif os.path.splitext(item["name"])[1][1:] in ext:
                collect_inner(item, p)

        fields = "id, name, mimeType, modifiedTime, size"

        try:
            c = service.files().get(fileId=id, fields=fields).execute()
            p = c["name"]

            stack = []  # DFS
            stack.append((c, p))
            dirs = []

            while stack:
                c, p = stack.pop()

                if c["mimeType"] == "application/vnd.google-apps.folder":  # is not leaf (dir)
                    dirs.append(p)

                    id = c["id"]
                    results = (
                        service.files()
                        .list(
                            q=f"'{id}' in parents and trashed = false",
                            fields=f"nextPageToken, files({fields})",
                            pageSize=1000,
                        )
                        .execute()
                    )
                    cc = results.get("files", [])

                    for c_ in reversed(cc):
                        stack.append((c_, os.path.join(p, c_["name"])))
                else:  # leaf (file)
                    collect_finfo(c, p)

        except HttpError as error:
            print(f"An error occurred: {error}")

        return finfo, dirs

    @classmethod
    def mkdirs_(cls):
        for p in cls.dirs:
            os.makedirs(os.path.join(cls.dst_root, p), exist_ok=True)

    @staticmethod
    def _get_creds(cred_root=os.curdir):
        # If modifying these scopes, delete the file token.json.
        # SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
        SCOPES = ["https://www.googleapis.com/auth/drive"]

        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.

        token_p = os.path.join(cred_root, "token.json")
        cred_p = os.path.join(cred_root, "credentials.json")

        if os.path.exists(token_p):
            creds = Credentials.from_authorized_user_file(token_p, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(cred_p, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_p, "w") as token:
                token.write(creds.to_json())

        return creds

    @staticmethod
    def _download_file(service, file_id, file_path, prog_queue: mp.Queue, prog_item: list):
        """Blocking"""
        assert type(file_id) == str
        assert type(file_path) == str

        per_first = prog_item["per"]
        N = prog_item["N"]

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            prog_item["per"] = per_first + status.progress() / N
            prog_queue.put({**prog_item, "file_per": status.progress()})
            # print(f"{status.progress() * 100:.1f} %")
        fh.seek(0)
        with open(file_path, "wb") as f:
            f.write(fh.read())
        # print("Done")


if __name__ == "__main__":
    main()
