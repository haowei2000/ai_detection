import os


class TempProxy:
    def __init__(self):
        self.original_HTTP = os.environ.get("HTTP_PROXY", "")
        self.original_HTTPS = os.environ.get("HTTPS_PROXY", "")
        self.original_http = os.environ.get("http_proxy", "")
        self.original_https = os.environ.get("https_proxy", "")

    def start_proxy(self, proxy_url):
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url

    def reset_proxy(self):
        os.environ["HTTP_PROXY"] = self.original_HTTP
        os.environ["HTTPS_PROXY"] = self.original_HTTPS
        os.environ["http_proxy"] = self.original_http
        os.environ["https_proxy"] = self.original_https
