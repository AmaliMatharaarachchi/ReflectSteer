"""Worker-only HTTP reliability layer for the existing wikipedia client."""
import time

import requests


class WikipediaHTTPError(RuntimeError):
    pass


class WikipediaHTTP:
    def __init__(self, log, session=None, sleep=time.sleep):
        self.log = log
        self.session = session or requests.Session()
        self.sleep = sleep
        self.deadline = None

    def request(self, params):
        params = dict(params)
        params.setdefault('action', 'query')
        params['format'] = 'json'
        for attempt in range(1, 4):
            remaining = self.deadline - time.monotonic() if self.deadline else 90
            if remaining <= 0:
                raise WikipediaHTTPError('Wikipedia search exceeded its 90-second HTTP budget')
            diagnostic = {'attempt': attempt, 'params': params}
            retryable = True
            wait = 2 ** attempt
            try:
                response = self.session.get(
                    'https://en.wikipedia.org/w/api.php', params=params,
                    headers={'User-Agent': 'ReflectSteerResearch/1.0 (HotPotQA reproduction; python-requests)'},
                    timeout=min(15, remaining))
                diagnostic.update(status=response.status_code,
                                  content_type=response.headers.get('Content-Type', ''),
                                  url=response.url)
                retryable = response.status_code in (429, 500, 502, 503, 504)
                if response.status_code != 200:
                    raise WikipediaHTTPError(f'HTTP {response.status_code}')
                # Some gateways return HTML (including with HTTP 200).
                retryable = True
                payload = response.json()
                if 'error' in payload:
                    retryable = payload['error'].get('code') in ('maxlag', 'ratelimited', 'readonly')
                    raise WikipediaHTTPError(str(payload['error']))
                self.log({**diagnostic, 'ok': True})
                return payload
            except (requests.RequestException, ValueError, WikipediaHTTPError) as exc:
                diagnostic.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                if 'response' in locals():
                    diagnostic['body_excerpt'] = response.text[:500]
                    try:
                        wait = max(wait, float(response.headers.get('Retry-After', 0)))
                    except ValueError:
                        pass
                self.log(diagnostic)
                if not retryable or attempt == 3:
                    raise WikipediaHTTPError(str(diagnostic)) from exc
                remaining = self.deadline - time.monotonic() if self.deadline else 90
                if wait >= remaining:
                    raise WikipediaHTTPError(str(diagnostic)) from exc
                self.sleep(wait)
            finally:
                # Do not attach the preceding response to a subsequent timeout.
                if 'response' in locals():
                    del response


class ReliableWikipedia:
    """Keep page resolution and suggestion behavior in the original docstore."""
    def __init__(self, log):
        import wikipedia.wikipedia as client
        from langchain_community.docstore.wikipedia import Wikipedia
        self.http = WikipediaHTTP(log)
        # Runtime hook in this dedicated worker; no dependency files are edited.
        client._wiki_request = self.http.request
        self.store = Wikipedia()

    def search(self, query):
        self.http.deadline = time.monotonic() + 90
        try:
            return self.store.search(query)
        finally:
            self.http.deadline = None
