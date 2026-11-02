"""Wikipedia docstore transport over shared storage; no GPU-side networking."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import uuid


class RelayFailure(BaseException):
    """Abort rather than let upstream treat infrastructure failure as evidence."""


def publish(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


class RelayDocstore:
    def __init__(self, directory, timeout=600):
        self.directory = Path(directory)
        self.timeout = timeout
        for name in ('requests', 'responses'):
            (self.directory / name).mkdir(parents=True, exist_ok=True)

    def request(self, operation, query=''):
        request_id = uuid.uuid4().hex
        response = self.directory / 'responses' / (request_id + '.json')
        publish(self.directory / 'requests' / (request_id + '.json'),
                {'operation': operation, 'query': query, 'created': time.time()})
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if response.exists():
                result = json.loads(response.read_text(encoding='utf-8'))
                if result['kind'] == 'error':
                    raise RelayFailure('Wikipedia worker failed: ' + result['error'])
                return result
            time.sleep(1)
        raise RelayFailure(f'Wikipedia relay timed out after {self.timeout}s: {request_id}')

    def ping(self):
        self.request('ping')

    def search(self, query):
        from langchain_core.documents import Document
        result = self.request('search', query)
        if result['kind'] == 'document':
            return Document(page_content=result['text'], metadata=result['metadata'])
        return result['text']


def serve_once(directory, store):
    """Single worker per directory. Retain requests/responses as retrieval logs."""
    directory = Path(directory)
    for request in sorted((directory / 'requests').glob('*.json')):
        response = directory / 'responses' / request.name
        if response.exists():
            continue
        data = json.loads(request.read_text(encoding='utf-8'))
        try:
            if data['operation'] == 'ping':
                result = {'kind': 'pong'}
            elif data['operation'] == 'search':
                value = store.search(data['query'])
                if isinstance(value, str):
                    result = {'kind': 'text', 'text': value}
                else:
                    result = {'kind': 'document', 'text': value.page_content,
                              'metadata': value.metadata}
            else:
                raise ValueError('Unknown relay operation')
        except Exception as exc:
            result = {'kind': 'error', 'error': f'{type(exc).__name__}: {exc}'}
        result['served_at'] = time.time()
        publish(response, result)
        print(json.dumps({'request': request.stem, 'operation': data['operation'],
                          'query': data['query'], 'kind': result['kind']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=3)
    args = parser.parse_args()
    from wiki_http import ReliableWikipedia
    socket.setdefaulttimeout(30)
    RelayDocstore(args.directory)
    # OS-exclusive lock prevents two workers serving the same directory.
    lock = args.directory / 'worker.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.close(descriptor)
        deadline = time.monotonic() + args.hours * 3600
        def http_log(event):
            event['time'] = time.time()
            with (args.directory / 'http.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(event, ensure_ascii=False) + '\n')
            if not event['ok']:
                print('Wikipedia HTTP: ' + json.dumps(event), flush=True)

        store = ReliableWikipedia(http_log)
        print('Wikipedia worker ready: ' + str(args.directory), flush=True)
        while time.monotonic() < deadline:
            serve_once(args.directory, store)
            time.sleep(1)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
